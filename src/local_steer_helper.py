import torch
import time

from tqdm import tqdm
from typing import Any
from nnsight.intervention.base import InterventionProxy
from sklearn.feature_selection import mutual_info_classif

from functools import partial
from src.utils import get_GPU_mem_str
from src.wrappers import NNsightWrapper, SAEWrapper
from rich import print

MAX_INP_TOK = 7000


def inverv_fun_mutual_info(
    sae: SAEWrapper,                            # partial call
    edit_latents: torch.Tensor,                 # partial call
    noedit_latents: torch.Tensor,               # partial call
    alpha: float,                               # partial call
    activations: InterventionProxy,             # runtime call
):

    original_device = activations.device
    activations = activations.to(sae.device)

    features: torch.Tensor = sae.encode(activations).detach()
    # reconstructed: torch.Tensor = sae.decode(features).detach()
    # error = activations - reconstructed

    # ########## old inefficient loop ###########
    # z_edit = torch.zeros_like(features)
    # z_noedit = torch.zeros_like(features)
    # for lat_idx in range(features.shape[-1]):
    #     if edit_latents[lat_idx] > 0:
    #         z_edit[0, 0, lat_idx] = max(edit_latents[lat_idx] - features[:, -1, lat_idx], 0)
    #     if noedit_latents[lat_idx] > 0:
    #         z_noedit[0, 0, lat_idx] = min(noedit_latents[lat_idx], features[:, -1, lat_idx])
    # z_edit_old = z_edit.clone()
    # z_noedit_old = z_noedit.clone()

    z_edit = torch.zeros_like(features)
    z_noedit = torch.zeros_like(features)

    last_features_first_batch = features[0, -1, :] # Shape: (feature_dim,)

    # create a boolean mask based on conditions
    edit_mask = edit_latents > 0
    noedit_mask = noedit_latents > 0

    # Calculate values for z_edit where edit_mask is True
    if torch.any(edit_mask):
        diff = edit_latents[edit_mask] - last_features_first_batch[edit_mask]
        # Use torch.clamp(min=0) which is equivalent to max(..., 0) element-wise
        edit_values = torch.clamp(diff, min=0)
        z_edit[0, 0, edit_mask] = edit_values.to(z_edit.dtype)

    # Calculate values for z_noedit where noedit_mask is True
    if torch.any(noedit_mask):
        # Use torch.minimum for element-wise min
        min_vals = torch.minimum(noedit_latents[noedit_mask], last_features_first_batch[noedit_mask])
        # Assign calculated values to the slice z_noedit[0, 0, :] where mask is True
        z_noedit[0, 0, noedit_mask] = min_vals.to(z_edit.dtype)

    act_edit = sae.decode(
        z_edit
    ).detach()

    act_noedit = sae.decode(
        z_noedit
    ).detach()

    new_act = activations + ((act_edit - act_noedit) * alpha)
    new_act = new_act.to(original_device)


    return new_act



def custom_online_interv_fun(
    sae: SAEWrapper,                        # partial call
    good_features: list[int],               # partial call
    bad_features: list[int],                # partial call
    worse_features: list[int],              # partial call
    logits_to_use: dict[int, float],        # partial call
    activations: InterventionProxy,         # runtime call
) -> InterventionProxy:
    """
    Function to apply the SAE intervention on the activations.
    Args:
        hook_point (str): The hook point to use for extracting features.
        activations (torch.Tensor): The activations to use.
        sae (SparseAutoEncoder): The SAE model to use.
    Returns:
        torch.Tensor: The new activations after applying the SAE intervention.

    - for each gen token:
        - get active latents
        - if (good - (bad U worse)) are in active latents:
            - il modello vuole generare un token according to PE
        - if (bad) are in active latents:
            - il modello vuole generare altro NOT according to PE
            - ablate bad latents
        - if (worse) latents are active, the model is copying MT while it should edit instead
            - steer the model with good latents here
        - if (good and bad) are in active latents:
    """

    original_device = activations.device
    activations = activations.to(sae.device)

    features: torch.Tensor = sae.encode(activations).detach()
    reconstructed: torch.Tensor = sae.decode(features).detach()
    error = activations - reconstructed

    # consider only the current generation step
    good_features_active = (features[:, -1, good_features] > 0).any().item() # there are good features active
    bad_features_active = (features[:, -1, bad_features] > 0).any().item() # there are bad features active
    worse_features_active = (features[:, -1, worse_features] > 0).any().item() # there are worse features active

    if good_features_active and not (bad_features_active or worse_features_active):
        # il modello vuole generare un token according to PE
        pass
    elif bad_features_active:
        # il modello vuole generare altro NOT according to PE
        # ablate bad latents
        features[:, -1, bad_features] = 0.0
    elif worse_features_active:
        # il modello sta copiando MT mentre invece dovrebbe editare
        # steer the model with good features
        features[:, -1, good_features] = torch.tensor([logits_to_use[latent] for latent in good_features], dtype=sae.dtype, device=sae.device)
    else:
        # il modello sta facendo altro, lasciagli fare altro (no intervention)
        return activations.to(original_device)
        # features[:, -1, good_features] = features[:, -1, bad_features]

    final_reconstruct = sae.decode(features) + error.to(sae.device)     # add the error back
    final_reconstruct = final_reconstruct.to(original_device)

    return final_reconstruct




def sae_interv_function(
    sae: SAEWrapper,                 # partial call
    features_to_intervene: list[int],       # partial call
    logits_or_alpha: list[float] | float,   # partial call
    activations: InterventionProxy,         # runtime call
) -> InterventionProxy:
    """
    Function to apply the SAE intervention on the activations.
    Args:
        hook_point (str): The hook point to use for extracting features.
        activations (torch.Tensor): The activations to use. Shape: (batch_size, seq_len[1], hidden_dim)
        sae (SparseAutoEncoder): The SAE model to use.
    Returns:
        torch.Tensor: The new activations after applying the SAE intervention.
    """

    original_device = activations.device
    activations = activations.to(sae.device)

    features = sae.encode(activations).detach()
    reconstructed = sae.decode(features).detach()

    error = activations - reconstructed     # to compensate

    # intervention features
    if isinstance(logits_or_alpha, list):
        assert len(features_to_intervene) == len(logits_or_alpha), "features_to_intervene and logits_or_alpha must have the same length."

        logits = logits_or_alpha
        features[:, :, features_to_intervene] = torch.tensor(logits, dtype=sae.dtype, device=sae.device)
    elif isinstance(logits_or_alpha, float):
        alpha = logits_or_alpha
        # make sure the features to intervene are > 0, if not, set them to the min value
        null_features = features[:, :, features_to_intervene] <= 0
        features[:, :, null_features] = torch.min(features[:, :, features_to_intervene][~null_features])
        features[:, :, features_to_intervene] *= alpha
    else:
        raise Exception(f"logits_or_alpha must be a list of logits (list[float]) or a float alpha. {logits_or_alpha=} was given.")

    final_reconstruct = sae.decode(features) + error.to(sae.device)     # add the error back
    final_reconstruct = final_reconstruct.to(original_device)

    return final_reconstruct


def get_features_local(
    model: NNsightWrapper,
    hook_point: str,
    sae: SAEWrapper,
    max_new_tokens: int,
    train_data: list[dict],
    PROMPT_TEMPL: str,
    LANG_MAP: dict[str, str],
) -> list[list[dict[str, Any]]]:
    """
    Function to run the model and extract features while generating.
    Args:
        model_name (str): The name of the model to use.
        hook_point (str): The hook point to use for extracting features.
        max_new_tokens (int): The maximum number of tokens to generate.
        train_data (list[dict]): The training data.
        PROMPT_TEMPL (str): The prompt template to use.
        LANG_MAP (dict[str, str]): The language map to use.
    Returns:
        A list of features while generating.
        ```python
        features_while_generating: list[        # for every example in train
            list[                               # for every gen token
                dict[str, Any]                  # {"features": [list of features idxs], "logits": [list of corresponding logits], "rec_token": {"gen": gen_token, "pe": pe_token, "mt": mt_token}}
            ]
        ]
        ```
    """

    features_while_generating: list[list[dict[str, Any]]] = []
    for example in (pbar := tqdm(train_data, desc=f"Running local examples | {get_GPU_mem_str()}", leave=True)):
        # prepare prompt
        prompt = PROMPT_TEMPL.format(
            src_text=example["src_text"],
            lang=LANG_MAP[example["tgt_lang"]],
            mt_text=example["mt_text"],
        )

        # tokenize
        inp_tokens: torch.Tensor = model.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors='pt',
        )
        pe_tokens = model.tokenizer(example["pe_text"], return_tensors='pt', add_special_tokens=False).input_ids
        mt_tokens = model.tokenizer(example["mt_text"], return_tensors='pt', add_special_tokens=False).input_ids

        # run forward pass anche cache activations
        pbar.set_description(f"Caching {hook_point} | {get_GPU_mem_str()}")
        cache_activations, tokens_type = model.cache_forward_forced(
            inp_tokens=inp_tokens,
            pe=pe_tokens,
            mt=mt_tokens,
            hook_points=[hook_point],
            max_new_tokens=max_new_tokens,
        )
        pbar.set_description(f"Caching {hook_point} | {get_GPU_mem_str()}")

        # aggregate cache and add batchsize
        aggregated_cache = torch.vstack(
            [cache_activations[step][hook_point] for step in range(len(cache_activations))]
        ).unsqueeze(0)

        # get sae's features
        with torch.no_grad():
            features = sae.encode(
                aggregated_cache.to(sae.device)
            ).detach() # (1, gen_tokens, 65k)
        
        # store features and info
        active_feat: list[dict[str, Any]]= []
        for features, token_t in zip(features.squeeze(), tokens_type):
            index_active_features = (features > 0).nonzero().flatten().tolist()
            active_feat.append({
                "features": index_active_features,
                "logits": [features[i].item() for i in index_active_features],
                "rec_token": token_t,
            })
        features_while_generating.append(active_feat)

    return features_while_generating




def get_contrastive_features_local(
    model: NNsightWrapper,
    hook_point: str,
    sae: SAEWrapper,
    max_new_tokens: int,
    train_data: list[dict],
    val_data: list[dict],
    PROMPT_TEMPL: str,
    LANG_MAP: dict[str, str],
    select_top_k: int = 30,
    seed: int = 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function to run the model and extract features while generating.
    Args:
        model_name (str): The name of the model to use.
        hook_point (str): The hook point to use for extracting features.
        max_new_tokens (int): The maximum number of tokens to generate.
        train_data (list[dict]): The training data.
        PROMPT_TEMPL (str): The prompt template to use.
        LANG_MAP (dict[str, str]): The language map to use.
    Returns:
        A list of features while generating.
        ```python
        features_while_generating: list[        # for every example in train
            list[                               # for every gen token
                dict[str, Any]                  # {"features": [list of features idxs], "logits": [list of corresponding logits], "rec_token": {"gen": gen_token, "pe": pe_token, "mt": mt_token}}
            ]
        ]
        ```
    """
    icl_conversation_edit = []
    icl_conversation_noedit = []
    for icl_example in train_data:
        prompt_std = PROMPT_TEMPL.format(
            src_text=icl_example["src_text"],
            lang=LANG_MAP[icl_example["tgt_lang"]],
            mt_text=icl_example["mt_text"],
        )
        icl_conversation_edit.append({"role": "user", "content": prompt_std})
        icl_conversation_edit.append({"role": "assistant", "content": icl_example["pe_text"]})
        icl_conversation_noedit.append({"role": "user", "content": prompt_std})
        icl_conversation_noedit.append({"role": "assistant", "content": icl_example["mt_text"]})

    # test inp length
    tokenized_inp = model.tokenizer.apply_chat_template(
        icl_conversation_edit,
        return_tensors="pt",
    )
    if tokenized_inp.shape[-1] > MAX_INP_TOK:
        removed_examples = 0
        print(f"[X] Warning: the prompt is too long ({tokenized_inp.shape[-1]} tokens).")
        print("Trying to reduce the number of shots or use a smaller model.")
        print("Removed examples: ", end="")
        for i in range(0, len(icl_conversation_edit), 2):
            print(f'{i // 2}', end=', ')
            base_conversation = icl_conversation_edit[i:]
            tokenized_inp = model.tokenizer.apply_chat_template(
                base_conversation,
                return_tensors="pt",
            )
            if tokenized_inp.shape[-1] < MAX_INP_TOK:
                removed_examples = i
                print('ok', end=' ')
                break
        icl_conversation_edit = icl_conversation_edit[removed_examples:]
        icl_conversation_noedit = icl_conversation_noedit[removed_examples:]
        print(f"| now icl is {len(icl_conversation_edit) // 2} examples long")

    features_while_generating: list[list[dict[str, Any]]] = []
    tot_edit_feat = []
    tot_noedit_feat = []
    tot_edit_logits = []
    tot_noedit_logits = []
    for example in (pbar := tqdm(val_data, desc=f"Running local contrastive examples | {get_GPU_mem_str()}", leave=True)):
        # prepare prompt
        prompt = PROMPT_TEMPL.format(
            src_text=example["src_text"],
            lang=LANG_MAP[example["tgt_lang"]],
            mt_text=example["mt_text"],
        )
        # tokenize
        inp_tokens_edit: torch.Tensor = model.tokenizer.apply_chat_template(
            conversation=icl_conversation_edit + [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors='pt',
        )
            
        inp_tokens_noedit: torch.Tensor = model.tokenizer.apply_chat_template(
            conversation=icl_conversation_noedit + [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors='pt',
        )

        # run forward pass anche cache activations
        pbar.set_description(f"Caching {hook_point} | {get_GPU_mem_str()}")
        # TODO: review this part, now the activation might be different
        # the result should not change but it might
        edit_cache_activations = model.cache_forward(
            inp_tokens=inp_tokens_edit,
            hook_points=[hook_point],
            max_new_tokens=max_new_tokens,
        )       # list[dict[hook(s): activations], ..., dict[hook(s): activations]]

        noedit_cache_activations = model.cache_forward(
            inp_tokens=inp_tokens_noedit,
            hook_points=[hook_point],
            max_new_tokens=max_new_tokens,
        )
        pbar.set_description(f"Caching {hook_point} | {get_GPU_mem_str()}")

        # aggregate cache and add batchsize
        edit_aggregated_cache = torch.stack(
            [edit_cache_activations[step][hook_point] for step in range(len(edit_cache_activations))]
        ).unsqueeze(0)      # shape: (1, gen_tokens, 4k)
        noedit_aggregated_cache = torch.stack(
            [noedit_cache_activations[step][hook_point] for step in range(len(noedit_cache_activations))]
        ).unsqueeze(0)      # shape: (1, gen_tokens, 4k)

        # get sae's features, but keep only the first generated token
        with torch.no_grad():
            edit_features = sae.encode(
                edit_aggregated_cache.to(sae.device)
            ).detach().squeeze()[0] # (send in input: (4k), get active features: (65k))
            noedit_features = sae.encode(
                noedit_aggregated_cache.to(sae.device)
            ).detach().squeeze()[0]
        # NOTE: first token is expected to encode the edit/nonedit behaviour the model will take

        boolean_edit_features = (edit_features > 0).to(torch.int)
        boolean_noedit_features = (noedit_features > 0).to(torch.int)

        tot_edit_feat.append(boolean_edit_features)
        tot_noedit_feat.append(boolean_noedit_features)

        tot_edit_logits.append(edit_features)
        tot_noedit_logits.append(noedit_features)

    # cast to tensor features and logits
    tot_edit_feat = torch.stack(tot_edit_feat).to(torch.int)        # shape: (val_size(20), 65k)
    tot_noedit_feat = torch.stack(tot_noedit_feat).to(torch.int)
    tot_edit_logits = torch.stack(tot_edit_logits).to(torch.float)  # shape: (val_size, 65k)
    tot_noedit_logits = torch.stack(tot_noedit_logits).to(torch.float)

    mutual_info = mutual_info_classif(
        X=torch.cat([tot_edit_feat.cpu(), tot_noedit_feat.cpu()], dim=0),
        y=[1] * len(tot_edit_feat) + [0] * len(tot_noedit_feat),     # the model here expect to edit or look for edits (even if it doesn't edit the final sentence)
        random_state=seed,
    )       # 65k, amount of mutual info for each feature being responsible for the edit

    print(f"mutual info: {mutual_info}")
    # get top-k features idxs according to the mutual info
    top_k_features = mutual_info.argsort()[-select_top_k:][::-1]
    print(f"top {select_top_k} features according to their mi: {top_k_features}")
    print(f"their mi: {[round(mutual_info[i], 2) for i in top_k_features]}")

    prob_edit_act = tot_edit_feat.sum(0) / len(tot_edit_feat)     # (65k,)
    prob_noedit_act = tot_noedit_feat.sum(0) / len(tot_noedit_feat)
    expected_values_edit = sum(tot_edit_logits * prob_edit_act)
    expected_values_noedit = sum(tot_noedit_logits * prob_noedit_act)

    expected_values = {}
    final_latents_edit = torch.zeros_like(expected_values_edit)
    final_latents_noedit = torch.zeros_like(expected_values_noedit)
    for top_feat in top_k_features:
        if expected_values_edit[top_feat] > expected_values_noedit[top_feat]:
            final_latents_edit[top_feat] = tot_edit_logits[:, top_feat].mean()

        if expected_values_edit[top_feat] < expected_values_noedit[top_feat]:
            final_latents_noedit[top_feat] = tot_noedit_logits[:, top_feat].mean()

    print(f"features to intervene: {top_k_features}")
    where_g0 = (final_latents_edit > 0).nonzero().flatten().tolist()
    print("expected values edit: (latent: exp_value) {exp_edits}".format(
        exp_edits={lat: expected_values_edit[lat].item() for lat in where_g0}
    ))
    where_g0 = (final_latents_noedit > 0).nonzero().flatten().tolist()
    print("expected values noedit: (latent: exp_value) {exp_noedits}".format(
        exp_noedits={lat: expected_values_noedit[lat].item() for lat in where_g0}
    ))

    print()
    print(f"final latents edit: {(final_latents_edit > 0).sum()}")      # spero che questo abbia almeno qualche latente, WOW ne ha 1
    print(f"final latents noedit: {(final_latents_noedit > 0).sum()}")  # e questo anche dai, WOW ne ha 5

    return final_latents_edit, final_latents_noedit





if __name__ == "__main__":


    train_data = [
        {
            # "unit_id": "qe4pe-main-eng-ita-5-1-no_highlight_t1",
            "src_text": "One word I simply can’t say properly is water...",
            "mt_text": "Una parola che non riesco propriamente a pronunciare è acqua...",
            "pe_text": "Una parola che non riesco proprio a pronunciare è acqua...",
            "tgt_lang": "ita",
            # "tot_chars_edits": 0,
            # "edit_ratio": 0.0,
            # "bin_number": 0
        }
    ]


    PROMPT_TEMPL = """\
    You are requested to perform edits to obtain a translation of publishable quality in fluent {lang}.

    The original sentence is: \"{src_text}\";
    The translation is: \"{mt_text}\"

    Remember to write only the translation, without any additional text or explanation.
    """


    LANG_MAP = {
        "ita": "Italian",
        "nld": "Dutch",
    }


    # # to get the features, run this function
    # get_features_local(
    #     model_name="meta-llama/Llama-3.1-8B-Instruct",
    #     hook_point="model.layers.19",
    #     max_new_tokens=512,
    #     train_data=train_data,
    #     PROMPT_TEMPL=PROMPT_TEMPL,
    #     LANG_MAP=LANG_MAP,
    # )



    model = NNsightWrapper(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit=False,
    )
    hook_point = "model.layers.19"

    prompt = PROMPT_TEMPL.format(
        src_text=train_data[0]["src_text"],
        lang=LANG_MAP[train_data[0]["tgt_lang"]],
        mt_text=train_data[0]["mt_text"],
    )

    inp_tokens = model.tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors='pt',
    )


    sae_name = "Llama-3.1-8B-Instruct-SAE-l19"
    sae: SAEWrapper = SAEWrapper(
        sae_name=sae_name,
        sae_layer=19,
        device=torch.device('cpu'),
    )

    interventions = {
        hook_point: partial(
            sae_interv_function,
            sae=sae,
            features_to_intervene=[4890, 1238, 38921],
            logits_or_alpha=200.0,
        )
    }
    
    # inp_tokens: Tensor,
    # interventions: dict[str, (InterventionProxy) -> Tensor],
    # max_new_tokens: int = 512
    out = model.set_forward(
        inp_tokens=inp_tokens,
        interventions=interventions,
        max_new_tokens=512,

    )

    print(model.tokenizer.batch_decode(out, skip_special_tokens=False))


