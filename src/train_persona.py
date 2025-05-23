import fire
import os
import json
import torch
import random
import warnings
import pyreft
import time

from typing import Any
from functools import partial
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    PreTrainedModel, PreTrainedTokenizer, 
    BatchEncoding, TrainingArguments
)
from tqdm import tqdm
from rich import print
from dotenv import load_dotenv

from src.goodfire_api_helper import contrastive_api, generate_features
from src.local_steer_helper import get_features_local, get_contrastive_features_local, custom_online_interv_fun, inverv_fun_mutual_info
from src.utils import get_GPU_mem_str, latents_classificaiton, clean_txt
from src.wrappers import NNsightWrapper, SAEWrapper, SAE_MAPPER
from src.metric import eval_out

load_dotenv()
GOODFIRE_API_KEY = os.getenv("GOODFIRE_API")

PROMPT_TEMPL = """\
You are requested to check the fluency of the {lang} translation proposed below between the angular parentheses.
Edit the translation if you think it can be improved; otherwise, leave it as is if you think it's fluent.

The original sentence is: <{src_text}>;
The translation is: <{mt_text}>

Remember to write only the translation, without any additional text or explanation.
"""

MT_template = """\
Translate the following sentence between the angular parentheses into {lang}.

The original sentence is: <{src_text}>.

Remember to write only the translation, without any additional text or explanation.
"""

MT_template_explainations = """\
Translate the following sentence between the angular parentheses into {lang}.

Follow the following guidelines when translating:
{explanations}

The original sentence is: <{src_text}>.

Remember to write only the translation, without any additional text or explanation.
"""

# NOTE: since we are not using the qe4pe dataset, there is no need to ask the model to edit the translation
# about the missing {mt_text} in MT_template, the format function will just ignore the parameter
# (so no code changes in downstream functions)
# PROMPT_TEMPL = MT_template
TEMPLATE_MAP = {
    "translate": MT_template,
    "edit": PROMPT_TEMPL,
}

LANG_MAP = {
    "ita": "Italian",
    "nld": "Dutch",
    "eng": "English",
}

MAX_INP_TOK = 6000


def eval_and_save(
    output_results: list[dict],
    experiment: str,
    model_name: str,
    persona_name: str,
    template: str,
    seed: int,
    output_path: Path,
    base_experiment_path: Path,
    additional_info: str | None = None,
):
    
    # evaluate
    output_eval, metric_table = eval_out(
        output=output_results,
        fields=[
            ("original_pe", "output_clean"),    # how close the std mt to the pe
            ("original_mt", "original_pe"),
            ("original_mt", "output_clean"),    # distance between nllb and llama
        ],
        experiment_path=base_experiment_path,
        model_name=model_name,
    )

    # save
    if additional_info:
        eval_file_name = f"{persona_name}-{seed}-{experiment}-{template}-{additional_info}-eval.json"
    else:
        eval_file_name = f"{persona_name}-{seed}-{experiment}-{template}-eval.json"
    with open(output_path / eval_file_name, "w") as f:
        json.dump(output_eval, f, indent=4, ensure_ascii=False)
    # print infos
    print(f"\n\n##############################")
    print(f"Experiment: {persona_name} | {experiment} | {model_name} | {template} | {seed} | {additional_info}")
    print(f"\n\n########## DONE ##############")
    print(f"\n\n##############################")
    # save the metric table
    metric_table.title = f"Metrics: {experiment} - {template}"
    with open(output_path / f"metric_tables-{seed}.md", "a") as f:
        f.write(f"### {model_name} | {experiment}\n")
        f.write(f"#### {persona_name} | {seed} | {template}\n")
        f.write(f"#### {additional_info}\n")
        # current time
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write(f"#### {current_time}\n")
        print(metric_table, file=f)
        f.write("\n\n")
    



def load_data(
    model_name: str,            # used to get the correct MT
    persona_name: str,
    lang: str,
    oth_persona: str | None,
):
    train_data_path = Path(f"data/train/{lang}/{persona_name}.json")
    # load data
    with open(train_data_path, "r") as f:
        train_data = json.load(f)
    with open(f'data/test/{lang}/{persona_name}.json', "r") as f:
        test_data = json.load(f)
    with open(f'data/val/{lang}/{persona_name}.json', "r") as f:
        val_data = json.load(f)

    # random train data
    parent_path = Path(train_data_path).parent
    all_oth_train_data = []
    oth_annotators = [ann for ann in os.listdir(parent_path)[:10] if ann.split('.')[0] != persona_name]     # (no more than 10 annotators)
    for annotator in oth_annotators:
        with open(parent_path / annotator, "r") as f:
            all_oth_train_data += json.load(f)
    # sample according to the len of train_data
    random_train_data = random.sample(all_oth_train_data, len(train_data))

    if oth_persona:
        if oth_persona[-2] == 't':
            raise ValueError(f"Invalid persona name: {oth_persona}. You are probably using personas from the qe4pe dataset. Remove this line if data is parallel, at the moment we are not using it.")

        # load data
        with open(f"data/train/{lang}/{oth_persona}.json", "r") as f:
            oth_train_data = json.load(f)
        with open(f'data/test/{lang}/{oth_persona}.json', "r") as f:
            oth_test_data = json.load(f)
        with open(f'data/val/{lang}/{oth_persona}.json', "r") as f:
            oth_val_data = json.load(f)

        # replace every MT with the PE of the other persona
        # now the MT will be another translator
        # the classifier should learn to distinguish between the two translators 
        # and not the human vs MT translation
        for i in range(len(train_data)):
            train_data[i]["mt_text"] = oth_train_data[i]["pe_text"]
        for i in range(len(test_data)):
            test_data[i]["mt_text"] = oth_test_data[i]["pe_text"]
        for i in range(len(val_data)):
            val_data[i]["mt_text"] = oth_val_data[i]["pe_text"]
    else:
        # replace every MT with the model MT
        model_key = model_name.split('/')[-1]
        for i in range(len(train_data)):
            train_data[i]["mt_text"] = train_data[i][model_key]
        for i in range(len(test_data)):
            test_data[i]["mt_text"] = test_data[i][model_key]
        for i in range(len(val_data)):
            val_data[i]["mt_text"] = val_data[i][model_key]

    return train_data, val_data, test_data, random_train_data




def _generate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tokenized_inp: torch.Tensor | BatchEncoding,
    repetition_penalty: float = 1.2,
    return_decoded: bool = False,
    **kwargs
):

    original_input_len: int = tokenized_inp.shape[-1]

    tokenized_inp = tokenized_inp.to(model.device)

    with torch.no_grad():
        # Generate the output
        output: torch.LongTensor = model.generate(
            tokenized_inp,
            **kwargs,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )

    if return_decoded:
        # Decode the output
        decoded_output = tokenizer.decode(
            output.squeeze()[original_input_len:],
            skip_special_tokens=True
        )
        return decoded_output
    else:
        return output.squeeze()[original_input_len:]



def mt_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_data: list[dict],
    seed: int,
) -> list[dict]:
    # Get the zero-shot baseline (single <mt_text> and ask the model to generate the <pe_text> from the test set)

    output: list[dict] = []
    for example in (pbar := tqdm(test_data, desc="justMT baseline")):

        # Tokenize the input
        prompt: str = MT_template.format(
            src_text=example["src_text"],
            lang=LANG_MAP[example["tgt_lang"]],
        )

        tokenized_inp = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Generate the output
        generated_output = _generate_model(
            model=model,
            tokenizer=tokenizer,
            tokenized_inp=tokenized_inp,
            return_decoded=False,
            max_new_tokens=1024,
            do_sample=False,
        )

        output.append({
            "config_details": {},
            "unit_id": example["unit_id"],
            "original_src": example["src_text"],
            "original_mt": example["mt_text"],
            "original_pe": example["pe_text"],
            "input_clean": tokenizer.decode(tokenized_inp.squeeze(), skip_special_tokens=True),
            "output_clean": clean_txt(tokenizer.decode(generated_output.squeeze(), skip_special_tokens=True)),
        })

    return output



def zero_shot_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_data: list[dict],
    template_to_use: str,
    seed: int,
    ) -> list[dict]:
    # Get the zero-shot baseline (single <mt_text> and ask the model to generate the <pe_text> from the test set)

    output: list[dict] = []
    for example in (pbar := tqdm(test_data, desc="Zero-shot baseline")):

        # Tokenize the input
        prompt: str = template_to_use.format(
            src_text=example["src_text"],
            lang=LANG_MAP[example["tgt_lang"]],
            mt_text=example["mt_text"],
        )

        tokenized_inp = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Generate the output
        generated_output = _generate_model(
            model=model,
            tokenizer=tokenizer,
            tokenized_inp=tokenized_inp,
            return_decoded=False,
            max_new_tokens=1024,
            do_sample=False,
        )

        output.append({
            "config_details": {},
            "unit_id": example["unit_id"],
            "original_src": example["src_text"],
            "original_mt": example["mt_text"],
            "original_pe": example["pe_text"],
            "input_clean": prompt,
            "output_clean": clean_txt(tokenizer.decode(generated_output.squeeze(), skip_special_tokens=True)),
        })

    return output



def multi_shot_with_mt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_data: list[dict],
    test_data: list[dict],
    shots: int | str,
    template_to_use: str,
    seed: int,
    ) -> list[dict]:
    # Get the few-shot (shuffle!) baseline (multiple <mt_text> and <pe_text> from the train set and then ask the model to produce the final <pe_text> from the test set)

    if isinstance(shots, str):
        assert shots == "all", "shots must be an int or 'all'"
        shots: int = len(train_data)

    output: list[dict] = []

    # Shuffle the train data
    random.seed(seed)
    random.shuffle(train_data)      # in-place shuffle

    # Get the few-shot examples
    few_shot_examples = train_data[:shots]

    # build prompt
    base_conversation = []
    for shot in few_shot_examples:
        prompt = template_to_use.format(
            src_text=shot["src_text"],
            lang=LANG_MAP[shot["tgt_lang"]],
            mt_text=shot["mt_text"],
        ) 
        base_conversation.append(
            {"role": "user", "content": prompt}
        )
        base_conversation.append(
            {"role": "assistant", "content": shot["mt_text"]}
        )

    output: list[dict] = []
    for example in (pbar := tqdm(test_data, desc=f"Few-shot baseline (MT edition) | {get_GPU_mem_str()}")):

        conversation = base_conversation.copy()
        prompt = template_to_use.format(
            src_text=example["src_text"],
            lang=LANG_MAP[example["tgt_lang"]],
            mt_text=example["mt_text"],
        )
        conversation.append(
            {"role": "user", "content": prompt}
        )
        
        tokenized_inp = tokenizer.apply_chat_template(
            conversation, 
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Generate the output
        pbar.set_description(f"Few-shot baseline (MT edition) | {get_GPU_mem_str()}")
        generated_output = _generate_model(
            model=model,
            tokenizer=tokenizer,
            tokenized_inp=tokenized_inp,
            return_decoded=False,
            max_new_tokens=1024,
            do_sample=False,
        )
        pbar.set_description(f"Few-shot baseline (MT edition) | {get_GPU_mem_str()}")

        output.append({
            "config_details": {"shots": shots},
            "unit_id": example["unit_id"],
            "original_src": example["src_text"],
            "original_mt": example["mt_text"],
            "original_pe": example["pe_text"],
            "input_clean": prompt,
            "output_clean": clean_txt(tokenizer.decode(generated_output.squeeze(), skip_special_tokens=True)),
        })

    return output


def multi_shot_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_data: list[dict],
    test_data: list[dict],
    shots: int | str,
    template_to_use: str,
    seed: int,
    ) -> list[dict]:
    # Get the few-shot (shuffle!) baseline (multiple <mt_text> and <pe_text> from the train set and then ask the model to produce the final <pe_text> from the test set)

    if isinstance(shots, str):
        assert shots == "all", "shots must be an int or 'all'"
        shots: int = len(train_data)

    output: list[dict] = []

    # Shuffle the train data
    random.seed(seed)
    random.shuffle(train_data)

    # Get the few-shot examples
    few_shot_examples = train_data[:shots]

    # build prompt
    base_conversation = []
    for shot in few_shot_examples:
        prompt = template_to_use.format(
            src_text=shot["src_text"],
            lang=LANG_MAP[shot["tgt_lang"]],
            mt_text=shot["mt_text"],
        ) 
        base_conversation.append(
            {"role": "user", "content": prompt}
        )
        base_conversation.append(
            {"role": "assistant", "content": shot["pe_text"]}
        )
    # try to tokenize to check length
    tokenized_inp = tokenizer.apply_chat_template(
        base_conversation,
        return_tensors="pt",
    )
    if tokenized_inp.shape[-1] > MAX_INP_TOK:
        print(f"[X] Warning: the prompt is too long ({tokenized_inp.shape[-1]} tokens).")
        print("Trying to reduce the number of shots")
        print("Removed examples: ", end="")
        for i in range(0, len(base_conversation), 2):
            print(f'{i // 2}', end=', ')
            base_conversation = base_conversation[i:]
            tokenized_inp = tokenizer.apply_chat_template(
                base_conversation,
                return_tensors="pt",
            )
            if tokenized_inp.shape[-1] < MAX_INP_TOK:
                print('ok', end=' ')
                break
        print(f"| now icl is {len(base_conversation) // 2} examples long")

    output: list[dict] = []
    for example in (pbar := tqdm(test_data, desc=f"Few-shot baseline | {get_GPU_mem_str()}")):
        conversation = base_conversation.copy()
        prompt = template_to_use.format(
            src_text=example["src_text"],
            lang=LANG_MAP[example["tgt_lang"]],
            mt_text=example["mt_text"],
        )
        conversation.append(
            {"role": "user", "content": prompt}
        )
        
        tokenized_inp: torch.Tensor = tokenizer.apply_chat_template(
            conversation, 
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Generate the output
        pbar.set_description(f"Few-shot baseline | {get_GPU_mem_str()}")
        generated_output = _generate_model(
            model=model,
            tokenizer=tokenizer,
            tokenized_inp=tokenized_inp,
            return_decoded=False,
            max_new_tokens=1024,
            do_sample=False,
        )
        pbar.set_description(f"Few-shot baseline | {get_GPU_mem_str()}")

        output.append({
            "config_details": {"shots": shots},
            "unit_id": example["unit_id"],
            "original_src": example["src_text"],
            "original_mt": example["mt_text"],
            "original_pe": example["pe_text"],
            "input_clean": prompt,
            "output_clean": clean_txt(tokenizer.decode(generated_output.squeeze(), skip_special_tokens=True)),
        })

    return output



def contrastive_remote(
    train_data: list[dict],
    test_data: list[dict],
    seed: int,
    output_path: Path,
    top_features: int = 2,
    steer_value: float = 0.6,
    template_to_use: str = PROMPT_TEMPL,
) -> list[dict]:
    
    # call API on the contrastive examples (train) to get the latents
    noedit_conversations = []
    edit_conversations = []
    for example in train_data:
        prompt = template_to_use.format(
            src_text=example["src_text"],
            lang=LANG_MAP[example["tgt_lang"]],
            mt_text=example["mt_text"],
        )
        noedit_conversations.append([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example["mt_text"]}
        ])
        edit_conversations.append([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example["pe_text"]}
        ])

    # call API
    default_features, edit_features = contrastive_api(
        default_chats=noedit_conversations,
        edit_chats=edit_conversations,
        n_feat=30,
    )
    # save features
    with open(output_path / "support" / f"{seed}-default_features.json", "w") as f:
        json.dump(default_features.json(), f, indent=4, ensure_ascii=False)
    with open(output_path / "support" / f"{seed}-edit_features.json", "w") as f:
        json.dump(edit_features.json(), f, indent=4, ensure_ascii=False)
    # You can create a feature group from the JSON data
    # using goodfire.features.features.Feature.from_json(data)
    # and group them into a goodfire.features.features.FeatureGroup.add(feature)
    
    # use features to generate on test data
    output: list[dict] = []
    for example in (pbar := tqdm(test_data, desc="Contrastive test", dynamic_ncols=True)):
        prompt = template_to_use.format(
            src_text=example["src_text"],
            lang=LANG_MAP[example["tgt_lang"]],
            mt_text=example["mt_text"],
        )

        generated_output = generate_features(
            chat=[{"role": "user", "content": prompt}],
            features=edit_features[0:top_features],
            seed=seed,
            steer_value=0.6,
            expected_gen_len=int(len(example["pe_text"]) * 1.3),    # up to 130% of the target length
        )

        output.append({
            "config_details": {"top_features": top_features, "steer_value": steer_value},
            "unit_id": example["unit_id"],
            "original_src": example["src_text"],
            "original_mt": example["mt_text"],
            "original_pe": example["pe_text"],
            "input_clean": prompt,
            "output_clean": generated_output,
        })

    return output



def contrastive_local(
    model_name: str,
    train_data: list[dict],
    val_data: list[dict],
    test_data: list[dict],
    layer: int,
    top_k: int,
    steer_alpha: float,
    template_to_use: str,
    output_path: Path,
    seed: int,
):
    start_time = time.time()

    model: NNsightWrapper = NNsightWrapper( 
        model_name,
        load_in_4bit=False,
    )

    print(f"Model {model_name} loaded on {model.device} in {time.time() - start_time:.2f} seconds")
    start_time = time.time()

    # load the SAE
    # if the sae_name does not include the layer number, the format won't replace anything
    sae_name = SAE_MAPPER[model_name].format(layer=layer)
    hook_point = f"model.layers.{layer}"
    sae: SAEWrapper = SAEWrapper(
        sae_name=sae_name,
        sae_layer=layer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    print(f"SAE {sae_name} loaded on {sae.device} in {time.time() - start_time:.2f} seconds")
    start_time = time.time()

    # extract features using contrastive examples
    
    final_latents_edit, final_latents_noedit = get_contrastive_features_local(
        model=model,
        hook_point=hook_point,
        sae=sae,
        max_new_tokens=1024,
        train_data=train_data,
        val_data=val_data,
        PROMPT_TEMPL=template_to_use,
        LANG_MAP=LANG_MAP,
        seed=seed,
    )
    
    interventions = {
        hook_point: partial(
            inverv_fun_mutual_info,
            sae=sae,
            edit_latents=final_latents_edit,
            noedit_latents=final_latents_noedit,
            alpha=steer_alpha,
        )
    }

    output: list[dict] = []
    all_new_activations: list[torch.Tensor] = []
    for example in (pbar := tqdm(test_data, desc="Contrastive test")):
        prompt = template_to_use.format(
            src_text=example["src_text"],
            lang=LANG_MAP[example["tgt_lang"]],
            mt_text=example["mt_text"],
        )
        inp_tokens = model.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors='pt',
        )

        pbar.set_description(f"Contrastive test | {get_GPU_mem_str()}")
        out, new_activations = model.set_forward(
            inp_tokens=inp_tokens,
            interventions=interventions,
            return_new_activations=True,
        )
        all_new_activations.append(new_activations)
        pbar.set_description(f"Contrastive test | {get_GPU_mem_str()}")

        output.append({
            "config_details": {
                "steer_value": steer_alpha, "top_k": top_k,
                "edit_latents": final_latents_edit.nonzero().flatten().tolist(),
                "noedit_latents": final_latents_noedit.nonzero().flatten().tolist(),
            },
            "unit_id": example["unit_id"],
            "original_src": example["src_text"],
            "original_mt": example["mt_text"],
            "original_pe": example["pe_text"],
            "input_clean": prompt,
            "output_clean": clean_txt(model.tokenizer.decode(out, skip_special_tokens=True)),
        })

    # stack new_activations
    for i, _ in enumerate(all_new_activations):
        # keep only the first generated token
        all_new_activations[i] = all_new_activations[i][0]
    all_new_activations = torch.stack(all_new_activations)
    # save all_new_activations
    torch.save(all_new_activations, output_path / "support" / f"{seed}-l{layer:02d}-all_new_activations.pt")
        
    return output


def steering_local(
    model_name: str,
    train_data: list[dict],
    test_data: list[dict],
    layer: int,
    top_k: int,
    steer_alpha: float,
    template_to_use: str,
    seed: int,
    output_path: Path,
):
    """
    Local token-level implementation of the contrastive steering.
    """
    # load model
    model: NNsightWrapper = NNsightWrapper( 
        model_name,
        load_in_4bit=False,
    )

    # load the SAE
    # if the sae_name does not include the layer number, the format won't replace anything
    sae_name = SAE_MAPPER[model_name].format(layer=layer)
    hook_point = f"model.layers.{layer}"
    sae: SAEWrapper = SAEWrapper(
        sae_name=sae_name,
        sae_layer=layer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    # check if file already exists
    if os.path.exists(output_path / "support" / "raw" / f"{seed}-l{layer:02d}-all_features.json"):
        with open(output_path / "support" / "raw" / f"{seed}-l{layer:02d}-all_features.json", "r") as f:
            features_while_generating: list[list[dict[str, Any]]] = json.load(f)
    else:
        features_while_generating: list[list[dict[str, Any]]] = get_features_local(
            model=model,
            hook_point=hook_point,
            sae=sae,
            max_new_tokens=1024,
            train_data=train_data,
            PROMPT_TEMPL=template_to_use,
            LANG_MAP=LANG_MAP,
        )
        # save features
        with open(output_path / "support" / "raw" / f"{seed}-l{layer:02d}-all_features.json", "w") as f:
            json.dump(features_while_generating, f, indent=4, ensure_ascii=False)

    # selection criteria


    latents_and_logits = latents_classificaiton(features_while_generating, top_k=top_k)

    all_logits: dict[int, float] = {}
    for logit_type in latents_and_logits:
        for lat, logit_info in latents_and_logits[logit_type].items():
            all_logits[lat] = logit_info["mean"] * steer_alpha

    interventions = {
        hook_point: partial(
            custom_online_interv_fun,
            sae=sae,
            good_features=list(latents_and_logits["good"].keys()),
            bad_features=list(latents_and_logits["bad"].keys()),
            worse_features=list(latents_and_logits["worse"].keys()),
            logits_to_use=all_logits,
        )
    }


    output: list[dict] = []
    for example in (pbar := tqdm(test_data, desc="Steering test")):

        prompt = template_to_use.format(
            src_text=example["src_text"],
            lang=LANG_MAP[example["tgt_lang"]],
            mt_text=example["mt_text"],
        )

        inp_tokens = model.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors='pt',
        )

        pbar.set_description(f"Steering test | {get_GPU_mem_str()}")
        out = model.set_forward(
            inp_tokens=inp_tokens,
            interventions=interventions,
        )
        pbar.set_description(f"Steering test | {get_GPU_mem_str()}")

        output.append({
            "config_details": {"steer_value": steer_alpha},
            "unit_id": example["unit_id"],
            "original_src": example["src_text"],
            "original_mt": example["mt_text"],
            "original_pe": example["pe_text"],
            "input_clean": prompt,
            "output_clean": clean_txt(model.tokenizer.decode(out, skip_special_tokens=True)),
        })

    return output


def reft_local(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer: int,
    train_data: list[dict],
    test_data: list[dict],
    output_path: Path,
    template_to_use: str,
):
    model = model.to(torch.device("cuda"))

    hook_point = f"model.layers.{layer}"

    reft_config = pyreft.ReftConfig(
        representations={
            "layer": layer,
            "component": hook_point + ".output",
            "low_rank_dimension": 4,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=4,
            )
        }
    )

    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(model.device)
    reft_model.print_trainable_parameters()

    # Prepare the training data
    training_examples: list[list[str]] = []
    for example in train_data:
        prompt = template_to_use.format(
            src_text=example["src_text"],
            lang=LANG_MAP[example["tgt_lang"]],
            mt_text=example["mt_text"],
        )
        tgt = example["pe_text"]
        training_examples.append([prompt, tgt])

    data_module = pyreft.make_last_position_supervised_data_module(
        tokenizer, model, 
        [inp for inp, _ in training_examples],
        [tgt for _, tgt in training_examples],
    )

    # train
    train_args = TrainingArguments(
        num_train_epochs=100.0,
        output_dir=output_path / "support" / f"reft-{layer:02d}",
        per_device_train_batch_size=10,
        learning_rate=4e-3,
        save_total_limit=1,     # Reduce storage consumption
        logging_steps=20,
    )
    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=train_args,
        **data_module,
    )
    _ = trainer.train()

    # test
    output: list[dict] = []
    for example in (pbar := tqdm(test_data, desc="Reft test")):
        prompt = template_to_use.format(
            src_text=example["src_text"],
            lang=LANG_MAP[example["tgt_lang"]],
            mt_text=example["mt_text"],
        )
        inp_tokens = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inp_tokens = tokenizer(inp_tokens, return_tensors="pt")

        for k in inp_tokens:
            inp_tokens[k] = inp_tokens[k].to(model.device)

        base_unit_location = inp_tokens['input_ids'].shape[-1] - 1      # last pos
        pbar.set_description(f"Reft test | {get_GPU_mem_str()}")
        _, reft_out = reft_model.generate(
            inp_tokens, 
            unit_locations={"sources->base": (None, [[[base_unit_location]]])},
            intervene_on_prompt=True, 
            max_new_tokens=1024, 
            do_sample=False, 
            eos_token_id=tokenizer.eos_token_id, 
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        )
        pbar.set_description(f"Reft test | {get_GPU_mem_str()}")

        input_len = inp_tokens['input_ids'].shape[-1]
        output.append({
            "config_details": {"reft_hook": hook_point},
            "unit_id": example["unit_id"],
            "original_src": example["src_text"],
            "original_mt": example["mt_text"],
            "original_pe": example["pe_text"],
            "input_clean": prompt,
            "output_clean": clean_txt(tokenizer.decode(reft_out.squeeze()[input_len:], skip_special_tokens=True)),
        })

    return output




def main(
    model_name: str,
    persona_name: str,
    lang: str,
    layers: int | list[int] = [19],
    templates: list[str] = ["translate", "edit"],
    steer_alpha: float | list[float] = 2.,
    top_k: int = 10,
    oth_persona: str | None = None,
    output_path: str = "/scratch/$ME/steer_outputs_scratch",
    seed: int = 25,
    debug: bool = False,
    low_mem: bool = False,
    small_test_set: bool = True,        # only use the classifier test set (which is smaller)
    experiments: str | list[str] = ["just_mt", "zero_shot", "multi_shot", "multi_mt", "steer", "contrastive", "random_shot", "reft"],
):
    print(f'[-------] Persona name: {persona_name} [-------]')

    model_name_path = model_name.split("/")[-1]
    if oth_persona:
        base_experiment_path: Path = Path(output_path) / f'{persona_name}-VS-{oth_persona}'
    else:
        base_experiment_path: Path = Path(output_path) / persona_name

    if isinstance(layers, int):
        layers: list[int] = [layers]

    output_path: Path = base_experiment_path / model_name_path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path / "support" / "raw", exist_ok=True)

    # Set the seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_data, val_data, test_data, random_train_data = load_data(
        model_name=model_name,
        persona_name=persona_name,
        lang=lang,
        oth_persona=oth_persona,
    )
        
    if small_test_set:
        try:
            with open(base_experiment_path / model_name.split('/')[-1] / "classifier_3way_out" / "test_structured_input.json", "r") as f:
                small_test_data = json.load(f)        # replace the test data with the classifier test data
            # select at most 150 examples to avoid long execution times with long novels
            idxs_small_test_data = [example["unit_id"] for example in small_test_data]
            test_data = [example for example in test_data if example["unit_id"] in idxs_small_test_data]
            test_data = test_data[:150]
        except FileNotFoundError:
            warnings.warn(f"Small test set is True but the classifier test data was not found, please train the classifier before running this script. The script will continue using the entire test set.")

    if debug:
        test_data = test_data[:10]

    if isinstance(experiments, str):
        experiments = [experiments]

    if isinstance(steer_alpha, float):
        steer_alpha = [steer_alpha]

    if 'just_mt' in experiments or 'zero_shot' in experiments or 'multi_shot' in experiments or 'multi_mt' in experiments or 'random_shot' in experiments or 'reft' in experiments:
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            load_in_4bit=low_mem,
            attn_implementation="eager",
        )
        print(f"Model {model_name} loaded on {model.device}")

        if 'just_mt' in experiments:
            print(f"\n\n##############################")
            print(f"Just MT with {persona_name}")
            mt_out = mt_baseline(
                model=model,
                tokenizer=tokenizer,
                test_data=test_data,
                seed=seed,
            )
            # eval and save
            eval_and_save(
                output_results=mt_out,
                experiment="just_mt",
                model_name=model_name,
                persona_name=persona_name,
                template="translate",
                seed=seed,
                output_path=output_path,
                base_experiment_path=base_experiment_path,
            )

        if 'zero_shot' in experiments:
            print(f"\n\n##############################")
            print(f"Zero-shot with {persona_name}")
            # Get the zero-shot baseline
            zero_shot_out = zero_shot_baseline(
                model=model,
                tokenizer=tokenizer,
                test_data=test_data,
                template_to_use=TEMPLATE_MAP['translate'],
                seed=seed,
            )
            # eval and save
            eval_and_save(
                output_results=zero_shot_out,
                experiment="zero_shot",
                model_name=model_name,
                persona_name=persona_name,
                template="translate",
                seed=seed,
                output_path=output_path,
                base_experiment_path=base_experiment_path,
            )

            print(f"\n\n##############################")
            print(f"Zero-shot with {persona_name} + explanations")
            # Get the zero-shot baseline
            print('Trying to load template explanations')
            
            # it's in train/lang/{persona_name}_instr.txt
            path_to_explanations = f"data/train/{lang}/{persona_name}_instr.txt"
            if os.path.exists(path_to_explanations):
                with open(path_to_explanations, "r") as f:
                    explanations = f.read()

                pre_filled_template_to_use = MT_template_explainations.format(
                    explanations=explanations,
                    # trick to preserve the format for later src_text
                    lang="{lang}",
                    src_text="{src_text}",
                )

                zero_shot_out = zero_shot_baseline(
                    model=model,
                    tokenizer=tokenizer,
                    test_data=test_data,
                    template_to_use=pre_filled_template_to_use,
                    seed=seed,
                )
                # eval and save
                eval_and_save(
                    output_results=zero_shot_out,
                    experiment="zero_shot",
                    model_name=model_name,
                    persona_name=persona_name,
                    template="explanations",
                    seed=seed,
                    output_path=output_path,
                    base_experiment_path=base_experiment_path,
                    additional_info="explain",
                )
            else:
                print(f"Template explanations not found at {path_to_explanations}, skipping this experiment.")

        if 'multi_mt' in experiments:
            print(f"\n\n##############################")
            print(f"Multi-shot with MT with {persona_name}")
            # params few-shot:
            shots: int = 20
            for template_name in templates:
                print(f"Doing {template_name} template")
                multi_shot_out = multi_shot_with_mt(
                    model=model,
                    tokenizer=tokenizer,
                    train_data=train_data,
                    test_data=test_data,
                    shots=shots,
                    template_to_use=TEMPLATE_MAP[template_name],
                    seed=seed,
                )
                # eval and save
                eval_and_save(
                    output_results=multi_shot_out,
                    experiment="multi_mt",
                    model_name=model_name,
                    persona_name=persona_name,
                    template=template_name,
                    seed=seed,
                    output_path=output_path,
                    base_experiment_path=base_experiment_path,
                    additional_info=f"k{shots}",
                )

        if 'multi_shot' in experiments:
            print(f"\n\n##############################")
            print(f"Few-shot with {persona_name}")
            # params few-shot:
            shots: int = 20
            multi_shot_out = multi_shot_baseline(
                model=model,
                tokenizer=tokenizer,
                train_data=train_data,
                test_data=test_data,
                shots=shots,
                template_to_use=TEMPLATE_MAP['translate'],
                seed=seed,
            )
            # eval and save
            eval_and_save(
                output_results=multi_shot_out,
                experiment="multi_shot",
                model_name=model_name,
                persona_name=persona_name,
                template="translate",
                seed=seed,
                output_path=output_path,
                base_experiment_path=base_experiment_path,
                additional_info=f"k{shots}",
            )

        if 'random_shot' in experiments:
            print(f"\n\n##############################")
            print(f"Random-shot with {persona_name}")
            # params few-shot:
            shots: int = 20
            for template_name in templates:
                print(f"Doing {template_name} template")
                random_multi_shot_out = multi_shot_baseline(
                    model=model,
                    tokenizer=tokenizer,
                    train_data=random_train_data,
                    test_data=test_data,
                    shots=shots,
                    template_to_use=TEMPLATE_MAP[template_name],
                    seed=seed,
                )
                # eval and save
                eval_and_save(
                    output_results=random_multi_shot_out,
                    experiment="random_shot",
                    model_name=model_name,
                    persona_name=persona_name,
                    template=template_name,
                    seed=seed,
                    output_path=output_path,
                    base_experiment_path=base_experiment_path,
                    additional_info=f"k{shots}",
                )

        if 'reft' in experiments:
            print(f"\n\n##############################")
            print(f"Reft with {persona_name}")

            # redownload the tokenizer with reft custom config
            reft_tokenizer = AutoTokenizer.from_pretrained(
                model_name, model_max_length=2048,
                padding_side='left', use_fast=False,
            )

            if reft_tokenizer.pad_token is None:
                reft_tokenizer.pad_token = reft_tokenizer.eos_token
                reft_tokenizer.pad_token_id = reft_tokenizer.eos_token_id

            for layer in layers:
                for template_name in templates:
                    print(f"Doing {template_name} template")
                    reft_out = reft_local(
                        model=model,
                        tokenizer=reft_tokenizer,
                        layer=layer,
                        train_data=train_data,
                        test_data=test_data,
                        template_to_use=TEMPLATE_MAP[template_name],
                        output_path=output_path,
                    )
                    # eval and save
                    eval_and_save(
                        output_results=reft_out,
                        experiment="reft",
                        model_name=model_name,
                        persona_name=persona_name,
                        template=template_name,
                        seed=seed,
                        output_path=output_path,
                        base_experiment_path=base_experiment_path,
                        additional_info=f"l{layer:02d}",
                    )


        # free memory, from now on using the NNsightWrapper
        del model


    # # params contrastive:
    # top_features: int = 2
    # steer_value: float = 0.3
    # contrastive_out = contrastive_remote(
    #     train_data=train_data,
    #     test_data=test_data,
    #     seed=seed,
    #     output_path=output_path,
    #     top_features = top_features,
    #     steer_value = steer_value,
    # )
    # with open(output_path / f"{persona_name}-{seed}-contrastive-{top_features}-{steer_value}.json", "w") as f:
    #     json.dump(contrastive_out, f, indent=4, ensure_ascii=False)


    if 'steer' in experiments:
        print(f"\n\n##############################")
        print(f"Steering with {persona_name}")
        for layer in layers:
            # params:
            # top_k: int = 40
            # steer_alpha: float = 2.0
            for template_name in templates:
                print(f"Doing {template_name} template")
                output = steering_local(
                    model_name=model_name,
                    train_data=train_data,
                    test_data=test_data,
                    top_k=top_k,
                    layer=layer,
                    steer_alpha=steer_alpha,
                    template_to_use=TEMPLATE_MAP[template_name],
                    seed=seed,
                    output_path=output_path,
                )
                if output == -1:
                    print(f"Skipping layer {layer} for model {model_name}")
                    continue
                # eval and save
                eval_and_save(
                    output_results=output,
                    experiment="steer",
                    model_name=model_name,
                    persona_name=persona_name,
                    template=template_name,
                    seed=seed,
                    output_path=output_path,
                    base_experiment_path=base_experiment_path,
                    additional_info=f"l{layer:02d}-k{top_k}-a{steer_alpha}",
                )

    if 'contrastive' in experiments:
        print(f"\n\n##############################")
        print(f"Contrastive with {persona_name}")
        for layer in layers:
            for alpha in steer_alpha:
                for template_name in templates:
                    print(f"Doing {template_name} template")
                    output = contrastive_local(
                        model_name=model_name,
                        train_data=train_data,
                        val_data=val_data,
                        test_data=test_data,
                        layer=layer,
                        top_k=top_k,
                        steer_alpha=alpha,
                        template_to_use=TEMPLATE_MAP[template_name],
                        output_path=output_path,
                        seed=seed,
                    )
                    # eval and save
                    eval_and_save(
                        output_results=output,
                        experiment="contrastive",
                        model_name=model_name,
                        persona_name=persona_name,
                        template=template_name,
                        seed=seed,
                        output_path=output_path,
                        base_experiment_path=base_experiment_path,
                        additional_info=f"l{layer:02d}-k{top_k}-a{alpha}",
                    )


if __name__ == "__main__":
    fire.Fire(main)

