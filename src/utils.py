import torch
import numpy as np

from typing import Any


def clean_txt(text: str) -> str:
    """
    If the text starts with a quote, remove it and everything after the next quote.
    Else return the text as is.

    To prevent this from happening:
    ```python
    # when the model outputs something like:
    text = "\"<generated translation>\" \\n I've changed the word1 to word2 ..."
    text = clean_txt(text)
    # text = "<generated translation>"
    ```
    """
    # remove \n\n
    is_newline = text.find("\n\n")
    if is_newline != -1:
        text = text[:is_newline]

    return text

    # if text[0] == "\"":
    #     text = text[1:]
    #     another_quote = text.find("\"")
    #     if another_quote != -1:
    #         text = text[:another_quote]
    #     return text
    # else:
    #     return text


def get_GPU_mem(device: str = 'cuda') -> tuple[float, float]:
    if torch.cuda.is_available():

        free, total = torch.cuda.mem_get_info(device)
        # used to GB
        mem_used_GB = (total - free) / 1024**3
        # total to GB
        total_GB = total / 1024**3

        return (mem_used_GB, total_GB)

    else:
        return (-1., -1.)


def get_GPU_mem_str(device: str = 'cuda') -> str:
    str_out = f'GPU: {get_GPU_mem(device)[0]:.1f} / {get_GPU_mem(device)[1]:.1f} ({get_GPU_mem(device)[0] / get_GPU_mem(device)[1]:.1%})'
    return str_out


def latents_classificaiton(
    features_while_generating: list[list[dict[str, Any]]],
    top_k: int = 40,
):
    """
    Function to classify the latents during generation.
    Returns the top_k latents for each group (if any).
    Groups:
    - good: latenti attivi quando il modello edita come PE (== pe != mt)
    - worse: active latents when the model misses the opportunity to edit (!= pe == mt)
    - bad: active latents when the model edits but it wasn't supposed to (!= pe != mt) and pe == mt
    - special: active latents when the model edits but in a different way than PE (!= pe != mt) and pe != mt
    - ignore: active latents when the model is just copying stuff (== pe == mt)

    features_while_generating: list[        # for every example in train
        list[                               # for every gen token
            dict[str, Any]                  # {"features": [list of features idxs], "logits": [list of corresponding logits], "rec_token": {"gen": gen_token, "pe": pe_token, "mt": mt_token}}
        ]
    ]
    """
    latents_and_logits: dict = {
        "good": {},
        "bad": {},
        "worse": {},
        "special": {},
        "ignore": {},
    }

    # groups
    for example in features_while_generating:
        for gen_step in example:
            # get the tokens
            gen, pe, mt = gen_step["rec_token"]["gen"], gen_step["rec_token"]["pe"], gen_step["rec_token"]["mt"]
            latents_here = gen_step["features"]
            logits_here = gen_step["logits"]

            # classify the latents
            if gen == pe and gen != mt:
                # good
                for lat, logit in zip(latents_here, logits_here):
                    if lat not in latents_and_logits["good"]:
                        latents_and_logits["good"][lat] = []
                    latents_and_logits["good"][lat].append(logit)
            elif gen != pe and gen == mt:
                for lat, logit in zip(latents_here, logits_here):
                    if lat not in latents_and_logits["worse"]:
                        latents_and_logits["worse"][lat] = []
                    latents_and_logits["worse"][lat].append(logit)
            elif gen != pe and gen != mt:
                if pe == mt:
                    for lat, logit in zip(latents_here, logits_here):
                        if lat not in latents_and_logits["bad"]:
                            latents_and_logits["bad"][lat] = []
                        latents_and_logits["bad"][lat].append(logit)
                else:
                    for lat, logit in zip(latents_here, logits_here):
                        if lat not in latents_and_logits["special"]:
                            latents_and_logits["special"][lat] = []
                        latents_and_logits["special"][lat].append(logit)
            else:
                for lat, logit in zip(latents_here, logits_here):
                    if lat not in latents_and_logits["ignore"]:
                        latents_and_logits["ignore"][lat] = []
                    latents_and_logits["ignore"][lat].append(logit)


    good_set = set(latents_and_logits['good'].keys())
    bad_set = set(latents_and_logits['bad'].keys())
    worse_set = set(latents_and_logits['worse'].keys())
    special_set = set(latents_and_logits['special'].keys())
    ignore_set = set(latents_and_logits['ignore'].keys())


    # latents to remove (common latents activating for multiple sets that have nothing to do with the token selection)
    remove_from_good = worse_set.union(bad_set).union(special_set).union(ignore_set)
    remove_from_bad = good_set.union(worse_set).union(special_set).union(ignore_set)
    remove_from_worse = good_set.union(bad_set).union(special_set).union(ignore_set)
    remove_from_special = good_set.union(bad_set).union(worse_set).union(ignore_set)
    remove_from_ignore = good_set.union(bad_set).union(worse_set).union(special_set)

    # remove the latents
    for lat in remove_from_good:
        if lat in latents_and_logits["good"]:
            del latents_and_logits["good"][lat]
    for lat in remove_from_bad:
        if lat in latents_and_logits["bad"]:
            del latents_and_logits["bad"][lat]
    for lat in remove_from_worse:
        if lat in latents_and_logits["worse"]:
            del latents_and_logits["worse"][lat]
    for lat in remove_from_special:
        if lat in latents_and_logits["special"]:
            del latents_and_logits["special"][lat]
    for lat in remove_from_ignore:
        if lat in latents_and_logits["ignore"]:
            del latents_and_logits["ignore"][lat]

    def _get_top_k(unique_latents: list[int], count: list[int], k: int) -> list[int]:
        # unique_latents, count = np.unique(latents, return_counts=True)
        unique_latents = np.array(unique_latents)
        count = np.array(count)
        sorted_idxs = np.argsort(-count)      # sort in descending order
        top_k_elements = sorted_idxs[:k]      # top k features
        return unique_latents[top_k_elements].tolist()

    # devo filtrare gli only good, bad, worse, special e ignore
    # però oltre a filtrarli mi devo anche portare dietro i logits


    # get the top k latents for each group
    for latent_type in latents_and_logits:
        # get the unique latents and their counts
        unique_latents = list(latents_and_logits[latent_type].keys())
        count = [len(latents_and_logits[latent_type][lat]) for lat in unique_latents]
        
        # get the top k latents
        top_k_latents = _get_top_k(unique_latents, count, top_k)
        # filter
        latents_and_logits[latent_type] = {lat: latents_and_logits[latent_type][lat] for lat in top_k_latents}



    # get the average logits for each latent
    for latent_type in latents_and_logits:
        for lat, logits in latents_and_logits[latent_type].items():
            latents_and_logits[latent_type][lat] = {
                "mean": np.mean(logits, axis=0).item(),
                "std": np.std(logits, axis=0).item(),
            }


    return latents_and_logits










if __name__ == "__main__":
    # test latents classification

    import json
    from rich import print

    name = 'oracle_t1'
    name = 'no_highlight_t1'
    with open(f'./steer_outputs/Llama-3.1-8B-Instruct/{name}/support/raw/25-l19-all_features.json', 'r') as f:
        features_while_generating = json.load(f)


    latents_classificaiton(
        features_while_generating=features_while_generating,
        top_k = 40,
    )















