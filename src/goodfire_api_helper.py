import os
import fire
# import torch
import json
import goodfire

from goodfire import FeatureGroup, Feature
# from pathlib import Path
# from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
# from datasets import load_dataset
from rich import print
# import numpy as np


# Load environment variables from .env file
load_dotenv()
GOODFIRE_API_KEY = os.getenv("GOODFIRE_API")

PROMPT_TEMPL = """\
You are given a sentence in English and its translation in {lang}. 
If you believe the translation is correct, write the sentence exactly as it is. If you think it can be improved, rewrite it with the necessary changes.

The original sentence is: \"{src_text}\";
The translation is: \"{mt_text}\"

Remember to write only the translation, without any additional text or explanation.
"""

LANG_MAP = {
    "ita": "Italian",
    "nld": "Dutch",
}



def contrastive_api(
    default_chats: list[list[dict[str, str]]],
    edit_chats: list[list[dict[str, str]]],
    n_feat: int,
) -> tuple[FeatureGroup, FeatureGroup]:

    client = goodfire.Client(api_key=GOODFIRE_API_KEY)

    # Instantiate a model variant. 
    variant = goodfire.Variant('meta-llama/Meta-Llama-3.1-8B-Instruct')

    variant.reset()

    default_features, edit_features = client.features.contrast(
        dataset_1=default_chats,
        dataset_2=edit_chats,
        model=variant,
        top_k=n_feat,
    )

    return default_features, edit_features


def generate_features(
    chat: list[dict[str, str]],
    features: FeatureGroup | Feature,
    seed: int,
    steer_value: float = 0.6,
    expected_gen_len: int | None = None,
) -> str:
    """
    Generate a response from the model using the given features.
    Args:
        chat (list[dict[str, str]]): The chat history.
        features (FeatureGroup | Feature): The features to use for generation.
        seed (int): The seed for random generation.
        steer_value (float, optional): The steering value. Defaults to 0.6.
        expected_gen_len (int | None, optional): If not None, the expected length of the generated response is used to stop the generation if the specified num of char are reached. A good value could be 120% of the target generation. Defaults to None.
    """

    if isinstance(features, Feature):
        features = FeatureGroup([features])

    client = goodfire.Client(api_key=GOODFIRE_API_KEY)

    # Instantiate a model variant. 
    variant = goodfire.Variant('meta-llama/Meta-Llama-3.1-8B-Instruct')
    variant.set(features, steer_value)


    out = ""
    for token in client.chat.completions.create(
        messages=chat,
        model=variant,
        stream=True,
        max_completion_tokens=512,
        top_p=0.0,
        seed=seed,
    ):
        out += token.choices[0].delta.content

        # stopping criteria
        if expected_gen_len is not None:
            if len(out) > expected_gen_len:
                break

    # response = client.chat.completions.create(
    #     messages=chat,
    #     model=variant,
    #     stream=False,
    #     max_completion_tokens=512,
    #     top_p=0.0,
    #     seed=seed,
    # )
    # out: str = response.choices[0].message["content"]

    return out







def main(
    train_persona_path: str,
    # test_persona_path: str,
):

    with open(train_persona_path, "r") as f:
        train_persona = json.load(f)

    noedit_chats = []
    edit_chats = []
    for instance in train_persona:
        # make chat
        prompt = PROMPT_TEMPL.format(
            src_text=instance["src_text"],
            lang=LANG_MAP[instance["tgt_lang"]],
            mt_text=instance["mt_text"],
        )

        noedit_chats.append([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": instance["mt_text"]}
        ])
        edit_chats.append([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": instance["pe_text"]}
        ])






if __name__ == "__main__":
    fire.Fire(main)

