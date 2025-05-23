from rich import print
import random
import pickle
import json
import math
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast

random.seed(24)
TRAIN_SIZE = 20
VAL_SIZE = 20


MT_template = """\
Translate the following sentence between the angular parentheses into {lang}.

The original sentence is: <{src_text}>.

Remember to write only the translation, without any additional text or explanation.
"""

LANG_MAP = {
    "ita": "Italian",
    "nld": "Dutch",
    "eng": "English",
}


def _generate_model(
    model,
    tokenizer,
    tokenized_inp,
    return_decoded: bool = False,
    **kwargs
):

    original_input_len: int = tokenized_inp.shape[-1]

    tokenized_inp = tokenized_inp.to(model.device)

    with torch.no_grad():
        # Generate the output
        with autocast():
            output: torch.LongTensor = model.generate(
                tokenized_inp,
                **kwargs,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
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


def clean_text(text):
    # do not clean text
    text = text.replace("”", "\"")
    text = text.replace("“", "\"")
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("- ", "\"")
    text = text.replace(" -", "\"")
    text = text.replace("- ", "\"")
    # text = text.replace("\"", "")

    return text



def generate_batch(
    model,
    tokenizer,
    inp_tokens,
    attention_mask,
):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inp_tokens,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=None,
        )

    # decode outputs
    input_lengths = inp_tokens.shape[1]
    generated_tokens = outputs[:, input_lengths:]

    batch_model_mts = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return batch_model_mts


def add_model_mt(
    model_name: str,
    batch_size: int,
    translator_data: list[dict],
) -> list[dict]:

    model_key_name = model_name.split('/')[-1] # Key to store results

    # check if the model's MTs are already in the data
    if model_key_name in translator_data[0].keys():
        print(f"Model {model_key_name} already exists in the data, skipping")
        return translator_data


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # set padding side to left
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    model.eval()
    model = torch.compile(model)

    num_paragraphs = len(translator_data)

    num_batches = math.ceil(num_paragraphs / batch_size)

    # replace every MT with the model MT
    for i in tqdm(
        range(0, num_paragraphs, batch_size), 
        total=num_batches, 
        desc=f"[{novel}] [1/2] Replacing MTs with {model_key_name}'s MT", 
        dynamic_ncols=True,
    ):
        # prepare batch
        batch_indices = range(i, min(i + batch_size, num_paragraphs))
        batch_data = [translator_data[j] for j in batch_indices]
        batch_src_texts = [data["src_text"] for data in batch_data]

        # format prompts for batch
        batch_prompts = [
            MT_template.format(src_text=src_text, lang=LANG_MAP['eng'])
            for src_text in batch_src_texts
        ]
        batch_conversations = [
            [{"role": "user", "content": prompt}]
            for prompt in batch_prompts
        ]

        # tokenize batch
        batch_chats = tokenizer.apply_chat_template(
            batch_conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        inp_tokens = tokenizer(
            batch_chats,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        # generate model MTs
        batch_model_mts = generate_batch(
            model=model,
            tokenizer=tokenizer,
            inp_tokens=inp_tokens.input_ids,
            attention_mask=inp_tokens.attention_mask,
        )

        # store results
        for j, result_text in enumerate(batch_model_mts):
            original_index = batch_indices[j]
            translator_data[original_index][model_key_name] = result_text.strip()

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return translator_data



models = [
    # "meta-llama/Llama-3.1-8B-Instruct", 
    # "google/gemma-2-2b-it", 
    "google/gemma-2-9b-it",
]
batch_size = 8
with open("./data/par3.pkl", "rb") as f:
    edit_dict_ita = pickle.load(f)

selected_novels = [
    "pinocchio_it",
    "around_the_world_in_eighty_days_fr",
    "beware_of_pity_de",
    "the_diary_of_a_young_girl_nl",
    "crime_and_punishment_ru",
    "don_quixote_es",
    "no_longer_human_ja",
    "dream_of_the_red_chamber_zh",
]


print(selected_novels)

for novel in edit_dict_ita:
    if novel not in selected_novels:
        continue

    # novel not exists: create from scratch
    if not os.path.exists(f"./data/train/eng/{novel}_tra1.json"):
        print('--------------------------------')
        print(f"Novel {novel} not exists, creating from scratch")
        print('--------------------------------')

        tran_1 = []
        tran_2 = []
        num_paragraphs = len(edit_dict_ita[novel]["source_paras"])

        print(f'\nProcessing {novel} | amount of paragraphs: {num_paragraphs}')
        for i in range(num_paragraphs):
            tran_1.append(
                {
                    "unit_id": i,
                    "src_text": clean_text(edit_dict_ita[novel]["source_paras"][i]),
                    "mt_text": clean_text(edit_dict_ita[novel]["gt_paras"][i]),
                    "pe_text": clean_text(edit_dict_ita[novel]["translator_data"]['translator_1']['translator_paras'][i]),
                    "tgt_lang": "eng",
                }
            )
            tran_2.append(
                {
                    "unit_id": i,
                    "src_text": clean_text(edit_dict_ita[novel]["source_paras"][i]),
                    "mt_text": clean_text(edit_dict_ita[novel]["gt_paras"][i]),
                    "pe_text": clean_text(edit_dict_ita[novel]["translator_data"]['translator_2']['translator_paras'][i]),
                    "tgt_lang": "eng",
                }
            )

        for model_name in models:
            print(f"[1/2] Processing {novel} with {model_name}")
            tran_1 = add_model_mt(
                model_name=model_name,
                batch_size=batch_size,
                translator_data=tran_1,
            )
            print(f"[2/2] Processing {novel} with {model_name}")
            tran_2 = add_model_mt(
                model_name=model_name,
                batch_size=batch_size,
                translator_data=tran_2,
            )

            # save at every model step to prevent some OOM memory at the end that results in a total loss of the data
            train_1 = tran_1[:TRAIN_SIZE]
            val_1 = tran_1[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
            test_1 = tran_1[TRAIN_SIZE + VAL_SIZE:]

            train_2 = tran_2[:TRAIN_SIZE]
            val_2 = tran_2[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
            test_2 = tran_2[TRAIN_SIZE + VAL_SIZE:]

            # save the data
            with open(f"./data/train/eng/{novel}_tra1.json", "w") as f:
                json.dump(train_1, f, ensure_ascii=False, indent=4)
            with open(f"./data/val/eng/{novel}_tra1.json", "w") as f:
                json.dump(val_1, f, ensure_ascii=False, indent=4)
            with open(f"./data/test/eng/{novel}_tra1.json", "w") as f:
                json.dump(test_1, f, ensure_ascii=False, indent=4)

            # save the data
            with open(f"./data/train/eng/{novel}_tra2.json", "w") as f:
                json.dump(train_2, f, ensure_ascii=False, indent=4)
            with open(f"./data/val/eng/{novel}_tra2.json", "w") as f:
                json.dump(val_2, f, ensure_ascii=False, indent=4)
            with open(f"./data/test/eng/{novel}_tra2.json", "w") as f:
                json.dump(test_2, f, ensure_ascii=False, indent=4)

    else:
        # load train, val and test data
        # check if the model's MTs are already in the data
        # if not, add the model MTs to each of them
        print('--------------------------------')
        print(f"Novel {novel} already exists, loading and adding any missing model MTs")
        print('--------------------------------')

        with open(f"./data/train/eng/{novel}_tra1.json", "r") as f:
            train_1 = json.load(f)
        with open(f"./data/val/eng/{novel}_tra1.json", "r") as f:
            val_1 = json.load(f)
        with open(f"./data/test/eng/{novel}_tra1.json", "r") as f:
            test_1 = json.load(f)

        with open(f"./data/train/eng/{novel}_tra2.json", "r") as f:
            train_2 = json.load(f)
        with open(f"./data/val/eng/{novel}_tra2.json", "r") as f:
            val_2 = json.load(f)
        with open(f"./data/test/eng/{novel}_tra2.json", "r") as f:
            test_2 = json.load(f)

        for model_name in models:
            print(f"[1/2] Processing {novel} with {model_name}")
            train_1 = add_model_mt(
                model_name=model_name,
                batch_size=batch_size,
                translator_data=train_1,
            )
            val_1 = add_model_mt(
                model_name=model_name,
                batch_size=batch_size,
                translator_data=val_1,
            )
            test_1 = add_model_mt(
                model_name=model_name,
                batch_size=batch_size,
                translator_data=test_1,
            )

            print(f"[2/2] Processing {novel} with {model_name}")
            train_2 = add_model_mt(
                model_name=model_name,
                batch_size=batch_size,
                translator_data=train_2,
            )
            val_2 = add_model_mt(
                model_name=model_name,
                batch_size=batch_size,
                translator_data=val_2,
            )
            test_2 = add_model_mt(
                model_name=model_name,
                batch_size=batch_size,
                translator_data=test_2,
            )
            
            # save the data
            with open(f"./data/train/eng/{novel}_tra1.json", "w") as f:
                json.dump(train_1, f, ensure_ascii=False, indent=4)
            with open(f"./data/val/eng/{novel}_tra1.json", "w") as f:
                json.dump(val_1, f, ensure_ascii=False, indent=4)
            with open(f"./data/test/eng/{novel}_tra1.json", "w") as f:
                json.dump(test_1, f, ensure_ascii=False, indent=4)

            # save the data
            with open(f"./data/train/eng/{novel}_tra2.json", "w") as f:
                json.dump(train_2, f, ensure_ascii=False, indent=4)
            with open(f"./data/val/eng/{novel}_tra2.json", "w") as f:
                json.dump(val_2, f, ensure_ascii=False, indent=4)
            with open(f"./data/test/eng/{novel}_tra2.json", "w") as f:
                json.dump(test_2, f, ensure_ascii=False, indent=4)




