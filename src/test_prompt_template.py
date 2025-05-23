import fire
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich import print

PROMPT_TEMPL = """\
You are given a sentence in English and its translation in {lang}. 
If you believe the translation is correct, write the sentence exactly as it is. If you think it can be improved, rewrite it with the necessary changes.

The original sentence is: \"{src_text}\";
The translation is: \"{mt_text}\"

Remember to write only the translation, without any additional text or explanation.
"""



def main(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
):


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = PROMPT_TEMPL.format(
        src_text="Fracture detection by artificial intelligence and especially Deep Convolutional Neural Networks (DCNN) is a topic of growing interest in current orthopaedic and radiological research.",
        lang="Italian",
        mt_text="Il rilevamento delle fratture tramite l'intelligenza artificiale e in particolare le reti neurali convoluzionali profonde (DCNN) è un argomento di crescente interesse nell'attuale ricerca ortopedica e radiologica."
    )
    # generate
    input_chat = [
        {"role": "user", "content": prompt},
    ]
    tokenized_inp = tokenizer.apply_chat_template(input_chat, add_generation_promtp=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            tokenized_inp,
            do_sample=False,
            max_new_tokens=1000,
        )

    print(tokenizer.decode(out.squeeze()))








if __name__ == "__main__":
    fire.Fire(main)


