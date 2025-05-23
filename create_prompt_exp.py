import fire

import json



TEMPLATE = """\
Objective: Identify stylistic choices in translations for personalization purposes.

You will be provided with a source text, a standard translation, and a target translation by a specific translator whose style we want to emulate.
Your task is to analyze the 'Target translation' by comparing it to the 'Base translation' and the 'Source text'.
Identify and list the distinctive stylistic patterns, choices, and preferences exhibited in the Target translation.

These stylistic cues should help another translator (or an AI) to adapt their translations to match the style of the target translator.

{examples}

Please extract a concise list of key stylistic cues. Focus on aspects such as vocabulary choices, sentence structure, tone and register, handling of cultural nuances, punctuation/formatting preferences and overall creativity.

Output a short list of stylistic cues as bullet points. Write the list as if you were directly giving the guidelines to the translator and avoid using specific examples.
"""

EXAMPLES_TEMPLATE = """\
Source text: {original}
Base translation: {translation}
Target translation: {target_translation}
"""



def main(
    persona_name: str,
    model_name: str = "google/gemma-2-2b-it",
):
    print(f"Creating prompt experiment for persona: {persona_name}")


    # load train_examples:
    with open(f'./data/train/eng/{persona_name}.json', 'r') as f:
        train_examples = json.load(f)

    list_of_examples = []
    for example in train_examples:
        list_of_examples.append(
            EXAMPLES_TEMPLATE.format(
                original=example['src_text'],
                translation=example[model_name.split("/")[-1]],
                target_translation=example['pe_text']
            )
        )

    examples = "\n".join(list_of_examples)

    prompt = TEMPLATE.format(examples=examples)

    print('---------------------')
    print(prompt)
    print('---------------------')









if __name__ == "__main__":
    fire.Fire(main)
