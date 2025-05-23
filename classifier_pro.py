import torch
import fire
import random
import json
import os
import warnings
import shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from huggingface_hub import HfApi, create_repo # get_full_repo_name removed as not used
from datetime import date
import numpy as np
import textwrap
from rich import print # Ensure rich is installed: pip install rich
from tqdm import tqdm # Ensure tqdm is installed

# MODIFICATION: Updated LBL2ID for 3 classes
LBL2ID = {
    "MT": 0,  # Machine Translation
    "H1": 1,  # Human Translator 1
    "H2": 2,  # Human Translator 2
}


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# MODIFICATION: Updated _get_mock_data for 3 classes
def _get_mock_data():
    """
    Mock data for testing. Used to test the training pipeline for 3 classes.
    Should converge to high accuracy after a few steps.
    """
    _words_to_sample = ['miao', 'hello', 'world', 'foo', 'bar', 'home', 'cat', 'dog', 'fish', 'bird', 'car', 'bike', 'train', 'plane', 'boat']
    
    n_samples_train_class = 150 # Number of samples per class for training
    n_samples_val_test_class = 30 # Number of samples per class for validation and test

    train_data = []
    train_data += [{'text': f'{random.choice(_words_to_sample)} machine-generated text {random.choice(_words_to_sample)}', 'label': LBL2ID["MT"]} for _ in range(n_samples_train_class)]
    train_data += [{'text': f'{random.choice(_words_to_sample)} human-style-one translation {random.choice(_words_to_sample)}', 'label': LBL2ID["H1"]} for _ in range(n_samples_train_class)]
    train_data += [{'text': f'{random.choice(_words_to_sample)} human-style-two translation {random.choice(_words_to_sample)}', 'label': LBL2ID["H2"]} for _ in range(n_samples_train_class)]
    
    val_data = []
    val_data += [{'text': f'{random.choice(_words_to_sample)} machine-generated text {random.choice(_words_to_sample)}', 'label': LBL2ID["MT"]} for _ in range(n_samples_val_test_class)]
    val_data += [{'text': f'{random.choice(_words_to_sample)} human-style-one translation {random.choice(_words_to_sample)}', 'label': LBL2ID["H1"]} for _ in range(n_samples_val_test_class)]
    val_data += [{'text': f'{random.choice(_words_to_sample)} human-style-two translation {random.choice(_words_to_sample)}', 'label': LBL2ID["H2"]} for _ in range(n_samples_val_test_class)]

    test_data = []
    test_data += [{'text': f'{random.choice(_words_to_sample)} machine-generated text {random.choice(_words_to_sample)}', 'label': LBL2ID["MT"]} for _ in range(n_samples_val_test_class)]
    test_data += [{'text': f'{random.choice(_words_to_sample)} human-style-one translation {random.choice(_words_to_sample)}', 'label': LBL2ID["H1"]} for _ in range(n_samples_val_test_class)]
    test_data += [{'text': f'{random.choice(_words_to_sample)} human-style-two translation {random.choice(_words_to_sample)}', 'label': LBL2ID["H2"]} for _ in range(n_samples_val_test_class)]

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data


def _get_data(
    novel_name: str,      # e.g., "pinocchio_it"
    translator1_id: str,  # e.g., "tra1" (suffix for the first persona)
    translator2_id: str,  # e.g., "tra2" (suffix for the second persona)
    generator_name: str,  # Name of the MT generator model key in JSON
    lang: str,
    test_size: float,
    val_size: float,
):
    persona1_full_name = f"{novel_name}_{translator1_id}" # e.g., pinocchio_tra1
    persona2_full_name = f"{novel_name}_{translator2_id}" # e.g., pinocchio_tra2

    # Load data for H1 (persona1)
    with open(f"data/train/{lang}/{persona1_full_name}.json", "r") as f:
        original_train_h1 = json.load(f)
    with open(f"data/test/{lang}/{persona1_full_name}.json", "r") as f:
        original_test_h1 = json.load(f)
    with open(f"data/val/{lang}/{persona1_full_name}.json", "r") as f:
        original_val_h1 = json.load(f)

    # Load data for H2 (persona2)
    with open(f"data/train/{lang}/{persona2_full_name}.json", "r") as f:
        original_train_h2 = json.load(f)
    with open(f"data/test/{lang}/{persona2_full_name}.json", "r") as f:
        original_test_h2 = json.load(f)
    with open(f"data/val/{lang}/{persona2_full_name}.json", "r") as f:
        original_val_h2 = json.load(f)

    model_mt_key = generator_name.split('/')[-1] # e.g., "gemma-2-9b-it"

    # --- Process and align data ---
    # We'll create a unified list of dictionaries, where each dict has:
    # 'unit_id', 'src_text', 'mt_text', 'h1_text', 'h2_text'
    # This requires aligning data from H1 and H2 files, preferably by 'unit_id'.

    def align_data(data_h1, data_h2, dataset_name):
        aligned_entries = []
        map_h2 = {item['unit_id']: item for item in data_h2}
        
        for item_h1 in data_h1:
            unit_id = item_h1['unit_id']
            if unit_id in map_h2:
                item_h2 = map_h2[unit_id]
                # Ensure src_text is the same, otherwise warn
                if item_h1['src_text'] != item_h2['src_text']:
                    warnings.warn(
                        f"Source text mismatch for unit_id {unit_id} in {dataset_name} dataset. "
                        f"H1: '{item_h1['src_text'][:50]}...', H2: '{item_h2['src_text'][:50]}...'. Using H1's source."
                    )
                
                mt_text_content = item_h1.get(model_mt_key)
                if mt_text_content is None:
                    warnings.warn(f"MT text not found for unit_id {unit_id} using key '{model_mt_key}' in {dataset_name} from {persona1_full_name}.json. Skipping this entry.")
                    continue

                aligned_entries.append({
                    "unit_id": unit_id,
                    "src_text": item_h1["src_text"], # Assuming src_text should be identical
                    "mt_text": mt_text_content,
                    "h1_text": item_h1["pe_text"],  # Persona 1's post-edited text
                    "h2_text": item_h2["pe_text"],  # Persona 2's post-edited text
                })
            else:
                warnings.warn(f"Unit ID {unit_id} from {persona1_full_name} not found in {persona2_full_name} for {dataset_name} set. Skipping.")
        return aligned_entries

    processed_train = align_data(original_train_h1, original_train_h2, "train")
    processed_val = align_data(original_val_h1, original_val_h2, "validation")
    processed_test = align_data(original_test_h1, original_test_h2, "test")
    
    if not processed_train or not processed_val or not processed_test:
        raise ValueError(f"One of the processed datasets (train, val, or test) is empty after alignment and MT key check for '{model_mt_key}'. "
                         "Please verify data alignment by 'unit_id' and the presence of the MT generator key in JSON files.")

    # Shuffle the structured data
    random.shuffle(processed_train)
    random.shuffle(processed_val)
    random.shuffle(processed_test)

    # Take out a subset of the (aligned) test data
    n_test_actual = int(len(processed_test) * test_size)
    final_test_set_structured = processed_test[:n_test_actual]
    # The rest of processed_test is combined with train and val for a larger pool to draw from
    remaining_for_train_val = processed_test[n_test_actual:]

    # Merge the train, val and the remaining part of the original test data
    full_data_structured = processed_train + processed_val + remaining_for_train_val
    random.shuffle(full_data_structured) # Shuffle again after merging

    # Split for final validation and training sets from the merged pool
    n_val_actual = int(len(full_data_structured) * val_size)
    final_val_set_structured = full_data_structured[:n_val_actual]
    final_train_set_structured = full_data_structured[n_val_actual:]

    if not final_train_set_structured or not final_val_set_structured or not final_test_set_structured:
         raise ValueError("One of the final datasets (train, val, or test) is empty after splitting. Check sizes and input data.")

    return final_train_set_structured, final_val_set_structured, final_test_set_structured


# MODIFICATION: compute_metrics for multi-class
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Use 'macro' for multi-class to average metrics per class without considering label imbalance
    # Use 'weighted' to account for label imbalance.
    # 'micro' would be equivalent to accuracy in multi-class.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0 # Set zero_division to 0 or 1
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# MODIFICATION: get_sets function for 3 classes
def get_sets(
    novel_name: str,
    translator1_id: str,
    translator2_id: str,
    generator_name: str,
    lang: str,
    test_size: float,
    val_size: float,
    output_dir: Path,
):
    # Create the flat lists for the Hugging Face Dataset object
    # Each original entry (src, mt, h1, h2) will yield 3 entries in the final dataset list
    train_data_flat = []
    val_data_flat = []
    test_data_flat = []

    if novel_name != "mock": # "mock" is a special keyword for using _get_mock_data
        # These are lists of dicts like {"unit_id": ..., "mt_text": ..., "h1_text": ..., "h2_text": ...}
        train_set_structured, val_set_structured, test_set_structured = _get_data(
            novel_name=novel_name,
            translator1_id=translator1_id,
            translator2_id=translator2_id,
            generator_name=generator_name,
            lang=lang,
            test_size=test_size,
            val_size=val_size,
        )

        # Save the structured data (containing mt, h1, h2 texts per source entry)
        # This is useful for traceability and analysis
        with open(output_dir / "train_structured_input.json", "w") as f:
            json.dump(train_set_structured, f, indent=4, ensure_ascii=False)
        with open(output_dir / "val_structured_input.json", "w") as f:
            json.dump(val_set_structured, f, indent=4, ensure_ascii=False)
        with open(output_dir / "test_structured_input.json", "w") as f:
            json.dump(test_set_structured, f, indent=4, ensure_ascii=False)

        for example in train_set_structured:
            train_data_flat.append({'text': example['mt_text'], 'label': LBL2ID["MT"]})
            train_data_flat.append({'text': example['h1_text'], 'label': LBL2ID["H1"]})
            train_data_flat.append({'text': example['h2_text'], 'label': LBL2ID["H2"]})

        for example in val_set_structured:
            val_data_flat.append({'text': example['mt_text'], 'label': LBL2ID["MT"]})
            val_data_flat.append({'text': example['h1_text'], 'label': LBL2ID["H1"]})
            val_data_flat.append({'text': example['h2_text'], 'label': LBL2ID["H2"]})

        for example in test_set_structured:
            test_data_flat.append({'text': example['mt_text'], 'label': LBL2ID["MT"]})
            test_data_flat.append({'text': example['h1_text'], 'label': LBL2ID["H1"]})
            test_data_flat.append({'text': example['h2_text'], 'label': LBL2ID["H2"]})
    else:
        # Mock data for testing (already flat)
        train_data_flat, val_data_flat, test_data_flat = _get_mock_data()
        # Save the generated mock flat lists for inspection
        with open(output_dir / "train_mock_flat.json", "w") as f:
            json.dump(train_data_flat, f, indent=4, ensure_ascii=False)
        with open(output_dir / "val_mock_flat.json", "w") as f:
            json.dump(val_data_flat, f, indent=4, ensure_ascii=False)
        with open(output_dir / "test_mock_flat.json", "w") as f:
            json.dump(test_data_flat, f, indent=4, ensure_ascii=False)

    return train_data_flat, val_data_flat, test_data_flat


def main(
    novel_name: str,                 # e.g., "pinocchio_it"
    lang: str = "eng",
    generator_name: str = "meta-llama/Llama-3.1-8B-Instruct",   # meta-llama/Llama-3.1-8B-Instruct
    model_name: str = "FacebookAI/xlm-roberta-large",
    epochs: int = 10,
    max_len: int = 256,
    test_size: float = 0.12,         # Fraction of original *aligned* test data to be used as final test set
    val_size: float = 0.10,          # Fraction of *remaining* (train+val+rest_of_test) for validation
    base_output_dir: str = "/scratch/$ME/steer_outputs_scratch",
    seed: int = 25,
    # oth_persona: str | None = None, # MODIFICATION: Removed, no longer used
):
    n_labels = len(LBL2ID)
    
    # assert novel name not ending with _tra1 or _tra2
    if novel_name.endswith("_tra1") or novel_name.endswith("_tra2"):
        raise ValueError("Novel name should not end with '_tra1' or '_tra2'. Use the base name instead.")

    print(f"--- Running 3-Way Classifier (MT vs H1 vs H2) ---")
    print(f"Novel: [bold cyan]{novel_name}[/], Lang: [bold cyan]{lang}[/]")
    print(f"MT Generator: [bold magenta]{generator_name}[/]")
    print(f"----------------------------------------------------")

    _set_seeds(seed)
    translator1_id: str = 'tra1'
    translator2_id: str = 'tra2'

    classification_scenario_name = f"{novel_name}_{translator1_id}_vs_{translator2_id}_vs_MT"

    # save still in tra1
    output_dir = Path(base_output_dir) / f'{novel_name}_tra1' / generator_name.split('/')[-1] / "classifier_3way_out"
    
    # Remove existing output_dir if it's a symlink, to avoid writing into linked location if it's a mistake.
    if output_dir.is_symlink():
        print(f"Output directory {output_dir} is a symlink. Removing it.")
        output_dir.unlink()
    elif output_dir.is_dir():
        print(f"Output directory {output_dir} exists. Content might be overwritten.")

    os.makedirs(output_dir, exist_ok=True)

    # this classifier will be used by tra1, tra2 and tra1VStra2
    # will be saved in tra1 but both tra2 and tra1vstra2 will have symlinks to it

    # create symlink for tra2 

    symlink_tra2 = Path(base_output_dir) / f'{novel_name}_tra2' / generator_name.split('/')[-1] / "classifier_3way_out"
    os.makedirs(symlink_tra2, exist_ok=True)
    # remove first if it exists
    if symlink_tra2.is_symlink():
        print(f"Symlink for tra2 already exists. Removing it.")
        symlink_tra2.unlink()
    elif symlink_tra2.is_dir():
        shutil.rmtree(symlink_tra2)
    os.symlink(output_dir, symlink_tra2)

    symlink_VS = Path(base_output_dir) / f'{novel_name}_tra1-VS-{novel_name}_tra2' / generator_name.split('/')[-1] / "classifier_3way_out"
    # remove first if it exists
    if symlink_VS.is_symlink():
        print(f"Symlink for tra1VStra2 already exists. Removing it.")
        symlink_VS.unlink()
    elif symlink_VS.is_dir():
        shutil.rmtree(symlink_VS)
    os.symlink(output_dir, symlink_VS)


    print("\n--- Loading and Preparing Data ---")
    train_data, val_data, test_data = get_sets(
        novel_name=novel_name,
        translator1_id=translator1_id,
        translator2_id=translator2_id,
        generator_name=generator_name,
        lang=lang,
        test_size=test_size,
        val_size=val_size,
        output_dir=output_dir,
    )

    print("\n--- Data Counts (Flat List for Training) ---")
    # These counts are 3x the number of original source sentences if all are aligned
    print(f'Train examples: {len(train_data)}, approx. per label: {len(train_data)//n_labels if n_labels > 0 else "N/A"}')
    print(f'Validation examples: {len(val_data)}, approx. per label: {len(val_data)//n_labels if n_labels > 0 else "N/A"}')
    print(f'Test examples: {len(test_data)}, approx. per label: {len(test_data)//n_labels if n_labels > 0 else "N/A"}')

    if not train_data or not val_data or not test_data:
        raise ValueError("One of the datasets (train, val, test) is empty before creating Dataset objects. Check data loading and splitting.")

    raw_datasets = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })

    raw_datasets = raw_datasets.shuffle(seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=os.cpu_count() // 2 if os.cpu_count() else 1)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=n_labels,
        id2label={i: lbl for lbl, i in LBL2ID.items()}, # For model card and inference
        label2id=LBL2ID,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: [bold]{device}[/]")
    model.to(device)

    gradient_accumulation_steps = 1

    training_args = TrainingArguments(
        run_name=f'{classification_scenario_name}_{lang}_{generator_name.split("/")[-1]}',
        output_dir=output_dir.as_posix(),
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch", 
        save_total_limit=2, # Keep best and current epoch
        logging_dir=f"{output_dir}/logs",
        logging_steps=max(1, (len(tokenized_datasets["train"]) // (16 * gradient_accumulation_steps)) // 20), # Log ~20 times per epoch
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        # report_to="wandb" if "WANDB_API_KEY" in os.environ else "none", # Auto-detect wandb
        dataloader_num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] # Stop if no improvement after 3 epochs
    )

    print("\n--- Starting Training ---")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    # Best model is already saved by `load_best_model_at_end=True` and `save_strategy`
    # Explicitly save again to a clearly named directory if desired, or rely on Trainer's checkpoint.
    best_model_path = output_dir / "best_model_3way"
    trainer.save_model(best_model_path.as_posix()) 
    print(f"Best model saved to: {best_model_path}")


    print("\n--- Evaluating on Validation Set (with best model) ---")
    eval_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    print("Validation Metrics:")
    print(eval_metrics) # rich print
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("\n--- Evaluating on Test Set (with best model) ---")
    test_predictions = trainer.predict(tokenized_datasets["test"])
    # test_predictions.metrics will contain metrics like "test_loss", "test_accuracy", "test_f1" etc.
    # as calculated by our compute_metrics function passed to Trainer.
    # The keys will be prefixed with "test_" by the predict method.
    test_metrics = test_predictions.metrics 
    print("Test Metrics:")
    print(test_metrics) # rich print
    trainer.log_metrics("test", test_metrics) 
    # Save our computed metrics dictionary to a JSON file for clarity
    with open(output_dir / "test_metrics_results.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    # Tokenizer and README should be saved there too.
    tokenizer.save_pretrained(best_model_path)

    # Copy the flat test data (text, label) that was used for prediction
    flat_test_set_for_upload = best_model_path / "test_data_flat.json"
    with open(flat_test_set_for_upload, "w") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print(f"\nModel training complete. All results saved in: [bold green]{output_dir}[/]")


    # Copy the structured test input data as well for reproducibility
    structured_test_input_path_source = output_dir / "test_structured_input.json"
    if structured_test_input_path_source.exists():
        shutil.copy(structured_test_input_path_source, best_model_path / "test_structured_input.json")
    
if __name__ == "__main__":
    # For real data:
    # python your_script_name.py --novel_name "pinocchio" --generator_name "google/gemma-2-9b-it" 
    fire.Fire(main)



