import fire
import torch
import json
import numpy as np
import random
import os
import copy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from typing import List, Any, Callable
from rich import print
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from functools import partial

from src.wrappers import NNsightWrapper
from src.train_persona import TEMPLATE_MAP, LANG_MAP, MT_template
from src.metric import _predict_mt_ht_3way


layers_to_save = {
    'gemma-2-2b-it': 13,
    'gemma-2-9b-it': 21,
}


class SimpleLinearClassifier(nn.Module):
    """A simple linear classifier for binary classification."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)



def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(
    persona_name: str,
    model_name_or_path: str,
    n_icl_examples: int = 20,
):
    # loat train, val and test

    train_path = f"./data/train/eng/{persona_name}.json"
    val_path = f"./data/val/eng/{persona_name}.json"
    test_path = f"./data/test/eng/{persona_name}.json"

    with open(train_path, "r") as f:
        train_data = json.load(f)
    with open(val_path, "r") as f:
        val_data = json.load(f)
    with open(test_path, "r") as f:
        test_data = json.load(f)

    # replace MTs with model's MT
    model_key = model_name_or_path.split("/")[-1]
    for i in range(len(train_data)):
        train_data[i]['mt_text'] = train_data[i][model_key]
    for i in range(len(val_data)):
        val_data[i]['mt_text'] = val_data[i][model_key]
    for i in range(len(test_data)):
        test_data[i]['mt_text'] = test_data[i][model_key]

    # concatenate all data
    all_data = train_data + val_data + test_data

    # randomize
    random.shuffle(all_data)

    # limit to 150 examples
    all_data = all_data[:150]

    # build chats
    all_chats_edit = []
    all_chats_noedit = []
    for idx in range(0, len(all_data) - n_icl_examples):

        current_edit_chat = []
        current_noedit_chat = []

        examples_to_use_icl = all_data[idx:idx + n_icl_examples]
        random.shuffle(examples_to_use_icl)

        for example in examples_to_use_icl:
            prompt_std = TEMPLATE_MAP["translate"].format(
                src_text=example["src_text"],
                lang=LANG_MAP[example["tgt_lang"]],
                mt_text=example["mt_text"],
            )

            current_edit_chat.append({"role": "user", "content": prompt_std})
            current_noedit_chat.append({"role": "user", "content": prompt_std})

            current_edit_chat.append({"role": "assistant", "content": example["pe_text"]})
            current_noedit_chat.append({"role": "assistant", "content": example["mt_text"]})
        
        # add +1 example at the end
        last_example = all_data[idx + n_icl_examples]
        last_prompt = TEMPLATE_MAP["translate"].format(
            src_text=last_example["src_text"],
            lang=LANG_MAP[last_example["tgt_lang"]],
            mt_text=last_example["mt_text"],
        )

        current_edit_chat.append({"role": "user", "content": last_prompt})
        current_noedit_chat.append({"role": "user", "content": last_prompt})

        all_chats_edit.append(current_edit_chat)
        all_chats_noedit.append(current_noedit_chat)

    return all_chats_edit, all_chats_noedit


def classify_out(
    classifier: Any,    # partial call
    tokenizer: Any,     # partial call
    out: str,           # run time call
) -> str:

    """Classifies the output of the model using the classifier."""
    dict_res = _predict_mt_ht_3way(
        classifier=classifier,
        classifier_tokenizer=tokenizer,
        input_text=out,
    )
    label = dict_res["predicted"]
    if label == "MT":
        return "MT"
    return "PE"         # label is H1 or H2


def collect_flipped_activations(
    model: NNsightWrapper,
    conversations_edit: list[list],   # list of conversations
    conversations_noedit: list[list],   # list of conversations
    hook_points: list[str],
    output_path: Path,
    classify_fn: Callable[[str], str],  # function to classify the conversation
    max_new_tokens: int = 512,
):

    flipped_activations = {hp: {"MT": [], "PE": []} for hp in hook_points}
    flip_count = 0

    saved = []
    for conversation_edit, conversation_noedit in (pbar := tqdm(
        zip(conversations_edit, conversations_noedit),
        desc="Collecting flipped activations",
        total=len(conversations_edit),
    )):
        inp_tokens_edit: torch.Tensor = model.tokenizer.apply_chat_template(
            conversation_edit,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inp_tokens_noedit: torch.Tensor = model.tokenizer.apply_chat_template(
            conversation_noedit,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        try:
            raw_edit_cache, edit_out_tokens = model.cache_forward(
                inp_tokens=inp_tokens_edit,
                hook_points=hook_points,
                max_new_tokens=max_new_tokens,
                return_out_tokens=True,
            )
            # select only the last token of the prompt
            edit_cache = raw_edit_cache[0]

            edit_inp_len = inp_tokens_edit.shape[-1]
            edit_output = model.tokenizer.decode(edit_out_tokens[0][edit_inp_len:], skip_special_tokens=True)
            edit_label = classify_fn(
                edit_output
            )
            raw_noedit_cache, noedit_out_tokens = model.cache_forward(
                inp_tokens=inp_tokens_noedit,
                hook_points=hook_points,
                max_new_tokens=max_new_tokens,
                return_out_tokens=True,
            )
            # select only the last token of the prompt
            noedit_cache = raw_noedit_cache[0]

            noedit_inp_len = inp_tokens_noedit.shape[-1]
            noedit_output = model.tokenizer.decode(noedit_out_tokens[0][noedit_inp_len:], skip_special_tokens=True)
            noedit_label = classify_fn(
                noedit_output
            )
            
            if noedit_label == "MT" and edit_label == "PE":
                # the conversation was flipped by the edit ICL
                # keep activations of the last token of the prompt
            
                flip_count += 1
                pbar.set_description(f"Collecting flipped activations ({flip_count = })")
                for hp in hook_points:
                    flipped_activations[hp]["MT"].append(
                        noedit_cache[hp].detach().cpu()
                    )
                    flipped_activations[hp]["PE"].append(
                        edit_cache[hp].detach().cpu()
                    )

            # here for debugging, i shouldn't care about what are the flipped activations
            saved.append(
                {
                    "conversation_edit": conversation_edit[-1]["content"],
                    "edit_output": edit_output,
                    "edit_label": edit_label,
                    "conversation_noedit": conversation_noedit[-1]["content"],
                    "noedit_output": noedit_output,
                    "noedit_label": noedit_label,
                }
            )
            with open(output_path / 'tmp_save_flipped.json', 'w') as f:
                json.dump(saved, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error processing conversation: {e}")
            # print(f"Conversation edit: {conversation_edit}")
            # print(f"Conversation noedit: {conversation_noedit}")
            print(f"Skipping this conversation due to error.")
            continue


    print(f"Flipped activations: {flip_count} / {len(conversations_edit)} = {flip_count / len(conversations_edit) * 100:.2f}%")
    return flipped_activations


def train_and_evaluate_classifier(
    mt_activations: List[torch.Tensor],
    pe_activations: List[torch.Tensor],
    random_state: int,
    val_test_split_ratio: float = 0.15, # Fraction for validation AND test set each
    learning_rate: float = 0.001,
    num_epochs: int = 20, # Number of training epochs
    batch_size: int = 16,
    patience: int = 3, # Early stopping patience
    save_path: str = None, # Path to save the model and metadata
) -> tuple[float, float]:
    """
    Trains a PyTorch linear classifier with early stopping based on validation accuracy

    Args:
        mt_activations: List of activation tensors for the 'MT' class (label 0).
        pe_activations: List of activation tensors for the 'PE' class (label 1).
        random_state: Seed for reproducibility.
        val_test_split_ratio: Fraction of data for validation and test sets EACH (e.g., 0.15 means 15% val, 15% test, 70% train).
        learning_rate: Optimizer learning rate.
        num_epochs: Maximum number of training epochs.
        batch_size: Training/evaluation batch size.
        patience: Number of epochs to wait for validation improvement before stopping.

    Returns:
        The accuracy on the test set using the best model checkpoint found during training.
    """
    if not mt_activations or not pe_activations:
        print("Warning: Insufficient data for one or both classes. Cannot train classifier.")
        return 0.0

    # Combine data and create labels (0 for MT and 1 for PE)
    X_list = mt_activations + pe_activations
    y_list = [0] * len(mt_activations) + [1] * len(pe_activations)

    # stack tensors into a single np array
    X = torch.stack(X_list).float() # Ensure float type
    y = torch.tensor(y_list).float().unsqueeze(1) # Target shape [N, 1], float for BCEWithLogitsLoss

    input_dim = X.shape[1]
    print(f"Preparing data: {X.shape[0]} samples ({len(mt_activations)} MT, {len(pe_activations)} PE). Feature dim: {input_dim}")

    test_size = val_test_split_ratio # e.g., 0.15 for test set
    val_size_rel_to_trainval = val_test_split_ratio / (1.0 - test_size) # e.g., 0.15 / (1 - 0.15) = 0.15 / 0.85

    # first separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # second separate train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_rel_to_trainval, random_state=random_state, stratify=y_train_val
    )
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
         raise ValueError("One of the data splits is empty after splitting.")

    # data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=(len(train_dataset) % batch_size == 1))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # train stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLinearClassifier(input_dim).to(device)
    # Using BCEWithLogitsLoss combines Sigmoid layer and BCELoss for better numerical stability
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # early stop
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_state = None # Store the state_dict of the best model
    actual_epochs_run = 0


    print(f"Training on {device} for max {num_epochs} epochs (Patience: {patience})...")
    for epoch in range(num_epochs):
        actual_epochs_run += 1
        model.train()
        train_correct, train_total = 0, 0
        # Simplified training loop logging
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            predicted = (outputs > 0).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        # val
        val_correct, val_total = 0, 0
        val_logits_list, val_labels_list = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs > 0).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                val_logits_list.append(outputs.cpu())
                val_labels_list.append(labels.cpu())

        current_val_acc = (val_correct / val_total) if val_total > 0 else 0.0

        # AUROC
        val_logits = torch.cat(val_logits_list).numpy()
        val_labels = torch.cat(val_labels_list).numpy()
        try:
            current_val_auc = roc_auc_score(val_labels, val_logits)
        except ValueError:
            current_val_auc = float('nan')  # Happens if only one class present

        train_acc = (train_correct / train_total) if train_total > 0 else 0.0
        print(f"Epoch {epoch+1:02d}/{num_epochs} - Train Acc: {train_acc:.4f}, Val Acc: {current_val_acc:.4f}, Val AUROC: {current_val_auc:.4f}", end="")

        # early stop
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            # Use deepcopy to ensure the state is saved correctly at this point
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f" -> New best validation accuracy! Saving model.")
        else:
            epochs_no_improve += 1
            print(f" (No improvement for {epochs_no_improve} epoch(s))")
            if epochs_no_improve >= patience:
                print(f"[yellow]Early stopping triggered after {epoch + 1} epochs.[/yellow]")
                break

    # final evaluation on test
    if best_model_state is None:
        print("[yellow]Warning: No best model state saved (validation accuracy never improved or training didn't run). Evaluating initial model on test set.[/yellow]")
        if model is None: # Should not happen unless training failed very early
             print("[red]Error: Model is None before final evaluation.[/red]")
             return 0.0, 0.0
    else:
        print(f"Loading best model state (Val Acc: {best_val_acc:.4f}) for test evaluation.")
        model.load_state_dict(best_model_state)

    model.eval()

    test_correct, test_total = 0, 0
    test_logits_list, test_labels_list = [], []

    print("Evaluating on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0).float()
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            test_logits_list.append(outputs.cpu())
            test_labels_list.append(labels.cpu())

    final_test_accuracy: float = (test_correct / test_total) if test_total > 0 else 0.0

    test_logits = torch.cat(test_logits_list).numpy()
    test_labels = torch.cat(test_labels_list).numpy()
    try:
        test_auroc: float = roc_auc_score(test_labels, test_logits)
    except ValueError:
        test_auroc = float('nan')

    print(f"Final Test Accuracy: {final_test_accuracy:.4f}, Test AUROC: {test_auroc:.4f}")

    # Save the final model state if needed
    if save_path and best_model_state is not None:
        checkpoint = {
             'input_dim': input_dim,
            'model_state_dict': best_model_state,
            'best_val_acc': best_val_acc,
            'final_test_accuracy': final_test_accuracy,
            'test_auroc': test_auroc,
            'random_state': random_state,
            'learning_rate': learning_rate,
            'num_epochs_trained': actual_epochs_run,
            'batch_size': batch_size,
            'patience': patience,
            'val_test_split_ratio': val_test_split_ratio,
            'class_labels': {'MT': 0, 'PE': 1} # Example of saving class mapping
        }
        torch.save(checkpoint, save_path)
        print(f"Model and metadata saved to {save_path}")

    return final_test_accuracy, test_auroc

# python -m src.probing pinocchio_it_tra1 google/gemma-2-9b-it
def main(
    persona_name: str,              # classifier path best model contains train, test, val
    model_name_or_path: str = "google/gemma-2-2b-it",
    n_icl_examples: int = 15,
    low_mem: bool = False,
    hook_points: str | list = "all",
    seed: int = 25,
    max_new_tokens: int = 1024,
    val_set_fraction: float = 0.2, # Fraction for validation set
    learning_rate: float = 0.001,
    num_epochs: int = 20,
    batch_size: int = 16,
):
    print('-----------------------------')
    print(f"Persona: {persona_name}")
    print('-----------------------------')
    set_seed(seed)

    output_path = Path(f"./steer_outputs_scratch/{persona_name}/{model_name_or_path.split('/')[-1]}/probing")
    os.makedirs(output_path, exist_ok=True)

    model = NNsightWrapper(
        model_name=model_name_or_path,
        load_in_4bit=low_mem,
    )

    conversations_edit, conversations_noedit = load_data(
        persona_name=persona_name, model_name_or_path=model_name_or_path,
        n_icl_examples=n_icl_examples,
    )
    print(f"Total usable examples: {len(conversations_noedit)}")


    if hook_points == "all":
        n_layers = model.config.num_hidden_layers
        hook_points = [
            f"model.layers.{i}" for i in range(n_layers)
        ]

    classifier = AutoModelForSequenceClassification.from_pretrained(
        f'./steer_outputs_scratch/{persona_name}/{model_name_or_path.split("/")[-1]}/classifier_3way_out/best_model_3way/'
    )
    class_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    classifier.to(model.device)
    classifier.eval()
    
    classify_fn = partial(
        classify_out,
        classifier=classifier,
        tokenizer=class_tokenizer
    )

    flipped_activations = collect_flipped_activations(
        model=model,
        conversations_edit=conversations_edit,
        conversations_noedit=conversations_noedit,
        hook_points=hook_points,
        output_path=output_path,
        classify_fn=lambda x: classify_fn(out=x),
        max_new_tokens=max_new_tokens,
    )

    # --- Train Classifiers per Hook Point ---
    print("\n--- Training PyTorch Classifiers per Hook Point ---")
    results = {}

    for hp in tqdm(hook_points, desc="Training Classifiers", dynamic_ncols=True):
        print(f"\nProcessing hook point: {hp}")
        mt_acts = flipped_activations.get(hp, {}).get('MT', [])
        pe_acts = flipped_activations.get(hp, {}).get('PE', [])

        if not mt_acts or not pe_acts:
            print(f"Skipping {hp}: Not enough data (MT: {len(mt_acts)}, PE: {len(pe_acts)})")
            results[hp] = 0.0
            continue

        # Optional: Add explicit balancing/subsampling here if needed before split
        min_samples = min(len(mt_acts), len(pe_acts))
        if len(mt_acts) != len(pe_acts):
             print(f"Warning: Dataset for {hp} is imbalanced ({len(mt_acts)} MT, {len(pe_acts)} PE). Using {min_samples} samples per class for training/validation.")
             # Subsample the larger class before splitting
             random.shuffle(mt_acts) # Shuffle before subsampling
             random.shuffle(pe_acts)
             mt_acts = mt_acts[:min_samples]
             pe_acts = pe_acts[:min_samples]

        is_to_save = False
        if 'gemma-2-2b-it' in model_name_or_path:
            if '13' in hp:
                is_to_save = True
        elif 'gemma-2-9b-it' in model_name_or_path:
            if '21' in hp:
                is_to_save = True

        # Call the updated training function
        accuracy, auroc = train_and_evaluate_classifier(
            mt_activations=mt_acts,
            pe_activations=pe_acts,
            val_test_split_ratio=0.15,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            random_state=seed,
            save_path=output_path / f"{hp}_classifier.pth" if is_to_save else None,
        )
        results[hp] = (accuracy, auroc)

    # save
    with open(
        output_path / "probing_res.tsv",
        'w', encoding='utf-8'
    ) as f:
        f.write("HookPoint\tAccuracy\tAUROC\n")
        for hp, (acc, au) in results.items():
            f.write(f"{hp}\t{acc:.4f}\t{au:.4f}\n")

    print("Results")
    for hp, (acc, au) in results.items():
        print(f"- {hp}: {acc:.4f}\t{results[hp][1]:.4f}")



if __name__ == "__main__":
    fire.Fire(main)

