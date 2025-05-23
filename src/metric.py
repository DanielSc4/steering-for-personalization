from typing import Any
from pathlib import Path
from rich import print
from rich.console import Console
from rich.table import Table
from rich import print, box
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# avoid comet verbose logging and warnings
import logging
import warnings
warnings.filterwarnings("ignore", message="The `srun` command is available on your system but is not used.")

from evaluate import load
import scipy.stats
from tqdm import tqdm
import numpy as np
import torch
import json
import sys
import time
from comet import download_model, load_from_checkpoint
import os
import contextlib
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# Load the evaluation metric
ter = load("ter")
bleu = load("bleu")
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

BASE_MODEL_NAME = "xlm-roberta-large"


def _get_ci(
    data: list[float],
    confidence: float = 0.95,
) -> float:
    
    a = 1.0 * np.array(data)    # make sure it's a float array
    n = len(a)
    se = scipy.stats.sem(a)     # standard error of the mean
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)    # confidence interval
    return h


def _compute_ter(
    ref: str,
    hp: str,
) -> dict[str, Any]:

    result_dict: dict[str, Any] = ter.compute(
        predictions=[hp],
        references=[[ref]],
        case_sensitive=False,
    )
    # e.g. {"score": 30.76923076923077, "num_edits": 8, "ref_length": 26.0}
    return result_dict


def _compute_bleu(
    pred: str,
    ref: str,
) -> dict[str, Any]:
    if pred == "":
        pred = "---"    # score will be zero
    if ref == "":
        ref = "---"     # score will be zero
    try:
        result_dict: dict[str, Any] = bleu.compute(
            predictions=[pred],
            references=[[ref]],
        )
    except Exception as e:
        print(f"[eval] Error computing BLEU: {ref = } | {pred = } | {e}")
        result_dict = {
            "bleu": 0.0,
            "counts": [0, 0, 0, 0],
            "precisions": [0.0, 0.0, 0.0, 0.0],
            "bp": 1.0,
            "sys_len": 26,
            "ref_len": 26,
        }
    # e.g. {"score": 0.0, "counts": [0, 0, 0, 0], "precisions": [0.0, 0.0, 0.0, 0.0], "bp": 1.0, "sys_len": 26, "ref_len": 26}
    return result_dict


def _predict_mt_ht(
    classifier,
    classifier_tokenizer,
    input_text: str,
):
    MAP_LABELS = {0: "MT", 1: "PE"}     # PE means that the classifier thinks the text is a post-edited MT output
    with torch.no_grad():
        inputs = classifier_tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        # move to GPU
        inputs = {k: v.to(classifier.device) for k, v in inputs.items()}

        outputs = classifier(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu()
        preds = torch.argmax(probs, dim=-1)

        predicted = {
            i: round(probs[i].item(), 4)
            for i, _ in enumerate(probs)
        }   # {0: 0.1234, 1: 0.8766}

    return {
        "predicted": MAP_LABELS[preds.item()],
        "probs": predicted,
    }

def _predict_mt_ht_3way(
    classifier,
    classifier_tokenizer,
    input_text: str,
):
    MAP_LABELS = {
        0: "MT",
        1: "H1",
        2: "H2"
    }

    with torch.no_grad():
        inputs = classifier_tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        # move to GPU if available
        inputs = {k: v.to(classifier.device) for k, v in inputs.items()}

        outputs = classifier(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu()
        preds = torch.argmax(probs, dim=-1)

        predicted = {
            i: round(probs[i].item(), 4)
            for i in range(len(probs))
        }

    return {
        "predicted": MAP_LABELS[preds.item()],
        "probs": {MAP_LABELS[i]: predicted[i] for i in predicted},
    }



def eval_out(
    output: list[dict[str, Any]],
    fields: list[tuple[str, str]],
    model_name: str,
    verbose: bool = True,
    experiment_path: Path | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_3way = None
    classifier_3way_tokenizer = None
    classifier_test_ids_3way = None

    if experiment_path:         # TOOD: change experiment_path to link at the same 3way classifier wi
        classifier_path = experiment_path / model_name.split('/')[-1] / "classifier_3way_out"
        checkpoint_name = "best_model_3way"
        if classifier_path.exists():
            with open(classifier_path / "test_structured_input.json", "r") as f:
                test_set = json.load(f)
            classifier_test_ids_3way = [example["unit_id"] for example in test_set]
            if classifier_path.exists():
                print(f"[eval] Loading classifier from {classifier_path / checkpoint_name}...")
                classifier_3way = AutoModelForSequenceClassification.from_pretrained(classifier_path / checkpoint_name)
                classifier_3way_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
                classifier_3way.to(device)
                classifier_3way.eval()

    for instance in tqdm(output, desc="Evaluating", dynamic_ncols=True):
        # classifier
        if classifier_3way:
            if instance["unit_id"] in classifier_test_ids_3way:
                instance["classifier_3way"] = {
                    "original_mt": _predict_mt_ht_3way(classifier_3way, classifier_3way_tokenizer, instance["original_mt"]),
                    "original_pe": _predict_mt_ht_3way(classifier_3way, classifier_3way_tokenizer, instance["original_pe"]),
                    "output_clean": _predict_mt_ht_3way(classifier_3way, classifier_3way_tokenizer, instance["output_clean"]),
                }


        # comet
        instance["comet"] = {}
        # Prepare COMET input samples
        comet_inputs_mt = [{
            "src": instance["original_src"],
            "mt": instance["original_mt"],
            "ref": instance["original_pe"]
        }]
        comet_inputs_out = [{
            "src": instance["original_src"],
            "mt": instance["output_clean"],
            "ref": instance["original_pe"]
        }]
        # get rid of comet output
        
        pl_logger = logging.getLogger("pytorch_lightning")
        fabric_logger = logging.getLogger("lightning_fabric")
        original_pl_level, original_fabric_level = pl_logger.level, fabric_logger.level

        try:
            # --- Silence loggers for loading and prediction ---
            pl_logger.setLevel(logging.ERROR)
            fabric_logger.setLevel(logging.ERROR)
            with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
                comet_scores_mt = comet_model.predict(
                    comet_inputs_mt, 
                    batch_size=1, 
                    gpus=1 if torch.cuda.is_available() else 0,
                    progress_bar=False,
                )
                comet_scores_out = comet_model.predict(
                    comet_inputs_out, 
                    batch_size=1, 
                    gpus=1 if torch.cuda.is_available() else 0,
                    progress_bar=False,
                )
        finally:
            pl_logger.setLevel(original_pl_level)
            fabric_logger.setLevel(original_fabric_level)

        instance["comet"]["original_mt"] = comet_scores_mt['scores'][0]
        instance["comet"]["output_clean"] = comet_scores_out['scores'][0]


    table = Table(title="Scores", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("metric", justify="center", style="cyan")
    table.add_column("ref", justify="left", style="green")
    table.add_column("pred", justify="left", style="green")
    table.add_column("mean", justify="center", style="magenta")
    table.add_column("mean-ci", justify="center", style="magenta")
    table.add_column("median", justify="center", style="magenta")

    # for i, field in enumerate(fields):
    #     field_name = f'{field[0]}-{field[1]}'
    #     ter_scores = [instance["ter"][field_name] for instance in output]
    #     mean = np.mean(ter_scores)
    #     ci = _get_ci(ter_scores)
    #     median = np.median(ter_scores)
    #     # print(f'\t{field_name}: \t{mean:.4f} ± {ci:.2f} |\t {median:.4f}')
    #     table.add_row(
    #         "TER" if i == 0 else "",
    #         field[0],
    #         field[1],
    #         f'{mean:.4f}',
    #         f'{ci:.2f}',
    #         f'{median:.4f}',
    #     )

    # table.add_row("","","","","","")

    # for i, field in enumerate(fields):
    #     field_name = f'{field[0]}-{field[1]}'
    #     bleu_scores = [instance["bleu"][field_name] for instance in output]
    #     mean = np.mean(bleu_scores)
    #     ci = _get_ci(bleu_scores)
    #     median = np.median(bleu_scores)
    #     # print(f'\t{field_name}: \t{mean:.4f} ± {ci:.2f} |\t {median:.4f}')
    #     table.add_row(
    #         "BLEU" if i == 0 else "",
    #         field[0],
    #         field[1],
    #         f'{mean:.4f}',
    #         f'{ci:.2f}',
    #         f'{median:.4f}',
    #     )

    if classifier_3way:
        table.add_row("","","","","","")
        MT_wrong_3way = 0
        PE_as_H1 = 0
        PE_as_H2 = 0
        Out_as_H1 = 0
        Out_as_H2 = 0
        Out_as_H1_only_mt_correct = 0
        Out_as_H2_only_mt_correct = 0

        tot_examples_3way = 0
        for instance in output:
            if instance["unit_id"] in classifier_test_ids_3way:
                tot_examples_3way += 1
                if instance["classifier_3way"]["original_mt"]["predicted"] != "MT":
                    # examples where the classifier thinks the original MT is HT (wrong)
                    MT_wrong_3way += 1
                if instance["classifier_3way"]["original_pe"]["predicted"] == "H1":
                    # examples where the classifier thinks the original PE is from HT1 (correct)
                    PE_as_H1 += 1
                if instance["classifier_3way"]["original_pe"]["predicted"] == "H2":
                    # examples where the classifier thinks the original PE is from HT2 (correct)
                    PE_as_H2 += 1
                if instance["classifier_3way"]["output_clean"]["predicted"] == "H1":
                    # examples where the classifier thinks the model's output is from HT1 (correctly steered)
                    Out_as_H1 += 1
                if instance["classifier_3way"]["output_clean"]["predicted"] == "H2":
                    # examples where the classifier thinks the model's output is from HT2 (correctly steered)
                    Out_as_H2 += 1
                if instance["classifier_3way"]["original_mt"]["predicted"] == "MT":
                    # examples where the classifier thinks the original MT is MT and the model's output is HT (correctly steered)
                    # this means that the whatever technique was used, it successfully tricked the classifier (what we want)
                    if instance["classifier_3way"]["output_clean"]["predicted"] == "H1":
                        Out_as_H1_only_mt_correct += 1
                    if instance["classifier_3way"]["output_clean"]["predicted"] == "H2":
                        Out_as_H2_only_mt_correct += 1

        MT_wrong_3way = MT_wrong_3way / tot_examples_3way
        PE_as_H1 = PE_as_H1 / tot_examples_3way
        PE_as_H2 = PE_as_H2 / tot_examples_3way
        Out_as_H1 = Out_as_H1 / tot_examples_3way
        Out_as_H2 = Out_as_H2 / tot_examples_3way
        Out_as_H1_only_mt_correct = Out_as_H1_only_mt_correct / tot_examples_3way
        Out_as_H2_only_mt_correct = Out_as_H2_only_mt_correct / tot_examples_3way

        table.add_row(
            f"Class. (supp: {tot_examples_3way})",
            "MT as H*↓",
            "",
            f'{MT_wrong_3way:.4f}',
            "",
            "",
        )
        table.add_row(
            "",
            "PE as H1↑",
            "",
            f'{PE_as_H1:.4f}',
            "",
            "",
        )
        table.add_row(
            "",
            "PE as H2↑",
            "",
            f'{PE_as_H2:.4f}',
            "",
            "",
        )
        table.add_row(
            "",
            "Out as H1↑",
            "",
            f'{Out_as_H1:.4f}',
            "",
            "",
        )
        table.add_row(
            "",
            "Out as H2↑",
            "",
            f'{Out_as_H2:.4f}',
            "",
            "",
        )
        table.add_row(
            "",
            "Out as H1↑",
            "(when MT is classified as MT)",
            f'{Out_as_H1_only_mt_correct:.4f}',
            "",
            "",
        )
        table.add_row(
            "",
            "Out as H2↑",
            "(when MT is classified as MT)",
            f'{Out_as_H2_only_mt_correct:.4f}',
            "",
            "",
        )
        

        table.add_row("","","","","","")
        comet_scores_mts = [instance["comet"]['original_mt'] for instance in output]
        comet_scores_outs = [instance["comet"]['output_clean'] for instance in output]

        mean_mts = np.mean(comet_scores_mts)
        ci_mts = _get_ci(comet_scores_mts)
        median_mts = np.median(comet_scores_mts)

        mean_outs = np.mean(comet_scores_outs)
        ci_outs = _get_ci(comet_scores_outs)
        median_outs = np.median(comet_scores_outs)
        table.add_row(
            "COMET",
            "original_mt",
            "",
            f'{mean_mts.item():.4f}',
            f'{ci_mts:.2f}',
            f'{median_mts.item():.4f}',
        )
        table.add_row(
            "",
            "output_clean",
            "",
            f'{mean_outs.item():.4f}',
            f'{ci_outs.item():.2f}',
            f'{median_outs.item():.4f}',
        )

        if verbose:
            console = Console()
            console.print(table)


    return output, table










if __name__ == "__main__":
    # useful to test without running the benchmark again
    file_path = sys.argv[1]
    print(file_path)

    # file_path = "steer_outputs_scratch/pinocchio_it_tra1/Llama-3.1-8B-Instruct/pinocchio_it_tra1-25-contrastive-edit-l19-k10-a2-eval.json"

    file_path = Path(file_path)
    # file infos
    name = file_path.name

    infos = name.split("-")
    persona_name = infos[0]
    seed = infos[1]
    experiment = infos[2]
    template = infos[3]
    additional_info = '-'.join(infos[4:])

    model_name = file_path.parent.name

    print(f"[eval] Evaluating {model_name} | {experiment} | {template} | {additional_info}")

    # laod file
    with open(file_path, "r") as f:
        output = json.load(f)

    base_experiment_path = file_path.parent.parent

    # get eval
    output_eval, metric_table = eval_out(
        output=output,
        fields=[
            ("original_pe", "output_clean"),    # how close the std mt to the pe
            ("original_mt", "original_pe"),
            ("original_mt", "output_clean"),    # distance between nllb and llama
        ],
        experiment_path=base_experiment_path,
        model_name=model_name,
    )

    # overwrite the output file
    with open(file_path, "w") as f:
        json.dump(output_eval, f, indent=4, ensure_ascii=False)
    # save the metric table
    metric_table.title = f"Metrics: {experiment} - {template}"
    with open( file_path.parent / f"metric_tables.md", "a") as f:
        f.write(f"### {model_name} | {experiment}\n")
        # current time
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write(f"#### {current_time}\n")
        print(metric_table, file=f)
        f.write("\n\n")






