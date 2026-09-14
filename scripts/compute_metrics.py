# Copyright 2025 the PEFT-Factory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import json
import math
import operator
import re
import glob
import os

import evaluate
import numpy as np
import pandas as pd
from codebleu import calc_codebleu
from datasets import load_dataset
from sklearn.metrics import f1_score


def string_to_float(string, default=-1.0):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


def check_data_state(preds, targets):
    assert len(preds) == len(targets)


def binary_reverse(targets, labels):
    return [labels[0] if target == labels[1] else labels[1] for target in targets]


def em(preds, targets, labels):
    check_data_state(preds, targets)

    preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

    return {"exact_match": np.sum(preds == targets) / preds.size}


def f1(preds, targets, labels):
    preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

    labels = [x.lower() for x in labels]

    invalid_idx_mask = np.logical_and(preds != labels[0], preds != labels[1])

    preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask], labels)

    label_to_id = {label: idx for idx, label in enumerate(labels)}

    targets = list(map(label_to_id.get, targets))
    preds = list(map(label_to_id.get, preds))

    return {"f1": f1_score(targets, preds)}


def macro_f1(preds, targets, labels):
    preds, targets = np.asarray(preds, dtype="<U16"), np.asarray(targets, dtype="<U16")

    labels = [x.lower() for x in labels]

    invalid_idx_mask = ~np.isin(preds, labels)

    preds[invalid_idx_mask] = binary_reverse(targets[invalid_idx_mask], labels)

    label_to_id = {label: idx for idx, label in enumerate(labels)}

    targets = list(map(label_to_id.get, targets))
    preds = list(map(label_to_id.get, preds))

    return {"macro_f1": f1_score(targets, preds, average="macro")}


def pearsonr(preds, targets, labels):
    metric = evaluate.load("pearsonr")

    targets = [string_to_float(t) for t in targets]
    preds = [string_to_float(p) for p in preds]

    return metric.compute(predictions=preds, references=targets)


def spearmanr(preds, targets, labels):
    metric = evaluate.load("spearmanr")

    targets = [string_to_float(t) for t in targets]
    preds = [string_to_float(p) for p in preds]

    return metric.compute(predictions=preds, references=targets)


def record(preds):
    dataset = load_dataset("kinit/peft-factory", "record", split="validation")
    metric = evaluate.load("super_glue", "record")

    predictions = [
        {"idx": dataset[i]["idx"], "prediction_text": p} for i, p in enumerate(preds)
    ]

    references = [{"idx": d["idx"], "answers": d["answers"]} for d in dataset]

    return metric.compute(predictions=predictions, references=references)


def gsm8k(preds, targets, labels):
    def extract_final_answer(text):
        if "####" in text:
            return text.split("####")[-1].strip()
        return None

    def normalize_answer(ans):
        try:
            return float(ans.replace(",", "").replace("%", ""))
        except (AttributeError, ValueError):
            return None

    format_errors = 0
    correct = 0
    total = 0
    for i in range(len(preds)):
        pred = normalize_answer(extract_final_answer(preds[i]))
        gold = normalize_answer(extract_final_answer(targets[i]))
        if pred is not None and gold is not None:
            if abs(pred - gold) < 1e-6:  # allow tiny float error
                correct += 1

        if pred is None or gold is None:
            format_errors += 1

        total += 1

    accuracy = correct / total
    return {"accuracy": accuracy, "format_errors": format_errors}


def svamp(preds, targets, labels):
    OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def safe_eval(expr):
        def _eval(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value

            if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
                return OPS[type(node.op)](_eval(node.operand))

            if isinstance(node, ast.BinOp) and type(node.op) in OPS:
                return OPS[type(node.op)](_eval(node.left), _eval(node.right))

            if isinstance(node, ast.Expr):
                return _eval(node.value)

            raise ValueError("Disallowed expression")

        tree = ast.parse(expr.strip(), mode="eval")
        return float(_eval(tree.body))

    def is_single_outer_parens(s):
        s = s.strip()
        if not (s.startswith("(") and s.endswith(")")):
            return False

        depth = 0
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    return False  # outer pair closes before end -> not a single outer wrapper
        return depth == 0

    num_re = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)$")

    def parse_equation(eq: str):
        if eq.count("=") != 1:
            return None, None, "not_single_equation"
        lhs, rhs = [p.strip() for p in eq.split("=", 1)]
        if not is_single_outer_parens(lhs):
            return None, None, "lhs_not_single_parenthesized"
        if not num_re.match(rhs.replace(",", "")):
            return None, None, "rhs_not_numeric"
        return lhs, rhs, None

    def normalize_number_str(s):
        x = float(s.replace(",", ""))
        return (
            str(int(x))
            if math.isclose(x, round(x), rel_tol=0, abs_tol=1e-9)
            else str(x)
        )

    def canon(eq):
        # remove spaces and normalize RHS number to canonical form
        lhs, rhs = [p.strip() for p in eq.split("=", 1)]
        sub = re.sub(r"\\s+", "", lhs)
        return f"{sub}={normalize_number_str(rhs)}"

    def evaluate_svamp(pred, gold=None, tol=1e-6):
        lhs, rhs, err = parse_equation(pred)
        if err:
            return {"ok": False, "reason": err, "match_gold": False}
        try:
            lhs_val = safe_eval(lhs[1:-1])  # strip outer parentheses
            rhs_val = float(rhs.replace(",", ""))
        except Exception:
            # print(e)
            return {"ok": False, "reason": "eval_error", "match_gold": False}
        math_ok = abs(lhs_val - rhs_val) < tol
        gold_match = (canon(pred) == canon(gold)) if gold else None
        return {
            "ok": math_ok,
            "reason": None if math_ok else "math_mismatch",
            "match_gold": gold_match,
        }

    total = len(preds)
    fmt_errors = 0
    math_correct = 0
    gold_exact = 0

    for i in range(total):
        r = evaluate_svamp(preds[i], targets[i])
        if r["reason"] in {
            "not_single_equation",
            "lhs_not_single_parenthesized",
            "rhs_not_numeric",
            "eval_error",
        }:
            fmt_errors += 1
        if r["ok"]:
            math_correct += 1
        if r["ok"] and r["match_gold"]:
            gold_exact += 1

    accuracy = math_correct / total if total else 0.0

    return {
        "total": total,
        "math_correct": math_correct,
        "format_errors": fmt_errors,
        "gold_exact_equation": gold_exact,
        "accuracy": accuracy,
    }

def squad_v2_metric(preds):
    """examples: the preprocessed validation split (with 'id' and 'answers').
    generated_texts: list of raw model outputs, aligned with examples."""

    metric = evaluate.load("squad_v2")
    dataset = load_dataset("kinit/peft-factory", "squad_v2", split="validation")


    def clean(text):
        # keep only the first line; instruction-tuned models like to elaborate
        return text.strip().split("\n")[0].strip()

    predictions = []
    for ex, gen in zip(dataset, preds):
        pred = clean(gen)
        is_no_answer = pred.lower().strip(' ."') == "unanswerable"
        predictions.append({
            "id": ex["id"],
            "prediction_text": "" if is_no_answer else pred,
            "no_answer_probability": 1.0 if is_no_answer else 0.0,
        })

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset]
    results = metric.compute(predictions=predictions, references=references)
    return {"exact_match": results["exact"], "f1": results["f1"]}

def codebleu_metric(preds, targets, _):
    return calc_codebleu(
        targets, preds, lang="python", weights=(0.25, 0.25, 0.25, 0.25)
    )


def find_dirs_with_timestamp(search_path: str) -> list[tuple[str, int]]:
    """
    Search for directories whose last underscore-separated segment is a Unix timestamp.

    Args:
        search_path: Root directory to search in (supports glob patterns).

    Returns:
        List of (directory_path, timestamp) tuples, sorted ascending by timestamp.
    """
    # Match any directory ending with _<digits>
    timestamp_pattern = re.compile(r"_(\d{9,13})$")  # 9–13 digits covers seconds & ms

    results: list[tuple[str, int]] = []

    for entry in glob.glob(search_path):
        if not os.path.isdir(entry):
            continue

        name = os.path.basename(entry)
        match = timestamp_pattern.search(name)
        if match:
            timestamp = int(match.group(1))
            results.append((entry, timestamp))

    # Sort ascending by timestamp (latest last)
    results.sort(key=lambda x: x[1])
    return results


def get_latest_dir(search_path: str) -> str | None:
    """
    Return the directory with the most recent timestamp suffix.

    Args:
        search_path: Root directory to search in (supports glob patterns).

    Returns:
        Path to the latest directory, or None if no matches found.
    """
    dirs = find_dirs_with_timestamp(search_path)
    if not dirs:
        return None
    return dirs[-1][0]  # Last item has the highest (latest) timestamp


DATASET_TO_METRIC_MAPPING = {
    "mnli": {
        "metrics": [macro_f1, em],
        "labels": ["entailment", "neutral", "contradiction"],
    },
    "qqp": {"metrics": [f1, em], "labels": ["not_duplicate", "duplicate"]},
    "qnli": {"metrics": [f1, em], "labels": ["entailment", "not_entailment"]},
    "sst2": {"metrics": [f1, em], "labels": ["negative", "positive"]},
    "stsb": {"metrics": [pearsonr, spearmanr], "labels": []},
    "mrpc": {"metrics": [f1, em], "labels": ["not_equivalent", "equivalent"]},
    "rte": {"metrics": [f1, em], "labels": ["entailment", "not_entailment"]},
    "cola": {"metrics": [f1, em], "labels": ["unacceptable", "acceptable"]},
    "record": {"metrics": [record], "labels": []},
    "multirc": {"metrics": [f1, em], "labels": ["false", "true"]},
    "boolq": {"metrics": [f1, em], "labels": ["false", "true"]},
    "wic": {"metrics": [f1, em], "labels": ["false", "true"]},
    "wsc": {"metrics": [f1, em], "labels": ["false", "true"]},
    "cb": {
        "metrics": [macro_f1, em],
        "labels": ["entailment", "contradiction", "neutral"],
    },
    "copa": {"metrics": [f1, em], "labels": ["choice1", "choice2"]},
    "mmlu": {"metrics": [macro_f1, em], "labels": ["A", "B", "C", "D"]},
    "piqa": {"metrics": [f1, em], "labels": ["solution1", "solution2"]},
    "siqa": {"metrics": [macro_f1, em], "labels": ["A", "B", "C"]},
    "hellaswag": {
        "metrics": [macro_f1, em],
        "labels": ["ending1", "ending2", "ending3", "ending4"],
    },
    "winogrande": {"metrics": [f1, em], "labels": ["option1", "option2"]},
    "openbookqa": {"metrics": [macro_f1, em], "labels": ["A", "B", "C", "D"]},
    "math_qa": {"metrics": [macro_f1, em], "labels": ["a", "b", "c", "d", "e"]},
    "gsm8k": {"metrics": [gsm8k], "labels": []},
    "svamp": {"metrics": [svamp], "labels": []},
    "conala": {"metrics": [codebleu_metric], "labels": []},
    "codealpacapy": {"metrics": [codebleu_metric], "labels": []},
    "apps": {"metrics": [codebleu_metric], "labels": []},
    "squad_v2": {"metrics": [squad_v2_metric], "labels": []},
    "snli": {"metrics": [macro_f1, em], "labels": ["entailment", "neutral", "contradiction"]},
    "anli_r1": {"metrics": [macro_f1, em], "labels": ["entailment", "neutral", "contradiction"]},
    "paws": {"metrics": [f1, em], "labels": ["false", "true"]},
    "imdb": {"metrics": [f1, em], "labels": ["negative", "positive"]},
    "scitail": {"metrics": [f1, em], "labels": ["neutral", "entailment"]},
    "cr": {"metrics": [f1, em], "labels": ["negative", "positive"]},
    "rotten_tomatoes": {"metrics": [f1, em], "labels": ["negative", "positive"]}
}


datasets = ["mnli", "qnli", "qqp", "sst2", "record", "mrpc", "boolq", "rte", "cola", "snli", "anli_r1", "paws", "imdb", "squad_v2", "hellaswag", "winogrande", "cb", "scitail", "stsb", "cr", "rotten_tomatoes", "multirc", "copa", "piqa"]
models = ["llama-3.2-1b-instruct"]
peft_methods = ["lora", "base", "zero-shot"]

argparse_parser = argparse.ArgumentParser(
    prog="Compute metrics.",
    description="Compute metrics for single model.",
)

argparse_parser.add_argument("eval_dir", help="Directory created during evaluation.")
argparse_parser.add_argument(
    "--gather-results",
    action="store_true",
    help="Whether to gather results into a single table.",
)

args = argparse_parser.parse_args()


if args.gather_results:
    all_results = []
    for dataset in datasets:
        for m in models:
            for pm in peft_methods:
                eval_dir = get_latest_dir(f"{args.eval_dir}/{pm}/{m}/eval_{dataset}*")
                if eval_dir is None:
                    print(
                        f"No evaluation directory found for {dataset} {m} {pm}. Skipping."
                    )
                    continue

                with open(f"{eval_dir}/compute_metrics.jsonl") as json_file:
                    for line in json_file:
                        result = json.loads(line)
                        all_results.append(
                            {
                                "dataset": dataset,
                                "model": m,
                                "peft_method": pm,
                                **result,
                            }
                        )

    print(all_results)

    df = pd.DataFrame(all_results)
    id_cols = ["dataset", "model", "peft_method"]
    metric_cols = [c for c in df.columns if c not in id_cols]

    df_merged = (
        df.groupby(id_cols, as_index=False)[metric_cols]
        .max()
        .drop(columns=["macro_f1", "f1"])
    )
    print(df_merged)

    df_merged.to_csv("results.csv", index=False)

    # Collapse the per-task metrics into a single performance column: exact match
    # for classification tasks, falling back to Pearson correlation for regression
    # tasks (e.g. STS-B). Spearman is intentionally dropped.
    df_merged["performance"] = df_merged["exact_match"].fillna(df_merged["pearsonr"])

    # squad_v2's exact match is reported on a 0-100 scale; bring it in line with
    # the other tasks (0-1) so the table is comparable.
    df_merged.loc[df_merged["dataset"] == "squad_v2", "performance"] /= 100

    # Pivot into a datasets x PEFT-methods table (PEFT methods as rows).
    pivot = df_merged.pivot_table(
        index="dataset", columns="peft_method", values="performance"
    )
    ordered_methods = [pm for pm in peft_methods if pm in pivot.columns]
    pivot = pivot[ordered_methods]

    # Pretty display names for the LaTeX tables — shared with the figure scripts
    # via scripts/combo_names.py so an axis label and a table header cannot drift.
    from combo_names import DISPLAY_NAMES as DATASET_NAMES
    METHOD_NAMES = {"lora": "LoRA", "base": "Base", "zero-shot": "Zero-shot"}

    # Source vs target tasks, highlighted via cell background colour (xcolor).
    SOURCE_TASKS = {
        "qnli", "mnli", "snli", "anli_r1", "qqp", "paws", "sst2", "imdb",
        "record", "squad_v2", "hellaswag", "winogrande",
    }
    SOURCE_COLOR = "blue!15"
    TARGET_COLOR = "orange!20"

    def header_cell(dataset):
        color = SOURCE_COLOR if dataset in SOURCE_TASKS else TARGET_COLOR
        return f"\\cellcolor{{{color}}}{DATASET_NAMES.get(dataset, dataset)}"

    # Each table groups datasets by problem type. A table can hold several
    # groups, each rendered as a parent column spanning its datasets. cola
    # (linguistic acceptability) sits with the sentiment single-sentence tasks.
    table_specs = [
        (
            "Entailment / NLI",
            "nli",
            [
                (
                    "Entailment / NLI",
                    ["mnli", "qnli", "rte", "snli", "anli_r1", "cb", "scitail"],
                ),
            ],
        ),
        (
            "Paraphrase, sentiment and linguistic acceptability",
            "para_sent",
            [
                ("Paraphrase / semantic eq.", ["qqp", "mrpc", "paws", "stsb"]),
                (
                    "Sentiment \\& ling. acceptability",
                    ["sst2", "imdb", "cr", "rotten_tomatoes", "cola"],
                ),
            ],
        ),
        (
            "Question answering and commonsense reasoning",
            "qa_cs",
            [
                ("Question answering", ["boolq", "record", "squad_v2", "multirc"]),
                (
                    "Commonsense reasoning",
                    ["hellaswag", "winogrande", "copa", "piqa"],
                ),
            ],
        ),
    ]

    def make_table(title, label, groups):
        # Keep only datasets that are actually present, ordering source tasks
        # before target tasks within each group (stable within each subset).
        groups = [
            (
                g,
                sorted(
                    (d for d in ds if d in pivot.index),
                    key=lambda d: d not in SOURCE_TASKS,
                ),
            )
            for g, ds in groups
        ]
        groups = [(g, ds) for g, ds in groups if ds]
        if not groups:
            return ""

        flat_cols = [d for _, ds in groups for d in ds]
        sub = pivot.reindex(index=flat_cols).T

        # Format each cell, bolding the best (highest) method per dataset column.
        formatted = sub.copy().astype(object)
        for c in sub.columns:
            best = sub[c].max()
            formatted[c] = [
                "-"
                if pd.isna(v)
                else (f"\\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}")
                for v in sub[c]
            ]

        # Two-level column header: parent = problem group, child = dataset.
        formatted.index = [METHOD_NAMES.get(m, m) for m in sub.index]
        formatted.columns = pd.MultiIndex.from_tuples(
            [(g, header_cell(d)) for g, ds in groups for d in ds]
        )
        formatted.index.name = None

        tabular = formatted.to_latex(
            column_format="l" + "c" * formatted.shape[1],
            escape=False,
            multicolumn=True,
            multicolumn_format="c",
        )

        # Add booktabs \cmidrule under each parent-column header.
        rules, start = [], 2  # column 1 is the method-name label
        for _, ds in groups:
            rules.append(f"\\cmidrule(lr){{{start}-{start + len(ds) - 1}}}")
            start += len(ds)
        lines = tabular.splitlines()
        header_end = next(i for i, ln in enumerate(lines) if ln.strip().endswith("\\\\"))
        lines.insert(header_end + 1, "".join(rules))
        tabular = "\n".join(lines) + "\n"

        note = "Values are exact match."
        if any("stsb" in ds for _, ds in groups):
            note = (
                "Values are exact match, except for STS-B which reports "
                "Pearson correlation."
            )
        note += (
            f" Source tasks are shaded (\\colorbox{{{SOURCE_COLOR}}}{{blue}}), "
            f"target tasks (\\colorbox{{{TARGET_COLOR}}}{{orange}})."
        )

        return (
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            f"{tabular}"
            "}\n"
            f"\\caption{{{title}. {note}}}\n"
            f"\\label{{tab:results_{label}}}\n"
            "\\end{table*}\n"
        )

    tables = [make_table(title, label, groups) for title, label, groups in table_specs]
    latex = "\n".join(t for t in tables if t)

    with open("results.tex", "w") as tex_file:
        tex_file.write(latex)
    print(latex)

    exit(0)


for dataset in datasets:
    if dataset not in DATASET_TO_METRIC_MAPPING:
        raise ValueError(f"Dataset {dataset} not supported for metric computation.")

    for m in models:
        for pm in peft_methods:
            print(f"Computing metrics for {dataset} {m} {pm}...")
            eval_dir = get_latest_dir(f"{args.eval_dir}/{pm}/{m}/eval_{dataset}*")
            if eval_dir is None:
                print(
                    f"No evaluation directory found for {dataset} {m} {pm}. Skipping."
                )
                continue

            eval_samples = []
            with open(f"{eval_dir}/generated_predictions.jsonl") as json_file:
                for line in json_file:
                    eval_samples.append(json.loads(line))

            labels, predictions = [], []
            for es in eval_samples:
                labels.append(es["label"].split("</think>\n\n")[-1].strip().lower())
                predictions.append(
                    es["predict"].split("</think>\n\n")[-1].strip().lower()
                )

            with open(f"{eval_dir}/compute_metrics.jsonl", "w") as outfile:
                for metric in DATASET_TO_METRIC_MAPPING[dataset]["metrics"]:
                    if dataset in ["record", "squad_v2"]:
                        result = metric(predictions)
                    else:
                        result = metric(
                            predictions,
                            labels,
                            DATASET_TO_METRIC_MAPPING[dataset]["labels"],
                        )

                    # print(result)
                    json.dump(result, outfile)
                    outfile.write("\n")
