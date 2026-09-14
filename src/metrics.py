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

"""Task-specific compute_metrics plug-ins for the merged-model eval loop.

These classes share the contract of `llamafactory.train.sft.metric.ComputeClassification`:
a callable that the `CustomSeq2SeqTrainer` invokes with an `EvalPrediction` per
(possibly batched) chunk and a `compute_result` flag. They buffer decoded
predictions across batches and return a metric dict on the final call. The dict
must include an `accuracy` key, because `predict_accuracy()` in `eval.py` reads
`predict_results.metrics["predict_accuracy"]` to drive the coefficient sweep.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import evaluate
import numpy as np
from datasets import load_dataset

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import numpify

if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


# Module-level caches so repeated calls (multiple ComputeRecord instances during
# a sweep, or many test runs) don't re-pay the load_dataset / evaluate.load cost.
_RECORD_DATASET = None
_RECORD_METRIC = None


def _get_record_dataset_and_metric():
    global _RECORD_DATASET, _RECORD_METRIC
    if _RECORD_DATASET is None:
        _RECORD_DATASET = load_dataset("kinit/peft-factory", "record", split="validation")
    if _RECORD_METRIC is None:
        _RECORD_METRIC = evaluate.load("super_glue", "record")
    return _RECORD_DATASET, _RECORD_METRIC


def compute_record(preds: list[str]) -> dict[str, float]:
    """SuperGLUE ReCoRD metric for a flat list of normalized predictions.

    `preds[i]` must align by enumerate index with `kinit/peft-factory record`
    validation row i — exactly the contract used by `scripts/compute_metrics.py`.
    Returns `{"accuracy": exact_match, "f1": f1}`; `accuracy` is the headline
    `predict_accuracy()` picks up in the coef sweep.
    """
    dataset, metric = _get_record_dataset_and_metric()
    n = min(len(preds), len(dataset))
    predictions = [
        {"idx": dataset[i]["idx"], "prediction_text": preds[i]} for i in range(n)
    ]
    references = [
        {"idx": dataset[i]["idx"], "answers": dataset[i]["answers"]} for i in range(n)
    ]
    scores = metric.compute(predictions=predictions, references=references)
    em = float(scores.get("exact_match", 0.0))
    f1 = float(scores.get("f1", 0.0))
    return {"accuracy": em, "f1": f1}


@dataclass
class ComputeRecord:
    """SuperGLUE ReCoRD metric, plugged into the trainer's compute_metrics slot.

    The default `ComputeClassification` does per-row exact-match accuracy, which
    is wrong for ReCoRD: the `kinit/peft-factory` record validation set
    duplicates each (passage, query) once per gold answer, so a prediction
    matching *any* of those gold answers still counts as wrong on the rows
    whose specific gold answer it didn't match. The SuperGLUE record metric
    instead groups by `idx` and scores against the answer *set*. This mirrors
    `record()` in `scripts/compute_metrics.py`, but accumulated batch-by-batch
    inside the trainer so the value lands in
    `predict_results.metrics["predict_accuracy"]` — the same key
    `predict_accuracy()` already reads — and feeds the coefficient sweep
    directly.

    Exposes `accuracy = exact_match` (the grouped-by-`idx` ReCoRD EM, via
    `compute_record`) so the existing reader in `predict_accuracy()` and the
    per-task column in `*_acc_coef.csv` keep working. Also surfaces raw `f1`.

    Alignment caveat (same as the offline script): predictions are zipped to
    the validation set by enumerate index, which holds as long as the trainer's
    eval loop preserves dataset order. With `predict_with_generate=True`,
    `compute_classification_metrics: true`, and no eval shuffling that's true;
    a reordering sampler or out-of-order distributed gather would silently
    misalign.
    """

    tokenizer: "PreTrainedTokenizer"

    def __post_init__(self):
        self._dump()
        # Warm the module-level caches so the first `_dump()` doesn't pay the
        # load_dataset / evaluate.load cost on the first coefficient.
        _get_record_dataset_and_metric()

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "preds_buf") and len(self.preds_buf) > 0:
            result = compute_record(self.preds_buf)
        self.preds_buf = []
        return result

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds = numpify(eval_preds.predictions)
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        decoded = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Match the post-hoc normalization in scripts/compute_metrics.py so the
        # coef-sweep number agrees with the offline number.
        for d in decoded:
            self.preds_buf.append(d.split("</think>\n\n")[-1].strip().lower())

        if compute_result:
            return self._dump()


_SQUAD_DATASET = None
_SQUAD_METRIC = None


def _get_squad_dataset_and_metric():
    global _SQUAD_DATASET, _SQUAD_METRIC
    if _SQUAD_DATASET is None:
        _SQUAD_DATASET = load_dataset("kinit/peft-factory", "squad_v2", split="validation")
    if _SQUAD_METRIC is None:
        _SQUAD_METRIC = evaluate.load("squad_v2")
    return _SQUAD_DATASET, _SQUAD_METRIC


def compute_squad_v2(preds: list[str]) -> dict[str, float]:
    """SQuAD v2 EM/F1 for a flat list of normalized predictions.

    `preds[i]` must align by enumerate index with `kinit/peft-factory squad_v2`
    validation row i — the same contract as `compute_record` and as
    `squad_v2_metric()` in scripts/compute_metrics.py, whose prediction handling
    (first line only, "unanswerable" mapped to the no-answer slot) is mirrored
    here.

    **Rescaled to 0-1**, unlike the offline script: `evaluate`'s squad_v2 reports
    exact/f1 as percentages, so leaving it at 0-100 would put one column of the
    coefficient sweep two orders of magnitude above the rest — and `run_eval`
    picks the best coefficient by the *mean* over the combo's tasks, so the
    choice would collapse to "whatever is best for squad_v2". compute_metrics.py
    applies the same /100 when it builds the results table.
    """
    dataset, metric = _get_squad_dataset_and_metric()
    n = min(len(preds), len(dataset))

    predictions, references = [], []
    for i in range(n):
        ex = dataset[i]
        # Keep only the first line; instruction-tuned models like to elaborate.
        pred = preds[i].strip().split("\n")[0].strip()
        is_no_answer = pred.lower().strip(' ."') == "unanswerable"
        predictions.append({
            "id": ex["id"],
            "prediction_text": "" if is_no_answer else pred,
            "no_answer_probability": 1.0 if is_no_answer else 0.0,
        })
        references.append({"id": ex["id"], "answers": ex["answers"]})

    scores = metric.compute(predictions=predictions, references=references)
    return {"accuracy": float(scores["exact"]) / 100, "f1": float(scores["f1"]) / 100}


@dataclass
class ComputeSquadV2:
    """SQuAD v2 metric, plugged into the trainer's compute_metrics slot.

    The default `ComputeClassification` scores an extractive-QA answer by exact
    string equality against the gold string, which ignores SQuAD's answer
    normalization (articles, punctuation, casing), gives no credit for a partial
    span, and has no notion of the unanswerable class that half of SQuAD v2 is
    about. The official metric handles all three.

    Like `ComputeRecord`, predictions are buffered across batches and scored
    once at the end against the validation split, and `accuracy` carries the
    headline number `predict_accuracy()` reads. Note `accuracy` here is EM on a
    **0-1** scale — see `compute_squad_v2` — so it is directly comparable with
    the other task columns in `*_acc_coef.csv`, while the `exact_match` stored
    in saves_bts_preliminary by scripts/compute_metrics.py stays on 0-100 and
    needs the /100 that utils._eval_exact_match applies when read back.

    Alignment caveat is ReCoRD's: predictions are zipped to the validation set
    by enumerate index, which holds while the eval loop preserves dataset order.
    """

    tokenizer: "PreTrainedTokenizer"

    def __post_init__(self):
        self._dump()
        # Warm the caches so the first coefficient doesn't pay the load cost.
        _get_squad_dataset_and_metric()

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "preds_buf") and len(self.preds_buf) > 0:
            result = compute_squad_v2(self.preds_buf)
        self.preds_buf = []
        return result

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds = numpify(eval_preds.predictions)
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        decoded = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Match the post-hoc normalization in scripts/compute_metrics.py so the
        # coef-sweep number agrees with the offline number.
        for d in decoded:
            self.preds_buf.append(d.split("</think>\n\n")[-1].strip().lower())

        if compute_result:
            return self._dump()


_PEARSON_METRIC = None
_SPEARMAN_METRIC = None


def _get_correlation_metrics():
    global _PEARSON_METRIC, _SPEARMAN_METRIC
    if _PEARSON_METRIC is None:
        _PEARSON_METRIC = evaluate.load("pearsonr")
    if _SPEARMAN_METRIC is None:
        _SPEARMAN_METRIC = evaluate.load("spearmanr")
    return _PEARSON_METRIC, _SPEARMAN_METRIC


def _to_float(string, default=-1.0):
    """`string_to_float` from scripts/compute_metrics.py."""
    try:
        return float(string)
    except ValueError:
        return default


def compute_correlation(preds: list[str], labels: list[str]) -> dict[str, float]:
    """Pearson/Spearman over decoded regression outputs.

    Mirrors `pearsonr()` / `spearmanr()` in scripts/compute_metrics.py, down to
    the -1.0 stand-in for a prediction that is not a number at all, so the
    coef-sweep value agrees with the offline table. Returns `{"accuracy":
    pearson, ...}`; `accuracy` is the headline `predict_accuracy()` picks up.

    A degenerate input (fewer than two points, or zero variance on either side —
    which happens at the coefficients where the merged model emits one constant
    string) leaves correlation undefined. It reports 0.0 rather than NaN: the
    sweep takes an `argmax` over the coefficient curve, and numpy's argmax would
    happily return the NaN's index.
    """
    pearson, spearman = _get_correlation_metrics()
    p = [_to_float(x) for x in preds]
    t = [_to_float(x) for x in labels]

    if len(p) < 2 or len(set(p)) < 2 or len(set(t)) < 2:
        return {"accuracy": 0.0, "pearsonr": 0.0, "spearmanr": 0.0}

    r = float(pearson.compute(predictions=p, references=t)["pearsonr"])
    rho = float(spearman.compute(predictions=p, references=t)["spearmanr"])
    r = r if np.isfinite(r) else 0.0
    rho = rho if np.isfinite(rho) else 0.0
    return {"accuracy": r, "pearsonr": r, "spearmanr": rho}


@dataclass
class ComputeCorrelation:
    """Correlation metric for regression targets, plugged into the trainer.

    The default `ComputeClassification` exact-matches the decoded prediction
    against the decoded gold string, which for a similarity score in [0, 5] is
    wrong twice over: it demands the exact formatting of a float, and it treats
    2.4-vs-2.5 the same as 2.4-vs-0.0. STS-B is scored by correlation instead —
    its entry in scripts/compute_metrics.py is `[pearsonr, spearmanr]` and it
    has no exact_match at all.

    Correlation cannot be averaged over batches the way accuracy can, so
    predictions and labels are buffered across calls and scored once at the end,
    like `ComputeRecord`, rather than accumulated as per-batch means in a
    `score_dict`.

    Exposes `accuracy = pearsonr` so the existing reader in `predict_accuracy()`
    and the per-task column in `*_acc_coef.csv` keep working unchanged; also
    surfaces `spearmanr`.
    """

    tokenizer: "PreTrainedTokenizer"

    def __post_init__(self):
        self._dump()
        # Warm the caches so the first coefficient doesn't pay evaluate.load.
        _get_correlation_metrics()

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "preds_buf") and len(self.preds_buf) > 0:
            result = compute_correlation(self.preds_buf, self.labels_buf)
        self.preds_buf, self.labels_buf = [], []
        return result

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        for buf, ids in ((self.preds_buf, preds), (self.labels_buf, labels)):
            for d in self.tokenizer.batch_decode(ids, skip_special_tokens=True):
                # Same normalization as scripts/compute_metrics.py.
                buf.append(d.split("</think>\n\n")[-1].strip().lower())

        if compute_result:
            return self._dump()


# Dispatch from a task name to the compute_metrics class to use. Keep entries
# here only for tasks that need something other than the llamafactory default
# (`ComputeClassification`) — eval.py falls back to that for any task not
# listed.
TASK_COMPUTE_METRICS = {
    "record": ComputeRecord,
    "squad_v2": ComputeSquadV2,
    "stsb": ComputeCorrelation,
}
