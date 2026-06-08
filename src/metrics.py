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

    Exposes `accuracy = (em + f1) / 2` (the SuperGLUE ReCoRD summary score) so
    the existing reader in `predict_accuracy()` and the per-task column in
    `*_acc_coef.csv` keep working. Also surfaces raw `exact_match` / `f1`.

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


# Dispatch from a task name to the compute_metrics class to use. Keep entries
# here only for tasks that need something other than the llamafactory default
# (`ComputeClassification`) — eval.py falls back to that for any task not
# listed.
TASK_COMPUTE_METRICS = {
    "record": ComputeRecord,
}
