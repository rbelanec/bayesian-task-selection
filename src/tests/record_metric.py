"""Score a `generated_predictions.jsonl` from a record eval dir with the
SuperGLUE ReCoRD metric.

Run:
    PYTHONPATH=src python tests/record_metric.py \
        saves_bts_preliminary/lora/llama-3.2-1b-instruct/eval_record_42_1773159746

No arg → picks the latest `eval_record_*` dir under
`saves_bts_preliminary/lora/llama-3.2-1b-instruct` (lora baseline). Also prints
the naive per-row exact-match number (what `ComputeClassification` would have
produced) so the gap between "wrong metric" and "right metric" is visible at a
glance.
"""

import argparse
import glob
import json
import os

from metrics import compute_record


DEFAULT_GLOB = "saves_bts_preliminary/lora/llama-3.2-1b-instruct/eval_record_*"


def _latest_dir(pattern: str) -> str:
    # Mirrors scripts/compute_metrics.py:get_latest_dir — directory suffix is a
    # unix timestamp, biggest one wins.
    candidates = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not candidates:
        raise FileNotFoundError(f"No directories matched {pattern!r}")
    return max(candidates, key=lambda d: int(d.rsplit("_", 1)[-1]))


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _normalize(text: str) -> str:
    # Same normalization as scripts/compute_metrics.py and ComputeRecord — the
    # SuperGLUE metric handles punctuation/case internally but we still strip
    # any think prefix so a future reasoning model doesn't break the alignment.
    return text.split("</think>\n\n")[-1].strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a record eval dir with the SuperGLUE ReCoRD metric.")
    parser.add_argument(
        "eval_dir",
        nargs="?",
        help=f"Path to an eval_record_* dir. Defaults to the latest under {DEFAULT_GLOB!r}.",
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir or _latest_dir(DEFAULT_GLOB)
    preds_path = os.path.join(eval_dir, "generated_predictions.jsonl")
    samples = _load_jsonl(preds_path)
    print(f"Scoring {len(samples)} predictions from {preds_path}")

    preds = [_normalize(s["predict"]) for s in samples]
    labels = [_normalize(s["label"]) for s in samples]

    # The right metric — groups by idx, scores against the answer set.
    record_scores = compute_record(preds)

    # The wrong metric, just to make the gap visible — per-row exact match,
    # which penalizes the duplicated rows whose specific gold answer the model
    # didn't happen to emit.
    naive_em = sum(p == l for p, l in zip(preds, labels)) / max(len(preds), 1)

    print(f"  naive per-row exact match (ComputeClassification): {naive_em:.4f}")
    print(f"  SuperGLUE record exact_match (grouped by idx):     {record_scores['accuracy']:.4f}")
    print(f"  SuperGLUE record f1:                               {record_scores['f1']:.4f}")


if __name__ == "__main__":
    main()
