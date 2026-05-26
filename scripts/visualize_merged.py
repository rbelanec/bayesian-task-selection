"""Per-method figures of merged-model accuracy vs scaling coefficient, across task combos.

Reads every coefficient-sweep CSV under ``saves_bts_merged`` and renders **one figure per
method** (base / freeze / lora), each with one subplot per task combination. Within a
subplot every line is one task's accuracy as the merge scaling coefficient is swept;
colour encodes the *task*. Splitting by method keeps each figure readable (overlaying all
methods in a single figure was too cluttered).

Single-task performance (from ``saves_bts_preliminary/<method>/<model>/eval_<task>_<seed>_*/
predict_results.json``) is overlaid as a horizontal dotted line per task — the ceiling a
perfect merge would retain. A merged curve dropping far below its dotted line is a task
the merge sacrificed (e.g. qnli in base/freeze mnli+qnli).

Usage:
    PYTHONPATH=src python scripts/visualize_merged.py \
        --model llama-3.2-1b-instruct --seed 42 \
        --out-dir figures/merged
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MERGED_ROOT = "saves_bts_merged"
PRELIM_ROOT = "saves_bts_preliminary"

# Stable colour per task, so every subplot/figure reads the same way.
TASK_COLOR = {
    "mnli": "#1f77b4", "qnli": "#ff7f0e", "qqp": "#2ca02c",
    "sst2": "#d62728", "record": "#9467bd",
}
_FALLBACK_COLORS = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def task_color(task: str) -> str:
    if task not in TASK_COLOR:
        TASK_COLOR[task] = _FALLBACK_COLORS[len(TASK_COLOR) % len(_FALLBACK_COLORS)]
    return TASK_COLOR[task]


# ---------------------------------------------------------------------------- #
# data loading
# ---------------------------------------------------------------------------- #
def read_acc_coef(path: str):
    """Return (coefs, tasks, {task: accs}) from one *_acc_coef.csv."""
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        tasks = header[1:]
        coefs: list[float] = []
        cols: dict[str, list[float]] = {t: [] for t in tasks}
        for row in reader:
            if not row:
                continue
            coefs.append(float(row[0]))
            for t, v in zip(tasks, row[1:]):
                cols[t].append(float(v))
    return np.array(coefs), tasks, {t: np.array(v) for t, v in cols.items()}


def discover_by_method(model: str, seed: int):
    """method -> {combo (tuple of tasks): (coefs, {task: accs})}."""
    by_method: dict[str, dict[tuple[str, ...], tuple]] = defaultdict(dict)
    for method_dir in sorted(glob.glob(f"{MERGED_ROOT}/*/{model}")):
        method = method_dir.split(os.sep)[-2]
        for csvf in sorted(glob.glob(f"{method_dir}/*_{seed}_best/*_{seed}_acc_coef.csv")):
            coefs, tasks, accs = read_acc_coef(csvf)
            by_method[method][tuple(tasks)] = (coefs, accs)
    return by_method


_single_cache: dict[tuple[str, str], float | None] = {}


def single_task_acc(method: str, model: str, seed: int, task: str) -> float | None:
    """predict_accuracy of the standalone fine-tune for (method, task), or None."""
    key = (method, task)
    if key not in _single_cache:
        hits = sorted(glob.glob(
            f"{PRELIM_ROOT}/{method}/{model}/eval_{task}_{seed}_*/predict_results.json"))
        if hits:
            _single_cache[key] = json.load(open(hits[0])).get("predict_accuracy")
        else:
            _single_cache[key] = None
    return _single_cache[key]


# ---------------------------------------------------------------------------- #
# plotting
# ---------------------------------------------------------------------------- #
def plot_method(method: str, combos, model: str, seed: int, ncols: int, out_dir: str):
    """One figure for a single method: a grid of per-combo coefficient sweeps."""
    keys = sorted(combos, key=lambda k: (len(k), k))
    n = len(keys)
    ncols_eff = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols_eff)
    fig, axes = plt.subplots(nrows, ncols_eff, figsize=(4.0 * ncols_eff, 3.2 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    tasks_present: set[str] = set()

    for ax, key in zip(flat, keys):
        coefs, accs = combos[key]
        for task in key:
            tasks_present.add(task)
            ax.plot(coefs, accs[task], color=task_color(task), lw=1.6, alpha=0.9)
            st = single_task_acc(method, model, seed, task)  # single-task ceiling (dotted)
            if st is not None:
                ax.axhline(st, color=task_color(task), linestyle=":", lw=1.0, alpha=0.6)
        ax.set_title("+".join(key), fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("scaling coef", fontsize=8)
        ax.set_ylabel("accuracy", fontsize=8)

    for ax in flat[n:]:
        ax.axis("off")

    handles = [Line2D([0], [0], color=task_color(t), lw=2, label=t)
               for t in sorted(tasks_present)]
    handles.append(Line2D([0], [0], color="black", lw=1, linestyle=":", label="single-task acc"))
    fig.legend(handles=handles, loc="lower center", ncol=min(8, len(handles)),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"{method}: merged accuracy vs scaling coefficient — {model} (seed {seed})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"merged_coef_sweeps_{method}.{ext}"), dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir}/merged_coef_sweeps_{method}.{{png,pdf}} ({n} combos)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ncols", type=int, default=4, help="subplot columns per figure")
    ap.add_argument("--out-dir", default="figures/merged")
    args = ap.parse_args()

    by_method = discover_by_method(args.model, args.seed)
    if not by_method:
        raise SystemExit(f"no *_acc_coef.csv found under {MERGED_ROOT}/*/{args.model}")
    for method in sorted(by_method):
        plot_method(method, by_method[method], args.model, args.seed, args.ncols, args.out_dir)


if __name__ == "__main__":
    main()
