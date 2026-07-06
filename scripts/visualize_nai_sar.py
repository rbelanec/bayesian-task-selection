"""Scatter NAI vs SAR per task combination (Fig. 3 style, Marczak et al. 2502.04959).

Reads the CSV from scripts/compute_nai_sar.py and renders one scatter subplot per
task combination. Within a subplot every point is one (method, task): x = SAR,
y = NAI, colour = task, marker shape = method. So each subplot shows, for one
combo, how subspace alignment (x) relates to retained performance (y) and how the
merge method moves tasks through that space.

A dashed horizontal line marks NAI = 1 (merged matches the single-task fine-tune).

Usage:
    PYTHONPATH=src python scripts/visualize_nai_sar.py \
        --csv figures/nai_sar.csv --out-dir figures/nai_sar
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Stable colour per task (shared with scripts/visualize_merged.py).
TASK_COLOR = {
    "mnli": "#1f77b4", "qnli": "#ff7f0e", "qqp": "#2ca02c",
    "sst2": "#d62728", "record": "#9467bd",
}
_FALLBACK_COLORS = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Stable marker per method.
METHOD_MARKER = {"base": "o", "freeze": "s", "lora": "^"}
_FALLBACK_MARKERS = ["P", "X", "D", "v", "*"]


def task_color(task: str) -> str:
    if task not in TASK_COLOR:
        TASK_COLOR[task] = _FALLBACK_COLORS[len(TASK_COLOR) % len(_FALLBACK_COLORS)]
    return TASK_COLOR[task]


def method_marker(method: str) -> str:
    if method not in METHOD_MARKER:
        METHOD_MARKER[method] = _FALLBACK_MARKERS[len(METHOD_MARKER) % len(_FALLBACK_MARKERS)]
    return METHOD_MARKER[method]


def _f(x):
    """Parse a float; empty/invalid cells (e.g. --no-sar runs) become None."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_rows(csv_path: str, metric: str):
    """combo (tuple) -> list of (method, task, x=metric, y=nai) with both present."""
    combos: dict[tuple[str, ...], list[tuple]] = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            x, y = _f(r.get(metric)), _f(r.get("nai"))
            if x is None or y is None:
                continue
            combo = tuple(r["combo"].split("+"))
            combos[combo].append((r["method"], r["task"], x, y))
    return combos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="figures/nai_sar.csv")
    ap.add_argument("--metric", choices=["sar", "sar_iso"], default="sar",
                    help="which alignment metric on the x-axis")
    ap.add_argument("--ncols", type=int, default=4, help="subplot columns")
    ap.add_argument("--out-dir", default="figures/nai_sar")
    args = ap.parse_args()

    combos = read_rows(args.csv, args.metric)
    if not combos:
        raise SystemExit(
            f"no plottable rows in {args.csv} "
            f"(need non-empty '{args.metric}' and 'nai' — did SAR run?)"
        )

    keys = sorted(combos, key=lambda k: (len(k), k))
    n = len(keys)
    ncols = max(1, min(args.ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    tasks_present: set[str] = set()
    methods_present: set[str] = set()

    for ax, key in zip(flat, keys):
        for method, task, x, y in combos[key]:
            tasks_present.add(task)
            methods_present.add(method)
            ax.scatter(x, y, color=task_color(task), marker=method_marker(method),
                       s=90, alpha=0.9, edgecolors="black", linewidths=0.4)
        ax.axhline(1.0, color="grey", linestyle="--", lw=0.8, alpha=0.6)  # merged == FT
        ax.set_title("+".join(key), fontsize=9)
        ax.set_xlabel("SAR" + (" (iso)" if args.metric == "sar_iso" else ""), fontsize=8)
        ax.set_ylabel("NAI", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    for ax in flat[n:]:
        ax.axis("off")

    color_handles = [Line2D([0], [0], marker="o", linestyle="None", color=task_color(t),
                            label=t, markersize=8) for t in sorted(tasks_present)]
    shape_handles = [Line2D([0], [0], marker=method_marker(m), linestyle="None",
                            color="black", label=m, markersize=8)
                     for m in sorted(methods_present)]
    fig.legend(handles=color_handles + shape_handles, loc="lower center",
               ncol=min(8, len(color_handles) + len(shape_handles)),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"NAI vs {args.metric.upper()} per task combination", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"nai_vs_{args.metric}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir, f"{stem}.{ext}"), dpi=150)
    plt.close(fig)
    print(f"wrote {args.out_dir}/{stem}.{{png,pdf}} ({n} combos)")


if __name__ == "__main__":
    main()
