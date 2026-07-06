"""NAI vs SAR with both subspaces overlaid, one figure per method (Fig. 3a style).

Companion to scripts/visualize_nai_sar.py. Where that script colour-codes tasks
and shape-codes methods in a single figure, this one fixes the method (one figure
each for lora, base, ...) and instead overlays BOTH alignment metrics per task:

  *  marker at (SAR,      NAI)   -- alignment to the summed task-vector subspace
  ^  marker at (SAR_iso,  NAI)   -- alignment to the isotropic subspace

The two markers for a task share the same colour and are joined by a thin line, so
each task reads as a short segment showing how much the alignment ratio moves when
the reference subspace is isotropised. NAI is identical for both points (it comes
from the one merge we evaluated), so the segment is horizontal -- this view is about
the SAR shift, mirroring the TA-vs-Iso star/triangle pairs in the paper's Fig. 3a.

A dashed horizontal line marks NAI = 1 (merged matches the single-task fine-tune).

Usage:
    PYTHONPATH=src python scripts/visualize_nai_sar_compare.py \
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

# Stable colour per task (shared with the other viz scripts).
TASK_COLOR = {
    "mnli": "#1f77b4", "qnli": "#ff7f0e", "qqp": "#2ca02c",
    "sst2": "#d62728", "record": "#9467bd",
}
_FALLBACK_COLORS = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def task_color(task: str) -> str:
    if task not in TASK_COLOR:
        TASK_COLOR[task] = _FALLBACK_COLORS[len(TASK_COLOR) % len(_FALLBACK_COLORS)]
    return TASK_COLOR[task]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_rows(csv_path: str):
    """method -> combo (tuple) -> list of (task, sar, sar_iso, nai), all present."""
    out: dict[str, dict[tuple[str, ...], list[tuple]]] = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            sar, sar_iso, nai = _f(r.get("sar")), _f(r.get("sar_iso")), _f(r.get("nai"))
            if sar is None or sar_iso is None or nai is None:
                continue
            out[r["method"]][tuple(r["combo"].split("+"))].append(
                (r["task"], sar, sar_iso, nai)
            )
    return out


def plot_method(method: str, combos, ncols: int, out_dir: str):
    keys = sorted(combos, key=lambda k: (len(k), k))
    n = len(keys)
    ncols_eff = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols_eff)
    fig, axes = plt.subplots(nrows, ncols_eff, figsize=(4.0 * ncols_eff, 3.4 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    tasks_present: set[str] = set()

    for ax, key in zip(flat, keys):
        for task, sar, sar_iso, nai in combos[key]:
            tasks_present.add(task)
            c = task_color(task)
            ax.plot([sar, sar_iso], [nai, nai], color=c, lw=0.8, alpha=0.5, zorder=1)
            ax.scatter(sar, nai, color=c, marker="*", s=150, alpha=0.9,
                       edgecolors="black", linewidths=0.4, zorder=2)
            ax.scatter(sar_iso, nai, color=c, marker="^", s=80, alpha=0.9,
                       edgecolors="black", linewidths=0.4, zorder=2)
        ax.axhline(1.0, color="grey", linestyle="--", lw=0.8, alpha=0.6)
        ax.set_title("+".join(key), fontsize=9)
        ax.set_xlabel("SAR", fontsize=8)
        ax.set_ylabel("NAI", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    for ax in flat[n:]:
        ax.axis("off")

    handles = [Line2D([0], [0], marker="o", linestyle="None", color=task_color(t),
                      label=t, markersize=8) for t in sorted(tasks_present)]
    handles += [
        Line2D([0], [0], marker="*", linestyle="None", color="black",
               label="SAR (sum)", markersize=11),
        Line2D([0], [0], marker="^", linestyle="None", color="black",
               label="SAR (iso)", markersize=8),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(8, len(handles)),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"{method}: NAI vs SAR (sum vs iso subspace)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))

    os.makedirs(out_dir, exist_ok=True)
    stem = f"nai_vs_sar_compare_{method}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir}/{stem}.{{png,pdf}} ({n} combos)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="figures/nai_sar.csv")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="restrict to these methods (default: all found)")
    ap.add_argument("--ncols", type=int, default=4, help="subplot columns")
    ap.add_argument("--out-dir", default="figures/nai_sar")
    args = ap.parse_args()

    by_method = read_rows(args.csv)
    if args.methods:
        by_method = {m: c for m, c in by_method.items() if m in args.methods}
    if not by_method:
        raise SystemExit(
            f"no plottable rows in {args.csv} "
            "(need non-empty 'sar', 'sar_iso' and 'nai' — did SAR run?)"
        )

    for method in sorted(by_method):
        plot_method(method, by_method[method], args.ncols, args.out_dir)


if __name__ == "__main__":
    main()
