"""
Visualize merged task-vector evaluation results from saves_bts_merged.

Directory name format: eval_{task1}_{task2}..._on_{eval_task}_{seed}_{id}

Usage:
    python scripts/visualize_merged_results.py [saves_bts_merged_root] [results_csv]

Defaults:
    saves_bts_merged_root = saves_bts_merged
    results_csv           = results.csv  (used for single-task lora baseline)
"""

import sys
import re
import json
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
merged_root = sys.argv[1] if len(sys.argv) > 1 else "saves_bts_merged"
baseline_csv = sys.argv[2] if len(sys.argv) > 2 else "results.csv"

# ── collect results ────────────────────────────────────────────────────────────
PATTERN = re.compile(r"^eval_(.+)_on_(\w+)_(\d+)_(\d+)$")

records = []
for root, dirs, files in os.walk(merged_root):
    if "predict_results.json" not in files:
        continue
    dirname = os.path.basename(root)
    m = PATTERN.match(dirname)
    if not m:
        continue
    combo_str, eval_task, seed, run_id = m.groups()
    combo = tuple(sorted(combo_str.split("_")))
    combo_label = "+".join(combo)
    n_tasks = len(combo)

    with open(os.path.join(root, "predict_results.json")) as f:
        data = json.load(f)
    accuracy = data.get("predict_accuracy", 0.0)

    records.append(
        {
            "combo": combo,
            "combo_label": combo_label,
            "n_tasks": n_tasks,
            "eval_task": eval_task,
            "accuracy": accuracy,
            "seed": int(seed),
        }
    )

df = pd.DataFrame(records)

# drop runs where eval_task is not part of the combo (sanity / filter zero-accuracy failed runs)
df = df[df.apply(lambda r: r["eval_task"] in r["combo"], axis=1)]
df_valid = df.copy()

# ── baseline from results.csv ──────────────────────────────────────────────────
baseline = {}
if os.path.exists(baseline_csv):
    base_df = pd.read_csv(baseline_csv)
    lora_rows = base_df[base_df["peft_method"] == "lora"]
    baseline = dict(zip(lora_rows["dataset"], lora_rows["exact_match"]))

# ── Figure 1: grouped bar charts per eval dataset ──────────────────────────────
eval_tasks = sorted(df_valid["eval_task"].unique())
n_cols = 3
n_rows = math.ceil(len(eval_tasks) / n_cols)

fig1, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
axes = np.array(axes).flatten()

palette = {2: "#4c72b0", 3: "#dd8452", 4: "#55a868", 5: "#c44e52"}

for ax_idx, task in enumerate(eval_tasks):
    ax = axes[ax_idx]
    sub = df_valid[df_valid["eval_task"] == task].sort_values(
        ["n_tasks", "combo_label"]
    )

    xs = range(len(sub))
    bars = ax.bar(
        xs,
        sub["accuracy"],
        color=[palette.get(n, "#8172b3") for n in sub["n_tasks"]],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(sub["combo_label"], rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"eval: {task}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # single-task lora baseline
    if task in baseline:
        ax.axhline(
            baseline[task],
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"lora (single) = {baseline[task]:.3f}",
        )
        ax.legend(fontsize=7)

# legend for number of tasks
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=palette[n], label=f"{n} tasks") for n in sorted(palette)
]
fig1.legend(handles=legend_elements, loc="lower right", fontsize=9, title="Combo size")

# hide unused axes
for ax in axes[len(eval_tasks) :]:
    ax.set_visible(False)

fig1.suptitle(
    "Merged Task Vector Accuracy by Eval Dataset",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)
fig1.tight_layout()
fig1.savefig("merged_results_bars.png", dpi=150, bbox_inches="tight")
print("Saved merged_results_bars.png")

# ── Figure 2: heatmap — combo × eval_task ─────────────────────────────────────
pivot = df_valid.pivot_table(
    index="combo_label", columns="eval_task", values="accuracy", aggfunc="mean"
)
pivot = pivot.sort_index(key=lambda idx: idx.map(lambda s: (s.count("+"), s)))

fig2, ax2 = plt.subplots(
    figsize=(max(8, len(pivot.columns) * 1.4), max(6, len(pivot) * 0.45))
)
im = ax2.imshow(pivot.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)

ax2.set_xticks(range(len(pivot.columns)))
ax2.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=10)
ax2.set_yticks(range(len(pivot.index)))
ax2.set_yticklabels(pivot.index, fontsize=8)
ax2.set_title(
    "Merged Task Vector Accuracy Heatmap\n(rows = merged combo, cols = eval dataset)",
    fontsize=12,
)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax2.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if val > 0.5 else "dimgray",
            )

fig2.colorbar(im, ax=ax2, label="Accuracy", shrink=0.6)
fig2.tight_layout()
fig2.savefig("merged_results_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved merged_results_heatmap.png")

plt.show()

# ── Figure 3: bar plots per combo size, one PNG each ──────────────────────────
combo_sizes = sorted(df_valid["n_tasks"].unique())
all_eval_tasks = sorted(df_valid["eval_task"].unique())
colors = cm.tab10(np.linspace(0, 1, len(all_eval_tasks)))
task_color = dict(zip(all_eval_tasks, colors))

for n in combo_sizes:
    sub = df_valid[df_valid["n_tasks"] == n]
    combos = sorted(sub["combo_label"].unique())
    n_combos = len(combos)
    n_bars = len(all_eval_tasks)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(8, n_combos * (n_bars * 0.35 + 0.5)), 5))

    for i, task in enumerate(all_eval_tasks):
        task_vals = [
            sub[(sub["combo_label"] == c) & (sub["eval_task"] == task)][
                "accuracy"
            ].values
            for c in combos
        ]
        heights = [v[0] if len(v) > 0 else np.nan for v in task_vals]
        xs = np.arange(n_combos) + i * width
        ax.bar(
            xs,
            heights,
            width=width,
            color=task_color[task],
            alpha=0.85,
            label=task,
            edgecolor="white",
            linewidth=0.4,
        )

    # mean accuracy marker
    means = [sub[sub["combo_label"] == c]["accuracy"].mean() for c in combos]
    center_xs = np.arange(n_combos) + (n_bars - 1) * width / 2
    ax.plot(
        center_xs,
        means,
        marker="D",
        color="black",
        linewidth=1.2,
        markersize=6,
        zorder=5,
        label="mean",
    )

    ax.set_xticks(center_xs)
    ax.set_xticklabels(combos, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"{n}-task merged combos — accuracy per eval task",
        fontsize=12,
        fontweight="bold",
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(title="eval task", fontsize=8, loc="upper right")

    fig.tight_layout()
    fname = f"merged_results_scatter_{n}tasks.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
