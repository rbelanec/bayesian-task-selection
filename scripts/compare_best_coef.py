"""Emit a LaTeX table comparing the best merge coefficient across N models.

The best coefficient is not stored as its own field; it is recovered from each
combo's ``*_acc_coef.csv`` as the coefficient maximizing mean accuracy across
tasks (``df.mean(axis=1).idxmax()``) -- the same shared-coef selection that
eval.py made during the sweep and that scripts/summarize_merged.py records. The
accuracy at that coef is the corresponding mean over the combo's tasks
(``df.mean(axis=1).max()``), i.e. summarize_merged's ``mean_acc``.

For every (method, combo) present in the models it prints a booktabs table with a
two-column group per model (best coef + accuracy at that coef) plus a trailing
difference group: $\\Delta$ (second minus first) when exactly two models are
given, otherwise the spread (max minus min across models). Writes the same table
to --out.

With --plot it additionally renders a grouped horizontal barplot — best coef
(left panel) and accuracy at that coef (right panel) per combo, one bar per
(model, method) so base vs lora sit adjacent within each combo — written as
png+pdf next to --plot-out. Plotting needs matplotlib, i.e. the `pf` conda env
python.

Usage:
    # two models
    PYTHONPATH=src python scripts/compare_best_coef.py \
        --models llama-3.2-1b-instruct llama-3.2-3b-instruct \
        --labels 1B 3B --out figures/merged/best_coef_compare.tex --plot

    # any number of models
    PYTHONPATH=src python scripts/compare_best_coef.py \
        --models model_a model_b model_c model_d
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import OrderedDict
from statistics import mean

import pandas as pd

MERGED_ROOT = "saves_bts_merged"

# Categorical palette (light mode) from the validated reference instance; colour
# encodes the *model*, assigned in --models order and never cycled past 8.
MODEL_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]


def mean_curves(model: str, seed: int, methods: list[str] | None):
    """{(method, combo): mean-accuracy-across-tasks Series indexed by coef}."""
    out: dict[tuple[str, str], pd.Series] = {}
    for csvf in glob.glob(f"{MERGED_ROOT}/*/{model}/*_{seed}_best/*_{seed}_acc_coef.csv"):
        method = csvf.split(os.sep)[1]
        if methods and method not in methods:
            continue
        df = pd.read_csv(csvf, index_col="scaling_coef").sort_index()
        out[(method, "+".join(df.columns))] = df.mean(axis=1)
    return out


def best_coef_acc(curves: dict[tuple[str, str], pd.Series]):
    """{(method, combo): (best_coef, mean_acc_at_best_coef)} from the curves."""
    return {k: (float(s.idxmax()), float(s.max())) for k, s in curves.items()}


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def _diff_num(values: list[float | None], pairwise: bool) -> float | None:
    """Δ (v1 - v0) for two models, else spread (max - min) over present values."""
    present = [v for v in values if v is not None]
    if pairwise:
        if values[0] is None or values[1] is None:
            return None
        return values[1] - values[0]
    if len(present) < 2:
        return None
    return max(present) - min(present)


def _fmt(v: float | None, prec: int, signed: bool) -> str:
    if v is None:
        return "--"
    return f"{v:{'+' if signed else ''}.{prec}f}"


def _avg(xs: list[float | None]) -> float | None:
    present = [x for x in xs if x is not None]
    return mean(present) if present else None


def _row_cells(values, pairwise):
    """The 2*(nmodels+1) numeric cells for one combo: per-model (coef, acc) + Δ/spread."""
    coefs = [None if v is None else v[0] for v in values]
    accs = [None if v is None else v[1] for v in values]
    cells = []
    for c, a in zip(coefs, accs):
        cells += [_fmt(c, 2, False), _fmt(a, 3, False)]
    cells += [_fmt(_diff_num(coefs, pairwise), 2, pairwise),
              _fmt(_diff_num(accs, pairwise), 3, pairwise)]
    return cells, coefs, accs


def build_table(rows, labels, pairwise, caption, label):
    nmodels = len(labels)
    diff_head = r"$\Delta$" if pairwise else "spread"
    # ll | (rr per model) | rr  (coef, acc within each group)
    col_spec = "ll" + "rr" * nmodels + "rr"

    groups = [r"\multicolumn{2}{c}{" + tex_escape(l) + "}" for l in labels]
    groups.append(r"\multicolumn{2}{c}{" + diff_head + "}")
    header1 = "    & & " + " & ".join(groups) + r" \\"
    # \cmidrule under each two-column group (leading cols are 1-2)
    cmids = " ".join(
        f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(nmodels + 1)
    )
    subcols = ["coef", "acc"] * (nmodels + 1)
    header2 = "    Method & Combination & " + " & ".join(subcols) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        header1,
        "    " + cmids,
        header2,
        r"    \midrule",
    ]

    by_method: "OrderedDict[str, list]" = OrderedDict()
    for (method, combo), values in rows:
        by_method.setdefault(method, []).append((combo, values))

    for gi, (method, grp) in enumerate(by_method.items()):
        if gi > 0:
            lines.append(r"    \midrule")
        # per-combo accumulators for the method average (averages of the deltas too)
        per_model_coef = [[] for _ in labels]
        per_model_acc = [[] for _ in labels]
        dcoef, dacc = [], []
        for combo, values in grp:
            cells, coefs, accs = _row_cells(values, pairwise)
            for i in range(nmodels):
                per_model_coef[i].append(coefs[i])
                per_model_acc[i].append(accs[i])
            dcoef.append(_diff_num(coefs, pairwise))
            dacc.append(_diff_num(accs, pairwise))
            lines.append(f"    {tex_escape(method)} & {tex_escape(combo)} & "
                         + " & ".join(cells) + r" \\")

        avg_cells = []
        for i in range(nmodels):
            avg_cells += [_fmt(_avg(per_model_coef[i]), 2, False),
                          _fmt(_avg(per_model_acc[i]), 3, False)]
        avg_cells += [_fmt(_avg(dcoef), 2, pairwise), _fmt(_avg(dacc), 3, pairwise)]
        lines.append(r"    \addlinespace")
        lines.append(r"    & \textit{average} & "
                     + " & ".join(rf"\textit{{{c}}}" for c in avg_cells) + r" \\")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def plot_compare(keys, per_model, labels, seed, out_base):
    """Grouped horizontal barplot of best coef (left) and its accuracy (right).

    One bar group per combo; within a group one bar per (model, method) series,
    model-major so a model's base/lora bars sit adjacent and the method gap is
    read within neighbouring bars. Writes ``<out_base>.png`` and ``.pdf``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    methods = sorted({m for m, _ in keys})
    combos = sorted({c for _, c in keys}, key=lambda c: (len(c.split("+")), c))
    series = [(i, m) for i in range(len(labels)) for m in methods]
    if len(series) > len(MODEL_COLORS):
        raise SystemExit(f"--plot supports at most {len(MODEL_COLORS)} model x method bars")

    nser = len(series)
    bar_h = 0.8 / nser
    fig_h = len(combos) * (0.11 * nser + 0.16) + 1.4
    fig, (ax_coef, ax_acc) = plt.subplots(1, 2, figsize=(10.0, fig_h), sharey=True)

    for si, (mi, method) in enumerate(series):
        color = MODEL_COLORS[si]
        # bars of one series across all combos; group centre = combo index
        ys, coefs, accs = [], [], []
        for ci, combo in enumerate(combos):
            v = per_model[mi].get((method, combo))
            if v is None:
                continue
            ys.append(ci - 0.4 + (si + 0.5) * bar_h)
            coefs.append(v[0])
            accs.append(v[1])
        label = f"{labels[mi]} {method}"
        ax_coef.barh(ys, coefs, height=bar_h * 0.92, color=color, label=label)
        ax_acc.barh(ys, accs, height=bar_h * 0.92, color=color)

    for ax, xlabel in ((ax_coef, "best scaling coef"), (ax_acc, "accuracy at best coef")):
        ax.set_yticks(range(len(combos)))
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
    ax_coef.set_yticklabels(combos)
    ax_coef.invert_yaxis()  # first combo on top, matching the table order
    ax_coef.set_ylim(len(combos) - 0.5, -0.5)
    ax_acc.set_xlim(0.0, 1.0)

    handles = [Patch(color=MODEL_COLORS[si], label=f"{labels[mi]} {method}")
               for si, (mi, method) in enumerate(series)]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, nser),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"best merge coefficient and its accuracy (seed {seed})",
                 fontsize=12, y=1 - 0.12 / fig_h)
    fig.tight_layout(rect=(0, 0.55 / fig_h, 1, 1 - 0.45 / fig_h))

    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=150)
    plt.close(fig)
    print(f"% wrote {out_base}.{{png,pdf}} ({len(combos)} combos x {nser} series)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="two or more model names")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="column headers (default: model names); must match --models length")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="*", default=None, help="restrict to these methods")
    ap.add_argument("--union", action="store_true",
                    help="include combos missing from some models (shown as --)")
    ap.add_argument("--caption", default=None)
    ap.add_argument("--label", default="tab:best-coef-compare")
    ap.add_argument("--out", default="figures/merged/best_coef_compare.tex")
    ap.add_argument("--plot", action="store_true",
                    help="also render per-combo accuracy-vs-coef curves per model")
    ap.add_argument("--plot-out", default=None,
                    help="plot path without extension (default: --out with .tex stripped)")
    args = ap.parse_args()

    if len(args.models) < 2:
        ap.error("--models needs at least two models")
    labels = args.labels or args.models
    if len(labels) != len(args.models):
        ap.error("--labels must have the same length as --models")

    per_model_curves = [mean_curves(m, args.seed, args.methods) for m in args.models]
    per_model = [best_coef_acc(c) for c in per_model_curves]
    if not any(per_model):
        raise SystemExit(f"no sweep CSVs found for any model (seed {args.seed})")

    key_sets = [set(d) for d in per_model]
    keys = set().union(*key_sets) if args.union else set.intersection(*key_sets)
    if not keys:
        raise SystemExit("no (method, combo) shared by all models (use --union to list all)")
    keys = sorted(keys, key=lambda k: (k[0], len(k[1].split("+")), k[1]))
    rows = [(k, [d.get(k) for d in per_model]) for k in keys]

    pairwise = len(args.models) == 2
    diff_desc = ("$\\Delta$ is the second minus the first model"
                 if pairwise else "spread is max minus min across models")
    caption = args.caption or (
        "Best merge coefficient (mean-accuracy optimum) and accuracy at that coef "
        f"per task combination across {', '.join(tex_escape(l) for l in labels)}; "
        f"{diff_desc}."
    )
    table = build_table(rows, labels, pairwise, caption, args.label)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(table)
    print(table)
    print(f"% wrote {args.out} ({len(rows)} rows)")

    if args.plot:
        plot_base = args.plot_out or (args.out[:-4] if args.out.endswith(".tex") else args.out)
        plot_compare(keys, per_model, labels, args.seed, plot_base)


if __name__ == "__main__":
    main()
