"""Rank source-task mixtures by their transfer to each TARGET task.

Reads every ``*_target/*_target_acc_coef.csv`` written by src/eval/eval_target.py
and, for each (method, target task), ranks the source combinations — the
ground-truth "which source-task set is best for this target" ordering that a
task-selection method should be compared against. Per (method, combo, target):

  best_coef / best_acc  accuracy at the coefficient that is best FOR THIS
                        TARGET (df[target].idxmax())
  nai                   Normalized Accuracy Improvement (Marczak et al.) of
                        that accuracy: (merged - zero_shot) / (finetuned -
                        zero_shot), with zero-shot from
                        saves_bts_preliminary/zero-shot and finetuned = the
                        target's own single-task FT — the same sources as
                        utils.compute_nai. NAI 1.0 means the mixture matches
                        fine-tuning on the target; 0.0 means no gain over the
                        pretrained model.
  sar / sar_iso         Subspace Alignment Ratio of the target's own task
                        vector against the source mixture's summed subspace,
                        joined from the CSV written by
                        scripts/compute_target_sar.py (--sar; skipped with a
                        note if that file does not exist).

Writes the full table to --out, prints the top --top combos per (method,
target) ranked by best_acc, and writes the same ranking as a booktabs LaTeX
table to --tex (with a per-target references row: zero-shot and single-task FT
accuracy). Also writes the full combinations matrix (n_combinations x
n_targets, per-target best accuracy, best combo per target in bold) as a pivot
CSV (--matrix) and LaTeX table (--matrix-tex).

Usage:
    python scripts/summarize_target.py --model llama-3.2-1b-instruct --seed 42
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

MERGED_ROOT = "saves_bts_merged"
PRELIM_ROOT = "saves_bts_preliminary"


def read_acc_coef(path):
    with open(path) as f:
        reader = csv.reader(f)
        tasks = next(reader)[1:]
        coefs, cols = [], {t: [] for t in tasks}
        for row in reader:
            if not row:
                continue
            coefs.append(float(row[0]))
            for t, v in zip(tasks, row[1:]):
                cols[t].append(float(v))
    return np.array(coefs), tasks, {t: np.array(v) for t, v in cols.items()}


def _exact_match(pattern):
    """exact_match from the first compute_metrics.jsonl matching, else None.

    Same source as utils._eval_exact_match (correct for every task incl. ReCoRD),
    kept dependency-free here like summarize_merged.py.
    """
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None
    metrics = {}
    with open(hits[0]) as f:
        for line in f:
            if line.strip():
                metrics.update(json.loads(line))
    return metrics.get("exact_match")


_single, _zero = {}, {}


def single_task_acc(method, model, seed, task):
    """The target's own single-task fine-tune accuracy (NAI's ceiling)."""
    key = (method, task)
    if key not in _single:
        _single[key] = _exact_match(
            f"{PRELIM_ROOT}/{method}/{model}/eval_{task}_{seed}_*/compute_metrics.jsonl")
    return _single[key]


def zero_shot_acc(model, seed, task):
    """Pretrained-model accuracy (NAI's floor); method-independent."""
    if task not in _zero:
        _zero[task] = _exact_match(
            f"{PRELIM_ROOT}/zero-shot/{model}/eval_{task}_{seed}_*/compute_metrics.jsonl")
    return _zero[task]


def nai(acc, zs, ft):
    """(acc - zero_shot) / (finetuned - zero_shot); None if inputs missing/degenerate."""
    if acc is None or zs is None or ft is None or abs(ft - zs) < 1e-9:
        return None
    return (acc - zs) / (ft - zs)


def tex_escape(s):
    return s.replace("_", r"\_")


def read_sar(path):
    """{(method, combo, target): (sar, sar_iso)} from compute_target_sar.py's CSV."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["method"], r["combo"], r["target"])] = (
                float(r["sar"]) if r["sar"] else None,
                float(r["sar_iso"]) if r["sar_iso"] else None,
            )
    return out


def build_tex_table(cells, top, model, seed, caption, label):
    """Booktabs table mirroring the printed ranking: top combos per (method, target).

    Columns: best coef for the target, accuracy and NAI at that coef, and the
    target's SAR against the mixture subspace; plus an italic references row
    per target (zero-shot and single-task FT). A bold header row separates
    methods when more than one is present.
    """
    fmt = lambda v, p=3: "--" if v is None else f"{v:.{p}f}"
    methods = sorted({m for m, _ in cells})

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    Target & Combination & coef & acc & NAI & SAR \\",
        r"    \midrule",
    ]

    for mi, method in enumerate(methods):
        if mi > 0:
            lines.append(r"    \midrule")
        if len(methods) > 1:
            lines.append(f"    \\multicolumn{{6}}{{l}}{{\\textbf{{{tex_escape(method)}}}}} \\\\")
        targets = sorted(t for m, t in cells if m == method)
        for ti, target in enumerate(targets):
            if ti > 0:
                lines.append(r"    \addlinespace")
            rs = sorted(cells[(method, target)], key=lambda r: r["best_acc"], reverse=True)
            for ri, r in enumerate(rs[:top]):
                tcell = tex_escape(target) if ri == 0 else ""
                lines.append(
                    f"    {tcell} & {tex_escape(r['combo'])} & "
                    f"{fmt(r['best_coef'], 2)} & {fmt(r['best_acc'])} & {fmt(r['nai'])} & "
                    f"{fmt(r['sar'])} \\\\")
            zs = zero_shot_acc(model, seed, target)
            ft = single_task_acc(method, model, seed, target)
            lines.append(
                f"    & \\multicolumn{{5}}{{l}}{{\\textit{{zero-shot {fmt(zs)}; "
                f"single-task fine-tune {fmt(ft)}}}}} \\\\")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _ranks(a):
    """Ranks with ties averaged (for Spearman without scipy)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a)
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):
        m = a == v
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(x, y):
    rx, ry = _ranks(x) - _ranks(x).mean(), _ranks(y) - _ranks(y).mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


# dataviz palette
TARGET_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]


def plot_sar_scatter(rows, targets, path):
    """Accuracy (at per-target best coef) vs SAR / SAR-iso, one subplot per
    target (columns) x metric (rows), each with its own axis scale and the
    per-target Spearman in the title, plus a dashed least-squares trend line.
    The best combo per target (ties included) is starred. Method blocks stack
    vertically."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = sorted({r["method"] for r in rows})
    metrics = [("sar", "SAR"), ("sar_iso", "SAR (iso)")]
    nrows = len(methods) * len(metrics)
    fig, axes = plt.subplots(nrows, len(targets), squeeze=False, sharey="col",
                             figsize=(3.2 * len(targets), 2.9 * nrows))
    for mi, method in enumerate(methods):
        for ki, (key, name) in enumerate(metrics):
            for ci, target in enumerate(targets):
                ax = axes[mi * len(metrics) + ki][ci]
                color = TARGET_COLORS[ci % len(TARGET_COLORS)]
                rs = [r for r in rows
                      if r["method"] == method and r["target"] == target
                      and r[key] is not None]
                if not rs:
                    ax.set_axis_off()
                    continue
                x = np.array([r[key] for r in rs])
                y = np.array([r["best_acc"] for r in rs])
                ax.scatter(x, y, s=38, color=color, alpha=0.85,
                           edgecolors="white", linewidths=0.6)
                if len(rs) > 1 and x.ptp() > 0:
                    slope, intercept = np.polyfit(x, y, 1)
                    xs = np.array([x.min(), x.max()])
                    ax.plot(xs, slope * xs + intercept, linestyle="--",
                            color="0.35", linewidth=1.2, zorder=2)
                top = max(r["best_acc"] for r in rs)
                for r in (r for r in rs if r["best_acc"] == top):
                    ax.scatter(r[key], r["best_acc"], marker="*", s=210,
                               color=color, edgecolors="black",
                               linewidths=0.7, zorder=5)
                title = f"{target} ($\\rho$={spearman(x, y):+.2f})"
                if len(methods) > 1:
                    title = f"{method}: {title}"
                ax.set_title(title, fontsize=10)
                ax.set_xlabel(name, fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
                # room so stars near the top edge don't get clipped
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.04 * (hi - lo))
            axes[mi * len(metrics) + ki][0].set_ylabel(
                "accuracy at per-target best coef", fontsize=9)
    fig.legend([Line2D([], [], marker="*", linestyle="", color="0.4",
                       markeredgecolor="black", markersize=12)],
               ["best combo for the target (ties included)"],
               loc="lower center", ncol=1, fontsize=9, frameon=False)
    fig.suptitle("Target transfer accuracy vs subspace alignment of the target task",
                 y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=150)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)


def build_full_tex_table(rows, targets, model, seed, caption, label):
    """Full combos x targets matrix of per-target best accuracy, one block per
    method, best combo per target in bold; references rows at the bottom."""
    fmt = lambda v, p=3: "--" if v is None else f"{v:.{p}f}"
    methods = sorted({r["method"] for r in rows})
    ncols = len(targets) + 1

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{l{'r' * len(targets)}}}",
        r"    \toprule",
        "    Combination & " + " & ".join(tex_escape(t) for t in targets) + r" \\",
        r"    \midrule",
    ]

    for mi, method in enumerate(methods):
        if mi > 0:
            lines.append(r"    \midrule")
        if len(methods) > 1:
            lines.append(f"    \\multicolumn{{{ncols}}}{{l}}{{\\textbf{{{tex_escape(method)}}}}} \\\\")
        acc = {(r["combo"], r["target"]): r["best_acc"]
               for r in rows if r["method"] == method}
        combos = sorted({c for c, _ in acc},
                        key=lambda c: (c.count("+"), c))
        best = {t: max(acc.get((c, t), float("-inf")) for c in combos) for t in targets}
        arity = None
        for combo in combos:
            if arity is not None and combo.count("+") != arity:
                lines.append(r"    \addlinespace")
            arity = combo.count("+")
            cells = []
            for t in targets:
                v = acc.get((combo, t))
                s = fmt(v)
                if v is not None and v == best[t]:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            lines.append(f"    {tex_escape(combo)} & " + " & ".join(cells) + r" \\")
        lines.append(r"    \midrule")
        zs = [fmt(zero_shot_acc(model, seed, t)) for t in targets]
        ft = [fmt(single_task_acc(method, model, seed, t)) for t in targets]
        lines.append(r"    \textit{zero-shot} & " + " & ".join(zs) + r" \\")
        lines.append(r"    \textit{single-task fine-tune} & " + " & ".join(ft) + r" \\")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=5, help="combos to print per (method, target)")
    ap.add_argument("--out", default="figures/target/target_summary.csv")
    ap.add_argument("--tex", default=None,
                    help="LaTeX table path (default: --out with .csv replaced by .tex)")
    ap.add_argument("--caption", default=None)
    ap.add_argument("--label", default="tab:target-transfer")
    ap.add_argument("--matrix", default="figures/target/target_matrix.csv",
                    help="pivot CSV path for the full combos x targets matrix")
    ap.add_argument("--matrix-tex", default=None,
                    help="LaTeX path for the full matrix (default: --matrix with .csv -> .tex)")
    ap.add_argument("--matrix-caption", default=None)
    ap.add_argument("--matrix-label", default="tab:target-transfer-full")
    ap.add_argument("--sar", default="figures/target/target_sar.csv",
                    help="target-SAR CSV from compute_target_sar.py to join in")
    ap.add_argument("--scatter", default="figures/target/target_sar_scatter.png",
                    help="accuracy-vs-SAR scatter path (skipped if no sar values)")
    args = ap.parse_args()

    if os.path.exists(args.sar):
        sar_vals = read_sar(args.sar)
    else:
        sar_vals = {}
        print(f"note: {args.sar} not found — sar columns will be empty "
              "(run scripts/compute_target_sar.py first)\n")

    target_order = []
    rows = []
    for method_dir in sorted(glob.glob(f"{MERGED_ROOT}/*/{args.model}")):
        method = method_dir.split(os.sep)[-2]
        for csvf in sorted(glob.glob(
                f"{method_dir}/*_{args.seed}_target/*_{args.seed}_target_acc_coef.csv")):
            coefs, targets, accs = read_acc_coef(csvf)
            for t in targets:
                if t not in target_order:
                    target_order.append(t)
            combo = "+".join(os.path.basename(csvf)
                             .removesuffix(f"_{args.seed}_target_acc_coef.csv").split("_"))
            for target in targets:
                curve = accs[target]
                bi = int(np.argmax(curve))
                zs = zero_shot_acc(args.model, args.seed, target)
                ft = single_task_acc(method, args.model, args.seed, target)
                sar, sar_iso = sar_vals.get((method, combo, target), (None, None))
                rows.append(dict(
                    method=method, combo=combo, n_tasks=combo.count("+") + 1,
                    target=target,
                    best_coef=float(coefs[bi]), best_acc=float(curve[bi]),
                    nai=nai(float(curve[bi]), zs, ft),
                    sar=sar, sar_iso=sar_iso,
                ))

    if not rows:
        raise SystemExit(f"no *_target_acc_coef.csv found under {MERGED_ROOT}/*/{args.model}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)\n")

    # ranking: best combo per (method, target) by per-target best accuracy
    cells = defaultdict(list)
    for r in rows:
        cells[(r["method"], r["target"])].append(r)

    fmt = lambda v, p=3: "--" if v is None else f"{v:.{p}f}"
    for (method, target) in sorted(cells):
        rs = sorted(cells[(method, target)], key=lambda r: r["best_acc"], reverse=True)
        single = single_task_acc(method, args.model, args.seed, target)
        zs = zero_shot_acc(args.model, args.seed, target)
        print(f"== {method} / target {target}  "
              f"(zero-shot: {fmt(zs)}, single-task FT: {fmt(single)}) ==")
        print(f"   {'combo':32} {'best_coef':>9} {'best_acc':>9} {'nai':>9} "
              f"{'sar':>9} {'sar_iso':>9}")
        for r in rs[:args.top]:
            print(f"   {r['combo']:32} {r['best_coef']:>9.2f} {r['best_acc']:>9.3f} "
                  f"{fmt(r['nai']):>9} {fmt(r['sar']):>9} {fmt(r['sar_iso']):>9}")
        print()

    caption = args.caption or (
        f"Top-{args.top} source-task combinations per target task for "
        f"{tex_escape(args.model)} (seed {args.seed}), ranked by accuracy at the "
        "merge coefficient that is best for the target. NAI is the normalized "
        "accuracy improvement over zero-shot relative to the target's "
        "single-task fine-tune; SAR is the subspace alignment ratio of the "
        "target's task vector against the mixture's summed subspace."
    )
    table = build_tex_table(cells, args.top, args.model, args.seed, caption, args.label)
    tex_path = args.tex or (args.out[:-4] + ".tex" if args.out.endswith(".csv")
                            else args.out + ".tex")
    with open(tex_path, "w") as f:
        f.write(table)
    print(table)
    print(f"% wrote {tex_path}\n")

    # full combos x targets matrix: pivot CSV + LaTeX
    methods = sorted({r["method"] for r in rows})
    acc = {(r["method"], r["combo"], r["target"]): r["best_acc"] for r in rows}
    with open(args.matrix, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow((["method"] if len(methods) > 1 else []) + ["combo"] + target_order)
        for method in methods:
            combos = sorted({r["combo"] for r in rows if r["method"] == method},
                            key=lambda c: (c.count("+"), c))
            for combo in combos:
                w.writerow((([method] if len(methods) > 1 else []) + [combo]
                            + [acc.get((method, combo, t)) for t in target_order]))
    print(f"wrote {args.matrix}\n")

    matrix_caption = args.matrix_caption or (
        f"Per-target best merged accuracy of every source-task combination for "
        f"{tex_escape(args.model)} (seed {args.seed}); each cell is the accuracy "
        "at the merge coefficient that is best for that target, best combination "
        "per target in bold."
    )
    matrix_table = build_full_tex_table(rows, target_order, args.model,
                                        args.seed, matrix_caption, args.matrix_label)
    matrix_tex = args.matrix_tex or (args.matrix[:-4] + ".tex"
                                     if args.matrix.endswith(".csv")
                                     else args.matrix + ".tex")
    with open(matrix_tex, "w") as f:
        f.write(matrix_table)
    print(matrix_table)
    print(f"% wrote {matrix_tex}\n")

    if any(r["sar"] is not None for r in rows):
        plot_sar_scatter(rows, target_order, args.scatter)
        print(f"wrote {args.scatter} (+.pdf)")
    else:
        print(f"no sar values joined — skipping scatter ({args.scatter})")


if __name__ == "__main__":
    main()
