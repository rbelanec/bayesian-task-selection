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

Writes the full table to --out, prints the top --top combos per (method,
target) ranked by best_acc, and writes the same ranking as a booktabs LaTeX
table to --tex (with a per-target references row: zero-shot and single-task FT
accuracy).

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


def build_tex_table(cells, top, model, seed, caption, label):
    """Booktabs table mirroring the printed ranking: top combos per (method, target).

    Columns: best coef for the target, accuracy and NAI at that coef; plus an
    italic references row per target (zero-shot and single-task FT). A bold
    header row separates methods when more than one is present.
    """
    fmt = lambda v, p=3: "--" if v is None else f"{v:.{p}f}"
    methods = sorted({m for m, _ in cells})

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Target & Combination & coef & acc & NAI \\",
        r"    \midrule",
    ]

    for mi, method in enumerate(methods):
        if mi > 0:
            lines.append(r"    \midrule")
        if len(methods) > 1:
            lines.append(f"    \\multicolumn{{5}}{{l}}{{\\textbf{{{tex_escape(method)}}}}} \\\\")
        targets = sorted(t for m, t in cells if m == method)
        for ti, target in enumerate(targets):
            if ti > 0:
                lines.append(r"    \addlinespace")
            rs = sorted(cells[(method, target)], key=lambda r: r["best_acc"], reverse=True)
            for ri, r in enumerate(rs[:top]):
                tcell = tex_escape(target) if ri == 0 else ""
                lines.append(
                    f"    {tcell} & {tex_escape(r['combo'])} & "
                    f"{fmt(r['best_coef'], 2)} & {fmt(r['best_acc'])} & {fmt(r['nai'])} \\\\")
            zs = zero_shot_acc(model, seed, target)
            ft = single_task_acc(method, model, seed, target)
            lines.append(
                f"    & \\multicolumn{{4}}{{l}}{{\\textit{{zero-shot {fmt(zs)}; "
                f"single-task fine-tune {fmt(ft)}}}}} \\\\")

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
    args = ap.parse_args()

    rows = []
    for method_dir in sorted(glob.glob(f"{MERGED_ROOT}/*/{args.model}")):
        method = method_dir.split(os.sep)[-2]
        for csvf in sorted(glob.glob(
                f"{method_dir}/*_{args.seed}_target/*_{args.seed}_target_acc_coef.csv")):
            coefs, targets, accs = read_acc_coef(csvf)
            combo = "+".join(os.path.basename(csvf)
                             .removesuffix(f"_{args.seed}_target_acc_coef.csv").split("_"))
            for target in targets:
                curve = accs[target]
                bi = int(np.argmax(curve))
                zs = zero_shot_acc(args.model, args.seed, target)
                ft = single_task_acc(method, args.model, args.seed, target)
                rows.append(dict(
                    method=method, combo=combo, n_tasks=combo.count("+") + 1,
                    target=target,
                    best_coef=float(coefs[bi]), best_acc=float(curve[bi]),
                    nai=nai(float(curve[bi]), zs, ft),
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
        print(f"   {'combo':32} {'best_coef':>9} {'best_acc':>9} {'nai':>9}")
        for r in rs[:args.top]:
            print(f"   {r['combo']:32} {r['best_coef']:>9.2f} {r['best_acc']:>9.3f} "
                  f"{fmt(r['nai']):>9}")
        print()

    caption = args.caption or (
        f"Top-{args.top} source-task combinations per target task for "
        f"{tex_escape(args.model)} (seed {args.seed}), ranked by accuracy at the "
        "merge coefficient that is best for the target. NAI is the normalized "
        "accuracy improvement over zero-shot relative to the target's "
        "single-task fine-tune."
    )
    table = build_tex_table(cells, args.top, args.model, args.seed, caption, args.label)
    tex_path = args.tex or (args.out[:-4] + ".tex" if args.out.endswith(".csv")
                            else args.out + ".tex")
    with open(tex_path, "w") as f:
        f.write(table)
    print(table)
    print(f"% wrote {tex_path}")


if __name__ == "__main__":
    main()
