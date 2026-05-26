"""Quantitative summary of merge quality across all combos/methods.

For each (method, combo) it picks the *best shared scaling coefficient* (the coef
maximizing mean task accuracy — the standard task-arithmetic merging protocol) and
reports, against each task's single-task ceiling:

  mean_acc      mean over the combo's tasks of accuracy at the best shared coef
  min_acc       worst task at that coef (interference indicator)
  mean_ret      mean over tasks of (merged_acc / single_task_acc)
  min_ret       worst task's retention (the "sacrificed task" indicator)
  oracle_min_ret  worst task's retention if EACH task could pick its own best coef
                  (upper bound; if this is still low the task is unrecoverable at any coef)

Writes figures/merged/merge_summary.csv and prints aggregates by method and by arity.
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


_single = {}


def single_task_acc(method, model, seed, task):
    key = (method, task)
    if key not in _single:
        hits = sorted(glob.glob(
            f"{PRELIM_ROOT}/{method}/{model}/eval_{task}_{seed}_*/predict_results.json"))
        _single[key] = json.load(open(hits[0]))["predict_accuracy"] if hits else None
    return _single[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="figures/merged/merge_summary.csv")
    args = ap.parse_args()

    rows = []
    for method_dir in sorted(glob.glob(f"{MERGED_ROOT}/*/{args.model}")):
        method = method_dir.split(os.sep)[-2]
        for csvf in sorted(glob.glob(f"{method_dir}/*_{args.seed}_best/*_{args.seed}_acc_coef.csv")):
            coefs, tasks, accs = read_acc_coef(csvf)
            mat = np.vstack([accs[t] for t in tasks])           # [T, n_coef]
            mean_curve = mat.mean(axis=0)
            bi = int(np.argmax(mean_curve))                     # best shared coef index
            at_best = mat[:, bi]
            singles = np.array([single_task_acc(method, args.model, args.seed, t) for t in tasks])
            ret = at_best / singles
            oracle_min_ret = float((mat.max(axis=1) / singles).min())  # each task own best coef
            rows.append(dict(
                method=method, combo="+".join(tasks), n_tasks=len(tasks),
                best_coef=float(coefs[bi]),
                mean_acc=float(at_best.mean()), min_acc=float(at_best.min()),
                mean_ret=float(ret.mean()), min_ret=float(ret.min()),
                oracle_min_ret=oracle_min_ret,
            ))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)\n")

    # aggregate by method
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    print(f"{'method':8} {'n':>3} {'mean_acc':>9} {'min_acc':>9} {'mean_ret':>9} {'min_ret':>9} {'oracle_min_ret':>15}")
    for method in sorted(by_method):
        rs = by_method[method]
        agg = lambda k: np.mean([r[k] for r in rs])
        print(f"{method:8} {len(rs):>3} {agg('mean_acc'):>9.3f} {agg('min_acc'):>9.3f} "
              f"{agg('mean_ret'):>9.3f} {agg('min_ret'):>9.3f} {agg('oracle_min_ret'):>15.3f}")

    # aggregate by method x arity
    print(f"\n{'method':8} {'arity':>5} {'k':>3} {'mean_ret':>9} {'min_ret':>9}")
    cells = defaultdict(list)
    for r in rows:
        cells[(r["method"], r["n_tasks"])].append(r)
    for (method, k) in sorted(cells):
        rs = cells[(method, k)]
        print(f"{method:8} {k:>5} {len(rs):>3} "
              f"{np.mean([r['mean_ret'] for r in rs]):>9.3f} {np.mean([r['min_ret'] for r in rs]):>9.3f}")


if __name__ == "__main__":
    main()
