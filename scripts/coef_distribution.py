# Copyright 2025 the Anonymous Authors.
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

"""Where do the winning merge coefficients actually land?

Reads a summary CSV written by scripts/summarize_target.py or
scripts/summarize_merged.py and reports the distribution of its `best_coef`
column: min/max/median, the share falling in each interval, the most common
individual values, and — the reason this exists — how much of the mass is
pinned against the ends of the evaluated grid.

Pinning is what makes the number actionable. A large share sitting on the
smallest coefficient the sweep evaluates means the true optimum is probably
below it and is being clipped, so the reported accuracies understate the merge;
a grid whose upper half never wins is spending evaluations where no answer
lives. The grid itself is read from src/eval/eval_target.py (COEF_MAX,
N_EVAL_POINTS) so "the smallest coefficient evaluated" is the sweep's, not
merely the smallest one observed.

Usage:
    python scripts/coef_distribution.py                       # target summary
    python scripts/coef_distribution.py --summary figures/merged/merge_summary.csv
    python scripts/coef_distribution.py --group-by n_tasks --width 0.25
    python scripts/coef_distribution.py --out figures/target/coef_distribution.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict

import numpy as np

from combo_names import coef_grid

# Denser at the bottom than the top: the winning coefficients pile up near zero,
# so uniform bins put almost everything in the first one and say nothing. Use
# --width for uniform bins or --edges for your own.
DEFAULT_EDGES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


def load(path, coef_col, group_col):
    """[(group, coef)] from a summary CSV; group is None when not grouping."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} is empty")
        if coef_col not in reader.fieldnames:
            raise SystemExit(
                f"{path} has no '{coef_col}' column (has {reader.fieldnames}) — "
                "expected a summary written by summarize_target.py or summarize_merged.py")
        if group_col and group_col not in reader.fieldnames:
            raise SystemExit(
                f"{path} has no '{group_col}' column (has {reader.fieldnames})")
        out = []
        for r in reader:
            if r[coef_col] == "":
                continue
            out.append((r[group_col] if group_col else None, float(r[coef_col])))
    if not out:
        raise SystemExit(f"no usable '{coef_col}' values in {path}")
    return out


def histogram(coefs, edges):
    """[(lo, hi, count)] over half-open bins, the last one closed at the top."""
    counts = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        last = i == len(edges) - 2
        n = sum(1 for c in coefs
                if (lo <= c <= hi if last else lo <= c < hi))
        counts.append((lo, hi, n))
    return counts


def describe(coefs, edges, grid_lo, grid_hi, tol):
    """Summary stats for one group of coefficients."""
    a = np.asarray(coefs, dtype=float)
    return dict(
        n=len(a),
        min=float(a.min()), max=float(a.max()),
        mean=float(a.mean()), median=float(np.median(a)),
        at_grid_min=int(np.sum(np.abs(a - grid_lo) < tol)),
        at_grid_max=int(np.sum(np.abs(a - grid_hi) < tol)),
        hist=histogram(a, edges),
    )


def fmt_row(label, d, width):
    pinned_lo = d["at_grid_min"] / d["n"]
    pinned_hi = d["at_grid_max"] / d["n"]
    return (f"{label:{width}} {d['n']:>7} {d['min']:>7.2f} {d['max']:>7.2f} "
            f"{d['median']:>8.2f} {d['mean']:>7.3f} {pinned_lo:>9.1%} {pinned_hi:>9.1%}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="figures/target/target_summary.csv",
                    help="summary CSV to read (default the target summary)")
    ap.add_argument("--coef-col", default="best_coef",
                    help="column holding the winning coefficient")
    ap.add_argument("--group-by", default=None,
                    help="column to break the distribution down by "
                         "(default: 'target' if present, else 'method', else none)")
    ap.add_argument("--no-group", action="store_true", help="report the overall row only")
    ap.add_argument("--edges", type=float, nargs="+", default=None,
                    help=f"interval edges (default {DEFAULT_EDGES})")
    ap.add_argument("--width", type=float, default=None,
                    help="use uniform bins of this width up to the grid max instead")
    ap.add_argument("--top", type=int, default=8,
                    help="how many individual coefficient values to list (0 to skip)")
    ap.add_argument("--out", default=None, help="write the per-interval table to this CSV")
    args = ap.parse_args()

    grid, step = coef_grid()
    grid_lo, grid_hi = grid[0], grid[-1]
    tol = step / 2

    if args.edges:
        edges = sorted(args.edges)
    elif args.width:
        n = int(np.ceil(grid_hi / args.width))
        edges = [round(args.width * i, 10) for i in range(n + 1)]
    else:
        edges = DEFAULT_EDGES

    # Resolve the grouping column against what the file actually has.
    group_col = args.group_by
    if not args.no_group and group_col is None:
        with open(args.summary, newline="") as f:
            fields = csv.DictReader(f).fieldnames or []
        group_col = next((c for c in ("target", "method") if c in fields), None)
    if args.no_group:
        group_col = None

    data = load(args.summary, args.coef_col, group_col)
    coefs = [c for _, c in data]

    print(f"{args.summary}: {len(coefs)} rows, column '{args.coef_col}'")
    print(f"evaluated grid: {grid_lo:.2f} .. {grid_hi:.2f} in steps of {step:.2f} "
          f"({len(grid)} points, from src/eval/eval_target.py)\n")

    overall = describe(coefs, edges, grid_lo, grid_hi, tol)

    label_w = max([len("overall")] + [len(str(g)) for g, _ in data if g is not None]) + 2
    print(f"{'group':{label_w}} {'n':>7} {'min':>7} {'max':>7} {'median':>8} {'mean':>7} "
          f"{'at grid':>9} {'at grid':>9}")
    print(f"{'':{label_w}} {'':>7} {'':>7} {'':>7} {'':>8} {'':>7} {'min':>9} {'max':>9}")
    print(fmt_row("overall", overall, label_w))

    groups = {}
    if group_col:
        by_group = defaultdict(list)
        for g, c in data:
            by_group[g].append(c)
        for g in sorted(by_group):
            groups[g] = describe(by_group[g], edges, grid_lo, grid_hi, tol)
            print(fmt_row(g, groups[g], label_w))

    # --- interval shares ---
    print(f"\ninterval shares (overall, n={overall['n']}):")
    print(f"   {'interval':>16} {'count':>8} {'share':>8} {'cumulative':>11}")
    cum = 0
    for i, (lo, hi, n) in enumerate(overall["hist"]):
        cum += n
        closing = "]" if i == len(overall["hist"]) - 1 else ")"
        print(f"   {f'[{lo:.2f}, {hi:.2f}{closing}':>16} {n:>8} "
              f"{n / overall['n']:>8.1%} {cum / overall['n']:>11.1%}")
    outside = overall["n"] - cum
    if outside:
        print(f"   {'(outside edges)':>16} {outside:>8} {outside / overall['n']:>8.1%}")

    if args.top:
        print(f"\nmost common individual coefficients:")
        for c, n in Counter(round(c, 10) for c in coefs).most_common(args.top):
            print(f"   {c:>6.2f} {n:>8} {n / overall['n']:>8.1%}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "n", "min", "max", "median", "mean",
                        "at_grid_min", "at_grid_max", "lo", "hi", "count", "share"])
            for label, d in [("overall", overall)] + sorted(groups.items()):
                for lo, hi, n in d["hist"]:
                    w.writerow([label, d["n"], f"{d['min']:.4f}", f"{d['max']:.4f}",
                                f"{d['median']:.4f}", f"{d['mean']:.4f}",
                                d["at_grid_min"], d["at_grid_max"],
                                f"{lo:.4f}", f"{hi:.4f}", n, f"{n / d['n']:.6f}"])
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
