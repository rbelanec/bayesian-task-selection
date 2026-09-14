#!/usr/bin/env python3
"""Which merge-sweep combinations actually produced a usable results CSV?

`sacct` is the obvious way to ask "did that array task succeed?", but slurmdbd is
not reachable on this cluster (`_open_persist_conn: Connection refused`), so the
job state is simply unavailable once a task leaves the queue. This checks the
thing we actually care about instead: did combination i write its acc_coef CSV?

The combination indexing is identical to eval.py / eval_target.py — the task list
is read straight out of those files (via AST, so importing torch/llamafactory
isn't needed) and expanded with the same `combinations(tasks, r) for r in 2..n`.

    scripts/combo_status.py target                 # every missing index
    scripts/combo_status.py target --count         # just how many are missing
    scripts/combo_status.py target --start 96 --n 48   # one batch's worth
    scripts/combo_status.py source --paths         # missing index + its CSV path
    scripts/combo_status.py target --require-nonzero stsb   # + stale-metric ones

Existence is not always enough: a metric computed by the wrong code path leaves a
perfectly well-formed CSV behind. `--require-nonzero COL` additionally reports
combinations whose CSV exists but holds an all-zero COL, which is what a fixed
metric leaves behind in the combinations evaluated before the fix.

Exit status is 1 when anything is missing, 0 when the range is complete, so it
can drive a shell conditional.
"""

import argparse
import ast
import csv
import os
import signal
import sys
from itertools import combinations

# `combo_status.py target | head` is the common interactive call; without this the
# closed pipe surfaces as a BrokenPipeError traceback.
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (module, task-list variable, save-dir suffix, csv-name infix)
KINDS = {
    "source": ("src/eval/eval.py", "TASKS", "best", ""),
    "target": ("src/eval/eval_target.py", "SOURCE_TASKS", "target", "target_"),
}
# accept the job-script names too, so this mirrors run_eval_batched.sh's argument
KINDS["eval_array.sh"] = KINDS["source"]
KINDS["eval_target_array.sh"] = KINDS["target"]


def _literals(path, names):
    """Values of top-level `NAME = <literal>` assignments, without importing."""
    tree = ast.parse(open(path).read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in names:
                    found[tgt.id] = ast.literal_eval(node.value)
    missing = set(names) - set(found)
    if missing:
        raise SystemExit(f"could not read {sorted(missing)} from {path}")
    return found


def expected_csvs(kind):
    """Ordered list of the CSV each combination index is supposed to write."""
    module, tasks_var, suffix, infix = KINDS[kind]
    cfg = _literals(os.path.join(REPO_ROOT, module), [tasks_var, "METHODS", "MODELS", "SEEDS"])
    tasks = cfg[tasks_var]
    # eval.py loops over methods/models/seeds; the sweep has only ever been run
    # with one of each, and a longer list would mean several CSVs per index.
    for var in ("METHODS", "MODELS", "SEEDS"):
        if len(cfg[var]) != 1:
            raise SystemExit(f"{module}: expected exactly one {var}, got {cfg[var]}")
    (method,), (model,), (seed,) = cfg["METHODS"], cfg["MODELS"], cfg["SEEDS"]

    combos = [c for r in range(2, len(tasks) + 1) for c in combinations(tasks, r)]
    out = []
    for combo in combos:
        stem = f"{'_'.join(combo)}_{seed}"
        save_dir = f"saves_bts_merged/{method}/{model}/{stem}_{suffix}"
        out.append(f"{save_dir}/{stem}_{infix}acc_coef.csv")
    return out


def stale_metric(path, columns):
    """Does an existing CSV hold an all-zero value for one of `columns`?

    `stsb` was scored by exact string match against a float regression target
    until `ComputeCorrelation` landed in src/metrics.py on 2026-07-30, which
    pins all 40 coefficient rows at exactly 0.0 — a plausible-looking file that
    is entirely uninformative. Those combinations have to be re-run, and the
    zeros are their fingerprint, so this needs no timestamps and stays correct
    if the files are copied, rsynced or touched.

    A real metric can of course be 0.0 at a single coefficient; being 0.0 at
    *every* coefficient is the part that does not happen by chance. It is not
    strictly impossible — `copa` and `piqa` each have a handful of genuinely
    all-zero columns — so only pass columns whose fix you are re-running for.
    """
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return True  # unreadable is as good as absent; re-running rewrites it
    if not rows:
        return True
    for col in columns:
        if col not in rows[0]:
            raise SystemExit(f"{path}: no column {col!r} (have: {', '.join(rows[0])})")
        try:
            values = [float(r[col]) for r in rows]
        except (TypeError, ValueError):
            return True
        if all(v == 0.0 for v in values):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=sorted(KINDS), help="which sweep to check")
    ap.add_argument("--start", type=int, default=0, help="first combination index (default 0)")
    ap.add_argument("--n", type=int, default=None, help="how many indices to check (default: to the end)")
    ap.add_argument("--count", action="store_true", help="print only the number missing")
    ap.add_argument("--paths", action="store_true", help="print the expected CSV path alongside each index")
    ap.add_argument(
        "--require-nonzero", metavar="COL", action="append", default=[],
        help="also report an index whose CSV exists but has COL all zeros, i.e. was "
             "evaluated before COL's metric was fixed (repeatable)",
    )
    args = ap.parse_args()

    csvs = expected_csvs(args.kind)
    stop = len(csvs) if args.n is None else min(len(csvs), args.start + args.n)
    if args.start >= len(csvs):
        raise SystemExit(f"--start {args.start} is past the last combination ({len(csvs) - 1})")

    missing = []
    for i in range(args.start, stop):
        path = os.path.join(REPO_ROOT, csvs[i])
        if not os.path.exists(path):
            missing.append(i)
        elif args.require_nonzero and stale_metric(path, args.require_nonzero):
            missing.append(i)

    if args.count:
        print(len(missing))
    else:
        for i in missing:
            print(f"{i}\t{csvs[i]}" if args.paths else i)

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
