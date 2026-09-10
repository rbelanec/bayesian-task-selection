"""Permutation test: do winning mixtures draw on the target's own task family?

Table 2 of the paper shows that each target's best-transferring mixture tends
to include sources from the same task family as the target. This script tests
whether that pattern exceeds chance, rather than asserting it from the
examples.

For every target we take its accuracy-optimal mixture from the exhaustive
grid (figures/target/target_matrix.csv), breaking ties toward the smaller
mixture as the paper does throughout, and record whether that mixture contains
at least one source from the target's own family. The null is a uniformly
random mixture of the same size drawn from the 12 sources, which holds mixture
size fixed so the statistic cannot be driven by the size distribution alone.

No checkpoints are touched, so this runs anywhere in a few seconds.

Usage:
    PYTHONPATH=src python scripts/analysis_family_alignment.py \
        --matrix figures/target/target_matrix.csv --out figures/analysis
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

# Task families follow the five categories of Section 4 (the sentiment and
# linguistic-acceptability category is one family, as in the paper's task list).
FAMILY = {
    "mnli": "NLI", "qnli": "NLI", "snli": "NLI", "anli_r1": "NLI",
    "rte": "NLI", "cb": "NLI", "scitail": "NLI",
    "qqp": "PARA", "paws": "PARA", "mrpc": "PARA", "stsb": "PARA",
    "sst2": "SENT", "imdb": "SENT", "cr": "SENT",
    "rotten_tomatoes": "SENT", "cola": "SENT",
    "record": "QA", "squad_v2": "QA", "boolq": "QA", "multirc": "QA",
    "hellaswag": "CS", "winogrande": "CS", "copa": "CS", "piqa": "CS",
}
SOURCES = [
    "mnli", "qnli", "qqp", "sst2", "record", "snli",
    "anli_r1", "paws", "imdb", "squad_v2", "hellaswag", "winogrande",
]


def best_mixtures(matrix_path: Path) -> dict[str, list[str]]:
    """Accuracy-optimal mixture per target, ties broken toward fewer sources."""
    rows = list(csv.DictReader(matrix_path.open()))
    targets = [c for c in rows[0] if c != "combo"]
    winners = {}
    for target in targets:
        scored = [(r["combo"], float(r[target])) for r in rows]
        best = max(acc for _, acc in scored)
        tied = [combo for combo, acc in scored if abs(acc - best) < 1e-12]
        winners[target] = sorted(tied, key=lambda c: (c.count("+"), c))[0].split("+")
    return winners


def hits(winners: dict[str, list[str]]) -> int:
    """Targets whose winning mixture contains a same-family source."""
    return sum(
        FAMILY[t] in {FAMILY[s] for s in mix} for t, mix in winners.items()
    )


def null_distribution(
    winners: dict[str, list[str]], trials: int, seed: int
) -> collections.Counter:
    rng = random.Random(seed)
    dist: collections.Counter = collections.Counter()
    for _ in range(trials):
        count = sum(
            FAMILY[t] in {FAMILY[s] for s in rng.sample(SOURCES, len(mix))}
            for t, mix in winners.items()
        )
        dist[count] += 1
    return dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path,
                    default=Path("figures/target/target_matrix.csv"))
    ap.add_argument("--out", type=Path, default=Path("figures/analysis"))
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    winners = best_mixtures(args.matrix)
    observed = hits(winners)
    dist = null_distribution(winners, args.trials, args.seed)
    expected = sum(k * v for k, v in dist.items()) / args.trials
    p_value = sum(v for k, v in dist.items() if k >= observed) / args.trials

    per_target = []
    for target, mix in winners.items():
        same = [s for s in mix if FAMILY[s] == FAMILY[target]]
        pool = [s for s in SOURCES if FAMILY[s] == FAMILY[target]]
        per_target.append({
            "target": target,
            "family": FAMILY[target],
            "mixture": "+".join(mix),
            "size": len(mix),
            "same_family_sources": len(same),
            "same_family_share": len(same) / len(mix),
            "pool_share": len(pool) / len(SOURCES),
            "aligned": bool(same),
        })

    print(f"{'target':16s} {'fam':5s} {'k':>2s} {'same':>5s} {'share':>6s} "
          f"{'pool':>6s}  aligned")
    for row in sorted(per_target, key=lambda r: (not r["aligned"], r["target"])):
        print(f"{row['target']:16s} {row['family']:5s} {row['size']:2d} "
              f"{row['same_family_sources']:5d} {row['same_family_share']:6.2f} "
              f"{row['pool_share']:6.2f}  {'yes' if row['aligned'] else 'NO'}")
    print(f"\nobserved  {observed}/{len(winners)} targets aligned")
    print(f"expected  {expected:.1f}/{len(winners)} under random mixtures "
          f"of the same sizes")
    print(f"p-value   {p_value:.4f}  (P[count >= observed], {args.trials} draws)")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "family_alignment.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_target[0]))
        writer.writeheader()
        writer.writerows(per_target)
    summary = {
        "n_targets": len(winners),
        "observed": observed,
        "expected": expected,
        "p_value": p_value,
        "trials": args.trials,
        "seed": args.seed,
        "null_distribution": {str(k): dist[k] / args.trials for k in sorted(dist)},
    }
    (args.out / "family_alignment.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out / 'family_alignment.csv'} and "
          f"{args.out / 'family_alignment.json'}")


if __name__ == "__main__":
    main()
