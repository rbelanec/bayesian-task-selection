"""Compute SAR of each TARGET task within every source-task mixture.

For every merged source combination found under ``saves_bts_merged`` this
script builds the summed-task-vector subspace from the SOURCE task vectors
only, projects each TARGET task's own task vector (from its single-task
fine-tune under saves_bts_preliminary) onto it, and reports the alignment:

  sar      Subspace Alignment Ratio of the target task vector against the
           source mixture's summed subspace — how much of the target's update
           already lives in the directions the merge keeps.
  sar_iso  Same against the isotropic (equal-singular-value) subspace.

This is the geometric counterpart of the transfer accuracies in
figures/target/target_summary.csv: a task-selection signal that needs no
target-task evaluation of the merged model (only the target's single-task FT).

Task vectors are loaded once per method and reused across combos. As with
compute_nai_sar.py: for lora the vectors hold raw lora_A/lora_B factors, so
SAR is computed over those factors, not the effective B@A delta.

Usage:
    PYTHONPATH=src python scripts/compute_target_sar.py \
        --model llama-3.2-1b-instruct --seed 42 --device cuda \
        --out figures/target/target_sar.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os

import utils

MERGED_ROOT = "saves_bts_merged"
TARGET_TASKS = ["mrpc", "boolq", "rte", "cola"]


def discover_combos(model: str, seed: int, methods: list[str] | None):
    """Yield (method, tasks) for each evaluated target-transfer combo."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for method_dir in sorted(glob.glob(f"{MERGED_ROOT}/*/{model}")):
        method = method_dir.split(os.sep)[-2]
        if methods and method not in methods:
            continue
        for csvf in sorted(glob.glob(
                f"{method_dir}/*_{seed}_target/*_{seed}_target_acc_coef.csv")):
            tasks = tuple(os.path.basename(csvf)
                          .removesuffix(f"_{seed}_target_acc_coef.csv").split("_"))
            out.append((method, tasks))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="restrict to these methods (default: all found)")
    ap.add_argument("--targets", nargs="*", default=TARGET_TASKS)
    ap.add_argument("--device", default="cpu", help="device for SAR SVDs (e.g. cuda)")
    ap.add_argument("--rank-threshold", type=float, default=0.95)
    ap.add_argument("--out", default="figures/target/target_sar.csv")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    combos = discover_combos(args.model, args.seed, args.methods)
    if not combos:
        raise SystemExit(
            f"no *_target_acc_coef.csv found under {MERGED_ROOT}/*/{args.model} "
            f"(seed {args.seed})"
        )

    rows = []
    for method in sorted({m for m, _ in combos}):
        source_tasks = sorted({t for m, ts in combos if m == method for t in ts})
        try:
            source_tvs = dict(zip(source_tasks, utils.build_task_vectors(
                args.model, method, args.seed, source_tasks)))
            target_tvs = utils.build_task_vectors(
                args.model, method, args.seed, args.targets)
        except (FileNotFoundError, KeyError) as e:
            print(f"[skip method] {method}: {e}")
            continue

        for m, tasks in combos:
            if m != method:
                continue
            combo = "+".join(tasks)
            tvs = [source_tvs[t] for t in tasks]
            sar = utils.sar(tvs, list(tasks), rank_threshold=args.rank_threshold,
                            device=args.device,
                            probe_vectors=target_tvs, probe_tasks=args.targets)
            sar_iso = utils.sar(tvs, list(tasks), rank_threshold=args.rank_threshold,
                                device=args.device, iso=True,
                                probe_vectors=target_tvs, probe_tasks=args.targets)
            for target in args.targets:
                rows.append(dict(
                    method=method, combo=combo, n_tasks=len(tasks), target=target,
                    sar=sar.get(target), sar_iso=sar_iso.get(target),
                ))
            print(f"[ok] {method} {combo}")

    if not rows:
        raise SystemExit("no rows computed (every combo was skipped)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
