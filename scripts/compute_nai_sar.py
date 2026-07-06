"""Compute NAI and SAR per (method, combo, task) and write one CSV.

For every merged task combination found under ``saves_bts_merged`` this script
computes, per task:

  nai      Normalized Accuracy Improvement (Marczak et al.):
           (merged - zero_shot) / (finetuned - zero_shot), with the merged
           accuracy taken at the best shared scaling coef.
  sar      Subspace Alignment Ratio against the summed task-vector subspace.
  sar_iso  SAR against the isotropic (equal-singular-value) subspace.

NAI reads only cached accuracies (fast). SAR loads the per-task fine-tuned and
pretrained checkpoints and runs SVDs, so it is the slow part — use --device cuda
when a GPU is available, and --no-sar to skip it for a quick NAI-only pass.

Note: SAR is most meaningful for the full-weight-delta methods (base / freeze,
filtered to self_attn/mlp). For lora the task vectors hold the raw lora_A/lora_B
factors, so its SAR is computed over those factors, not the effective B@A delta.

Usage:
    PYTHONPATH=src python scripts/compute_nai_sar.py \
        --model llama-3.2-1b-instruct --seed 42 --device cuda \
        --out figures/nai_sar.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os

import utils

MERGED_ROOT = "saves_bts_merged"


def discover_combos(model: str, seed: int, methods: list[str] | None):
    """Yield (method, tasks) for each merged combo CSV, sorted and de-duped."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for method_dir in sorted(glob.glob(f"{MERGED_ROOT}/*/{model}")):
        method = method_dir.split(os.sep)[-2]
        if methods and method not in methods:
            continue
        for csvf in sorted(glob.glob(f"{method_dir}/*_{seed}_best/*_{seed}_acc_coef.csv")):
            with open(csvf) as f:
                tasks = tuple(next(csv.reader(f))[1:])  # header minus scaling_coef
            out.append((method, tasks))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="restrict to these methods (default: all found)")
    ap.add_argument("--device", default="cpu", help="device for SAR SVDs (e.g. cuda)")
    ap.add_argument("--rank-threshold", type=float, default=0.95)
    ap.add_argument("--no-sar", action="store_true", help="skip SAR (NAI only)")
    ap.add_argument("--out", default="figures/nai_sar.csv")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    combos = discover_combos(args.model, args.seed, args.methods)
    if not combos:
        raise SystemExit(
            f"no *_acc_coef.csv found under {MERGED_ROOT}/*/{args.model} (seed {args.seed})"
        )

    rows = []
    for method, tasks in combos:
        combo = "+".join(tasks)
        try:
            nai = utils.compute_nai(args.model, method, args.seed, list(tasks))
        except (FileNotFoundError, KeyError) as e:
            print(f"[skip nai] {method} {combo}: {e}")
            continue

        sar = sar_iso = {t: None for t in tasks}
        if not args.no_sar:
            try:
                tvs = utils.build_task_vectors(args.model, method, args.seed, list(tasks))
                sar = utils.sar(tvs, list(tasks),
                                rank_threshold=args.rank_threshold, device=args.device)
                sar_iso = utils.sar(tvs, list(tasks), rank_threshold=args.rank_threshold,
                                    device=args.device, iso=True)
            except (FileNotFoundError, KeyError) as e:
                print(f"[skip sar] {method} {combo}: {e}")

        for task in tasks:
            rows.append(dict(
                method=method, combo=combo, n_tasks=len(tasks), task=task,
                nai=nai.get(task), sar=sar.get(task), sar_iso=sar_iso.get(task),
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
