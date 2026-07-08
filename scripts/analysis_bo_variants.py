"""Which GP features make simulated BO find good source mixtures fastest?

Retrospective BO on the precomputed transfer grid from
scripts/summarize_target.py (figures/target/target_summary.csv): the GP
predicts transfer accuracy and candidates come from the already-evaluated 26
combos, so no training runs happen here — but still run it via SLURM
(scripts/slurm/analysis.sh), not on the login node.

Feature variants for the GP input:
  SAR / SAR_iso            the target's subspace alignment against the mixture
  SAR (norm) / iso (norm)  the same divided by mixture size (arity de-biased)
  Binary                   the 5-dim source-subset indicator vector
  Binary + SAR_iso         indicator plus the alignment score

Each variant runs --seeds times; compared against a random-search baseline on
mean-best-at-budget, win rate (found the true best subset) and evaluations to
get within 1% of the true best. Writes convergence curves, summary bars and
bo_variants_summary.csv to --out.

Usage:
    PYTHONPATH=src python scripts/analysis_bo_variants.py \
        --summary figures/target/target_summary.csv --out figures/bo_variants
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesian_task_selection import (combo_to_binary, simulate_bo_on_grid,
                                     simulate_random_on_grid)

warnings.filterwarnings("ignore")

# dataviz palette
METHOD_COLORS = {
    "Random":            "#888780",
    "SAR":               "#4a3aa7",
    "SAR_iso":           "#1baf7a",
    "SAR (norm)":        "#e87ba4",
    "SAR_iso (norm)":    "#008300",
    "Binary":            "#eb6834",
    "Binary + SAR_iso":  "#2a78d6",
}

FEATURE_CONFIGS = {
    "SAR":              ["sar"],
    "SAR_iso":          ["sar_iso"],
    "SAR (norm)":       ["sar_norm"],
    "SAR_iso (norm)":   ["sar_iso_norm"],
    "Binary":           ["binary"],
    "Binary + SAR_iso": ["binary", "sar_iso"],
}

# the headline variants shown in the convergence plot (all appear in the
# table / bars / wins / speed sections)
PLOT_METHODS = ["Random", "SAR", "SAR_iso", "Binary", "Binary + SAR_iso"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="figures/target/target_summary.csv")
    ap.add_argument("--method", default="lora")
    ap.add_argument("--out", default="figures/bo_variants")
    ap.add_argument("--seeds", type=int, default=5, help="BO repetitions per variant")
    ap.add_argument("--n-initial", type=int, default=3)
    ap.add_argument("--n-iterations", type=int, default=10)
    ap.add_argument("--random-trials", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.summary)
    df = df[df["method"] == args.method].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"no rows for method {args.method} in {args.summary}")
    if df["sar"].isna().any():
        raise SystemExit("sar column has missing values — run "
                         "scripts/compute_target_sar.py then summarize_target.py")

    df["binary"] = df["combo"].apply(combo_to_binary)
    df["sar_norm"] = df["sar"] / df["n_tasks"]
    df["sar_iso_norm"] = df["sar_iso"] / df["n_tasks"]

    targets = sorted(df["target"].unique())
    budget = args.n_initial + args.n_iterations
    methods = ["Random"] + list(FEATURE_CONFIGS)

    all_results = {}
    for target in targets:
        tdf = df[df["target"] == target].reset_index(drop=True)
        true_best = tdf["best_acc"].max()
        true_best_combo = tdf.loc[tdf["best_acc"].idxmax(), "combo"]
        print(f"\n{'=' * 60}\nTARGET: {target} "
              f"(true best: {true_best:.4f} = {true_best_combo})\n{'=' * 60}")

        results = {}
        random_curves = simulate_random_on_grid(tdf, budget, args.random_trials)
        results["Random"] = random_curves
        print(f"  {'Random':<18s} mean_best@{budget}="
              f"{random_curves[:, -1].mean():.4f}")

        for name, feat_cols in FEATURE_CONFIGS.items():
            curves = np.array([
                simulate_bo_on_grid(tdf, feat_cols, args.n_initial,
                                    args.n_iterations, seed=s)["best_so_far"]
                for s in range(args.seeds)])
            results[name] = curves
            print(f"  {name:<18s} mean_best@{budget}={curves[:, -1].mean():.4f}"
                  f"  median={np.median(curves[:, -1]):.4f}")
        all_results[target] = results

    # ========================================================
    # Figure 1: convergence per target
    # ========================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    steps = np.arange(1, budget + 1)
    for ax, target in zip(axes.flat, targets):
        true_best = df[df["target"] == target]["best_acc"].max()
        ax.axhline(y=true_best, color="#333333", linestyle="--", linewidth=1,
                   alpha=0.6, label=f"True best ({true_best:.3f})")
        for name in PLOT_METHODS:
            curves = all_results[target][name]
            is_random = name == "Random"
            ax.plot(steps, curves.mean(axis=0), color=METHOD_COLORS[name],
                    label=name, linewidth=1.5 if is_random else 2,
                    alpha=0.7 if is_random else 1.0)
        ax.axvline(x=args.n_initial + 0.5, color="#CCCCCC", linestyle=":",
                   linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Evaluations (training runs)")
        ax.set_ylabel("Best accuracy found")
        ax.set_title(target, fontsize=13)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.1)
        ax.set_xlim(1, budget)
    fig.suptitle("BO convergence by GP feature set", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{args.out}/bo_convergence_all_variants.png", dpi=150,
                bbox_inches="tight")
    fig.savefig(f"{args.out}/bo_convergence_all_variants.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print("\nsaved: bo_convergence_all_variants.png")

    # ========================================================
    # Figure 2: mean best accuracy at budget, per target
    # ========================================================
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5))
    for ax, target in zip(axes, targets):
        true_best = df[df["target"] == target]["best_acc"].max()
        means = [all_results[target][m][:, -1].mean() for m in methods]
        stds = [all_results[target][m][:, -1].std() for m in methods]
        ax.barh(range(len(methods)), means, xerr=stds, height=0.6, capsize=3,
                alpha=0.85, color=[METHOD_COLORS[m] for m in methods])
        ax.axvline(x=true_best, color="#333333", linestyle="--", linewidth=1,
                   alpha=0.5)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=9)
        ax.set_xlabel("Best accuracy found")
        ax.set_title(target, fontsize=12)
        ax.invert_yaxis()
    fig.suptitle(f"Mean best accuracy after {budget} evaluations "
                 f"({args.seeds} seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{args.out}/bo_summary_bars.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{args.out}/bo_summary_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved: bo_summary_bars.png")

    # ========================================================
    # Summary table + CSV, wins, speed
    # ========================================================
    print("\n" + "=" * 80 + "\nSUMMARY: mean best accuracy at budget (+- std)\n"
          + "=" * 80)
    print(f"{'method':<18s}" + "".join(f" | {t:>12s}" for t in targets))
    rows = []
    for m in methods:
        line = f"{m:<18s}"
        row = dict(method=m)
        for target in targets:
            curves = all_results[target][m]
            mean, std = curves[:, -1].mean(), curves[:, -1].std()
            line += f" | {mean:>6.4f}+-{std:.3f}"
            row[f"{target}_mean"] = mean
            row[f"{target}_std"] = std
        print(line)
        rows.append(row)
    print(f"{'True best':<18s}" + "".join(
        f" | {df[df['target'] == t]['best_acc'].max():>6.4f}      "
        for t in targets))
    pd.DataFrame(rows).to_csv(f"{args.out}/bo_variants_summary.csv", index=False)
    print(f"\nsaved: bo_variants_summary.csv")

    print("\n" + "=" * 80 + "\nWINS: how often is the true best subset found?\n"
          + "=" * 80)
    for m in methods:
        wins = {}
        for target in targets:
            true_best = df[df["target"] == target]["best_acc"].max()
            curves = all_results[target][m]
            wins[target] = (curves[:, -1] >= true_best - 1e-6).mean()
        detail = "  ".join(f"{t}={w:.0%}" for t, w in wins.items())
        print(f"  {m:<18s} avg={np.mean(list(wins.values())):.0%}  ({detail})")

    print("\n" + "=" * 80 + "\nSPEED: mean evaluations to within 1% of true best\n"
          + "=" * 80)
    for m in methods:
        speeds = {}
        for target in targets:
            true_best = df[df["target"] == target]["best_acc"].max()
            curves = all_results[target][m]
            speeds[target] = np.mean([
                next((i + 1 for i, v in enumerate(c) if v >= true_best - 0.01),
                     budget)
                for c in curves])
        detail = "  ".join(f"{t}={s:.1f}" for t, s in speeds.items())
        print(f"  {m:<18s} avg={np.mean(list(speeds.values())):.1f}  ({detail})")

    print(f"\ndone — figures in {args.out}")


if __name__ == "__main__":
    main()
