"""Retrospective analysis: does target SAR predict transfer accuracy?

Operates entirely on the precomputed ground truth from
scripts/summarize_target.py (figures/target/target_summary.csv — per (combo,
target): transfer accuracy, NAI, and the target's SAR/iso-SAR against the
mixture subspace). No checkpoints are touched, so this is cheap, but run it
via SLURM (scripts/slurm/analysis.sh), not on the login node.

Sections:
  1. Spearman/Pearson correlations of SAR and iso-SAR vs accuracy and NAI,
     per target and overall (+ correlations.csv)
  2. Best subset per target by each criterion (acc / NAI / SAR / iso-SAR) —
     does selecting by SAR recover the accuracy-best subset?
  3. Simulated BO on the precomputed grid: GP with X = SAR, Y = accuracy,
     vs random search, greedy-by-SAR, and the pure argmax-SAR pick
  4. Ranking quality: rank correlation + top-5 overlap of SAR vs accuracy
  5. SAR vs iso-SAR: which predicts accuracy better per target

Figures: {sar,sar_iso}_vs_{accuracy,nai}_per_target, three_panel_comparison,
bo_convergence, ranking_comparison (png + pdf).

Usage:
    PYTHONPATH=src python scripts/analysis_correlations.py \
        --summary figures/target/target_summary.csv --out figures/analysis
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
from scipy.stats import pearsonr, spearmanr

from bayesian_task_selection import (simulate_bo_on_grid, simulate_greedy_by,
                                     simulate_random_on_grid)

warnings.filterwarnings("ignore")

# dataviz palette, same target->color mapping as summarize_target.py's scatter
TARGET_COLORS = {"mrpc": "#2a78d6", "boolq": "#1baf7a",
                 "rte": "#eda100", "cola": "#e34948"}
C_BO, C_GREEDY, C_SARPICK, C_NEUTRAL = "#4a3aa7", "#e34948", "#1baf7a", "#888780"


def savefig(fig, out_dir, name):
    fig.savefig(f"{out_dir}/{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="figures/target/target_summary.csv")
    ap.add_argument("--method", default="lora")
    ap.add_argument("--out", default="figures/analysis")
    ap.add_argument("--seeds", type=int, default=10, help="BO repetitions")
    ap.add_argument("--n-initial", type=int, default=3)
    ap.add_argument("--n-iterations", type=int, default=10)
    ap.add_argument("--random-trials", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.summary)
    df = df[df["method"] == args.method].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"no rows for method {args.method} in {args.summary}")
    if df["sar"].isna().any():
        raise SystemExit("sar column has missing values — run "
                         "scripts/compute_target_sar.py then summarize_target.py")
    targets = sorted(df["target"].unique())
    budget = args.n_initial + args.n_iterations
    print(f"loaded {len(df)} rows, {df['combo'].nunique()} combos, "
          f"targets: {targets}")

    # ========================================================
    # 1. Correlations: SAR / iso-SAR vs accuracy / NAI
    # ========================================================
    print("\n" + "=" * 80 + "\n1. CORRELATIONS\n" + "=" * 80)
    print(f"\n{'target':<8} | {'SAR-acc':>10} {'SAR-NAI':>10} | "
          f"{'iso-acc':>10} {'iso-NAI':>10}")
    corr_rows = []
    for target in targets + ["overall"]:
        tdf = df if target == "overall" else df[df["target"] == target]
        r = {f"{m}_{p}": spearmanr(tdf[m], tdf[p], nan_policy="omit")[0]
             for m in ("sar", "sar_iso") for p in ("best_acc", "nai")}
        print(f"{target:<8} | {r['sar_best_acc']:>10.4f} {r['sar_nai']:>10.4f} | "
              f"{r['sar_iso_best_acc']:>10.4f} {r['sar_iso_nai']:>10.4f}")
        corr_rows.append(dict(target=target, **r))
    pd.DataFrame(corr_rows).to_csv(f"{args.out}/correlations.csv", index=False)
    print(f"\nsaved: correlations.csv")

    # Figures 1-4: per-target scatters (SAR / iso-SAR vs accuracy / NAI)
    for metric, metric_label in (("sar", "SAR"), ("sar_iso", "SAR (iso)")):
        for perf, perf_label, perf_slug in (("best_acc", "Transfer Accuracy",
                                             "accuracy"),
                                            ("nai", "NAI", "nai")):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            for ax, target in zip(axes.flat, targets):
                tdf = df[df["target"] == target].dropna(subset=[metric, perf])
                c = TARGET_COLORS[target]
                ax.scatter(tdf[metric], tdf[perf], c=c, s=50, alpha=0.7,
                           edgecolors="white", linewidth=0.5)
                z = np.polyfit(tdf[metric], tdf[perf], 1)
                x_line = np.linspace(tdf[metric].min(), tdf[metric].max(), 100)
                ax.plot(x_line, np.poly1d(z)(x_line), "--", color=c, alpha=0.5,
                        linewidth=1.5)
                rho_s = spearmanr(tdf[metric], tdf[perf])[0]
                rho_p = pearsonr(tdf[metric], tdf[perf])[0]
                ax.set_title(f"{target}\nSpearman $\\rho$={rho_s:.3f}, "
                             f"Pearson r={rho_p:.3f}", fontsize=12)
                ax.set_xlabel(metric_label)
                ax.set_ylabel(perf_label)
                ax.grid(True, alpha=0.15)
            fig.suptitle(f"{metric_label} vs {perf_label} per target task", y=1.0)
            fig.tight_layout()
            savefig(fig, args.out, f"{metric}_vs_{perf_slug}_per_target")

    # Figure 3: three-panel overview (paper figure)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panels = [("sar", "best_acc", "SAR", "Transfer Accuracy"),
              ("sar_iso", "best_acc", "SAR (iso)", "Transfer Accuracy"),
              ("sar", "sar_iso", "SAR", "SAR (iso)")]
    for ax, (xk, yk, xl, yl) in zip(axes, panels):
        for target in targets:
            tdf = df[df["target"] == target]
            ax.scatter(tdf[xk], tdf[yk], c=TARGET_COLORS[target], s=40,
                       alpha=0.7, edgecolors="white", linewidth=0.5,
                       label=target)
        rho_s = spearmanr(df[xk], df[yk], nan_policy="omit")[0]
        rho_p = pearsonr(df[xk], df[yk])[0]
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{xl} vs {yl}\nSpearman $\\rho$={rho_s:.3f}, "
                     f"Pearson r={rho_p:.3f}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)
    fig.tight_layout()
    savefig(fig, args.out, "three_panel_comparison")

    # ========================================================
    # 2. Best subset per target by each criterion
    # ========================================================
    print("\n" + "=" * 80 + "\n2. BEST SUBSET PER TARGET\n" + "=" * 80)
    for target in targets:
        tdf = df[df["target"] == target]
        by = {crit: tdf.loc[tdf[crit].idxmax()]
              for crit in ("best_acc", "nai", "sar", "sar_iso")}
        print(f"\n{target}:")
        for crit, row in by.items():
            print(f"  best by {crit:<9}: {row['combo']:32s} "
                  f"acc={row['best_acc']:.4f}")
        if by["sar"]["combo"] == by["best_acc"]["combo"]:
            print("  SAR pick matches the accuracy-best subset")
        else:
            ranked = tdf.sort_values("best_acc", ascending=False).reset_index()
            rank = ranked[ranked["combo"] == by["sar"]["combo"]].index[0] + 1
            gap = by["best_acc"]["best_acc"] - by["sar"]["best_acc"]
            print(f"  SAR pick ranks #{rank}/{len(tdf)} by accuracy "
                  f"(gap {gap:.4f})")

    # ========================================================
    # 3. Simulated BO vs random vs greedy-SAR
    # ========================================================
    print("\n" + "=" * 80 + "\n3. SIMULATED BO (X=SAR, Y=accuracy)\n" + "=" * 80)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    steps = np.arange(1, budget + 1)
    bo_summary = []
    for ax, target in zip(axes.flat, targets):
        tdf = df[df["target"] == target].reset_index(drop=True)
        true_best = tdf["best_acc"].max()
        true_best_combo = tdf.loc[tdf["best_acc"].idxmax(), "combo"]

        bo_curves = np.array([
            simulate_bo_on_grid(tdf, ["sar"], args.n_initial,
                                args.n_iterations, seed=s, fallback_col="sar")
            ["best_so_far"] for s in range(args.seeds)])
        random_curves = simulate_random_on_grid(tdf, budget, args.random_trials)
        greedy_curve, greedy_pick = simulate_greedy_by(tdf, "sar")
        sar_row = tdf.loc[tdf["sar"].idxmax()]

        ax.axhline(y=true_best, color=C_NEUTRAL, linestyle="--", linewidth=1,
                   label=f"True best ({true_best:.3f})")
        rm = random_curves.mean(axis=0)
        ax.plot(steps, rm, color=C_NEUTRAL, linewidth=1.5, label="Random")
        ax.plot(steps[:len(greedy_curve[:budget])], greedy_curve[:budget],
                color=C_GREEDY, linewidth=1.5, label="Greedy (by SAR)")
        ax.plot(steps, bo_curves.mean(axis=0), color=C_BO, linewidth=2,
                label="BO (SAR-guided)")
        ax.axhline(y=sar_row["best_acc"], color=C_SARPICK, linestyle=":",
                   linewidth=1.5,
                   label=f"argmax-SAR pick ({sar_row['best_acc']:.3f})")
        ax.axvline(x=args.n_initial + 0.5, color="#CCCCCC", linestyle=":",
                   linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Best accuracy found")
        ax.set_title(target, fontsize=13)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.1)
        ax.set_xlim(1, budget)

        evals_to_near = [next((i + 1 for i, v in enumerate(c)
                               if v >= true_best - 0.01), budget)
                         for c in bo_curves]
        bo_summary.append(dict(
            target=target, true_best=true_best, true_best_combo=true_best_combo,
            sar_pick=sar_row["combo"], sar_pick_acc=sar_row["best_acc"],
            greedy_pick=greedy_pick, greedy_first_acc=greedy_curve[0],
            bo_median_best=float(np.median(bo_curves[:, -1])),
            bo_mean_evals_to_near_opt=float(np.mean(evals_to_near)),
            random_mean_best=float(rm[-1]),
        ))
    fig.suptitle("Simulated task selection: BO vs random vs greedy", y=1.0)
    fig.tight_layout()
    savefig(fig, args.out, "bo_convergence")
    pd.DataFrame(bo_summary).to_csv(f"{args.out}/bo_simulation_summary.csv",
                                    index=False)

    print(f"\n{'target':<8} {'true best':>10} {'SAR pick':>10} {'BO median':>10} "
          f"{'BO evals->1%':>13} {'random':>8}")
    for s in bo_summary:
        print(f"{s['target']:<8} {s['true_best']:>10.4f} "
              f"{s['sar_pick_acc']:>10.4f} {s['bo_median_best']:>10.4f} "
              f"{s['bo_mean_evals_to_near_opt']:>13.1f} "
              f"{s['random_mean_best']:>8.4f}")

    # ========================================================
    # 4. Ranking quality: SAR order vs accuracy order
    # ========================================================
    print("\n" + "=" * 80 + "\n4. SAR RANKING QUALITY\n" + "=" * 80)
    for target in targets:
        tdf = df[df["target"] == target].copy()
        rho, pval = spearmanr(tdf["best_acc"].rank(), tdf["sar"].rank())
        top5_acc = set(tdf.nlargest(5, "best_acc")["combo"])
        top5_sar = set(tdf.nlargest(5, "sar")["combo"])
        print(f"  {target:6s}: rank correlation rho={rho:.4f} (p={pval:.3e}), "
              f"top-5 overlap={len(top5_acc & top5_sar)}/5")

    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 6))
    for ax, target in zip(axes, targets):
        tdf = (df[df["target"] == target]
               .sort_values("best_acc", ascending=False).reset_index(drop=True))
        tdf["sar_rank"] = tdf["sar"].rank(ascending=False).astype(int)
        top10 = tdf.head(10)
        colors = [C_BO if r <= 5 else "#D3D1C7" for r in top10["sar_rank"]]
        ax.barh(range(len(top10)), top10["best_acc"], color=colors, height=0.7)
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels([f"{c}\n(SAR rank #{r})" for c, r in
                            zip(top10["combo"], top10["sar_rank"])], fontsize=7)
        ax.set_xlabel("Accuracy")
        ax.set_title(target, fontsize=12)
        ax.invert_yaxis()
    fig.suptitle("Top 10 subsets by accuracy (colored = also top 5 by SAR)",
                 y=1.02)
    fig.tight_layout()
    savefig(fig, args.out, "ranking_comparison")

    # ========================================================
    # 5. SAR vs iso-SAR: which predicts accuracy better?
    # ========================================================
    print("\n" + "=" * 80 + "\n5. SAR vs iso-SAR\n" + "=" * 80)
    print(f"{'target':<8} | {'rho(SAR,acc)':>13} {'rho(iso,acc)':>13} | better")
    for target in targets + ["overall"]:
        tdf = df if target == "overall" else df[df["target"] == target]
        r_sar = spearmanr(tdf["sar"], tdf["best_acc"])[0]
        r_iso = spearmanr(tdf["sar_iso"], tdf["best_acc"])[0]
        better = "SAR" if abs(r_sar) > abs(r_iso) else "iso-SAR"
        print(f"{target:<8} | {r_sar:>13.4f} {r_iso:>13.4f} | {better}")

    print(f"\ndone — figures in {args.out}")


if __name__ == "__main__":
    main()
