"""Which GP features make simulated BO find good source mixtures fastest?

Retrospective BO on the precomputed transfer grid from
scripts/summarize_target.py (figures/target/target_summary.csv): the GP
predicts transfer accuracy and candidates come from the already-evaluated
mixtures (4083 of them at 12 sources), so no training runs happen here — but
still run it via SLURM (scripts/slurm/analysis.sh), not on the login node.

Feature variants for the GP input:
  SAR / SAR_iso            the target's subspace alignment against the mixture
  SAR (norm) / iso (norm)  the same divided by mixture size (arity de-biased)
  Binary                   the source-subset indicator vector (one dim per
                           source task, read from src/eval/eval_target.py)
  Binary + SAR_iso         indicator plus the alignment score

Each variant runs --seeds times, compared against a random-search baseline on
mean-best-at-budget, win rate (found the true best subset) and evaluations to get
within 1% of the true best. The baseline is *paired* — it replays the BO runs' own
initial designs, so the curves coincide up to --n-initial instead of differing by
sampling noise in the stretch before any GP is fit.

Writes to --out: bo_convergence_all_variants and bo_summary_bars (png+pdf),
bo_variants_summary.csv (mean+-std at budget), bo_variants_wins_speed.csv (the
win-rate and evals-to-1% tables, per target) and bo_variants_curves.npz (raw
convergence curves, so further metrics need no re-run).

Usage:
    PYTHONPATH=src python scripts/analysis_bo_variants.py \
        --summary figures/target/target_summary.csv --out figures/bo_variants
"""

from __future__ import annotations

import argparse
import math
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesian_task_selection import (combo_to_binary, simulate_bo_on_grid,
                                     simulate_random_on_grid)
from combo_names import display_combo, display_name

warnings.filterwarnings("ignore")

# The baseline is *paired*: it replays BO's own initial designs, so its curve
# coincides with every BO variant up to n_initial and diverges only where the
# acquisition function starts choosing. Labelled plainly as "Random" because it is
# the only baseline shown; the pairing is a property of how it is sampled, not a
# competing method. simulate_random_on_grid still supports an unpaired many-trial
# mode, deliberately not plotted here.
RANDOM = "Random"

# dataviz palette. The baseline is neutral gray, not a categorical hue — it is the
# reference the variants are read against, not a seventh contender.
METHOD_COLORS = {
    RANDOM:       "#888780",
    "SAR":               "#4a3aa7",
    "SAR_iso":           "#1baf7a",
    "SAR (norm)":        "#e87ba4",
    "SAR_iso (norm)":    "#008300",
    "Binary":            "#eb6834",
    "Binary + SAR_iso":  "#2a78d6",
}

# Figure/table labels. The keys above are *identifiers*: they name the GP feature
# sets, index METHOD_COLORS/FEATURE_CONFIGS, and are baked into the
# "{target}|{method}" keys of bo_variants_curves.npz — renaming them would orphan
# every cached curve and force the ~2 h of GP fits again. So the published name of
# the method lives here instead, exactly as combo_names.DISPLAY_NAMES keeps task
# ids separate from the names that appear on an axis.
#
# `Random` is deliberately not a BOTAS variant: it is the baseline the variants are
# read against, not a configuration of the method (see the RANDOM comment above).
DISPLAY_METHODS = {
    RANDOM:              "Random",
    "SAR":               "BOTAS (SAR)",
    "SAR_iso":           "BOTAS (iso-SAR)",
    "SAR (norm)":        "BOTAS (SAR, norm)",
    "SAR_iso (norm)":    "BOTAS (iso-SAR, norm)",
    "Binary":            "BOTAS (binary)",
    "Binary + SAR_iso":  "BOTAS (binary + iso-SAR)",
}


def method_label(method: str) -> str:
    """Published name for a feature-set id, or the id itself if it has no entry."""
    return DISPLAY_METHODS.get(method, method)


#: Column width for every method column printed below. Derived from the labels
#: rather than hardcoded, so a renamed variant cannot overflow the column; the
#: floor keeps "Best possible" aligned with the variants it sits under.
LABEL_WIDTH = max(13, max(len(v) for v in DISPLAY_METHODS.values()))


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
PLOT_METHODS = [RANDOM, "SAR", "SAR_iso", "Binary", "Binary + SAR_iso"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="figures/target/target_summary.csv")
    ap.add_argument("--method", default="base")
    ap.add_argument("--out", default="figures/bo_variants")
    ap.add_argument("--seeds", type=int, default=10, help="BO repetitions per variant")
    ap.add_argument("--n-initial", type=int, default=3)
    # 60, not the original 10: at 4083 candidates a 13-evaluation budget is
    # 0.3% of the space and every method degenerates to sampling noise.
    ap.add_argument("--n-iterations", type=int, default=60)
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="recompute every target instead of continuing from "
                         "bo_variants_partial.npz")
    ap.add_argument("--from-curves", metavar="NPZ", nargs="?",
                    const="", default=None,
                    help="skip the GP fits entirely and redraw every figure and "
                         "table from a previous run's bo_variants_curves.npz "
                         "(defaults to the one in --out). Seconds instead of hours; "
                         "use it for restyling, relabelling or new metrics.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.from_curves is not None:
        path = args.from_curves or f"{args.out}/bo_variants_curves.npz"
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} not found — it is written by a full run, so the first "
                f"run after this feature landed still has to do the GP fits.")
        print(f"replotting from {path} (no BO)")
        make_outputs(*load_curves(path), args.out)
        return
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
    true_best_by = {t: df[df["target"] == t]["best_acc"].max() for t in targets}
    budget = args.n_initial + args.n_iterations
    methods = [RANDOM] + list(FEATURE_CONFIGS)

    # Checkpoint per target. Run 79104 spent 3h50m on GP fits and lost all of it to
    # a wall-clock timeout because nothing hit disk until every target was done.
    # Now each target is appended as it completes, so a timeout costs at most one
    # target and a resubmit continues where it stopped.
    ckpt = f"{args.out}/bo_variants_partial.npz"
    all_results = {}
    if args.resume and os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=False)
        done = [t for t in targets if f"{t}|{methods[0]}" in z.files]
        all_results = {t: {m: z[f"{t}|{m}"] for m in methods} for t in done}
        print(f"[resume] {ckpt} already holds {len(done)}/{len(targets)} targets: "
              f"{', '.join(done)}", flush=True)

    for target in targets:
        if target in all_results:
            continue
        tdf = df[df["target"] == target].reset_index(drop=True)
        true_best = tdf["best_acc"].max()
        true_best_combo = tdf.loc[tdf["best_acc"].idxmax(), "combo"]
        print(f"\n{'=' * 60}\nTARGET: {display_name(target)} "
              f"(best possible: {true_best:.4f} = {display_combo(true_best_combo)})"
              f"\n{'=' * 60}", flush=True)

        results = {}
        # Paired: replays the same permutations the BO runs below start from.
        results[RANDOM] = simulate_random_on_grid(
            tdf, budget, seeds=range(args.seeds))
        print(f"  {method_label(RANDOM):<{LABEL_WIDTH}s} mean_best@{budget}="
              f"{results[RANDOM][:, -1].mean():.4f}", flush=True)

        for name, feat_cols in FEATURE_CONFIGS.items():
            curves = np.array([
                simulate_bo_on_grid(tdf, feat_cols, args.n_initial,
                                    args.n_iterations, seed=s)["best_so_far"]
                for s in range(args.seeds)])
            results[name] = curves
            print(f"  {method_label(name):<{LABEL_WIDTH}s} mean_best@{budget}="
                  f"{curves[:, -1].mean():.4f}"
                  f"  median={np.median(curves[:, -1]):.4f}", flush=True)
        # The pairing is the whole point of RANDOM: up to n_initial it and
        # every BO variant are replaying the same permutation, so the curves must
        # match exactly. If they ever stop matching, the two sampling paths have
        # drifted apart and the baseline is silently unpaired again.
        paired_prefix = results[RANDOM][:, :args.n_initial]
        for name in FEATURE_CONFIGS:
            if not np.allclose(results[name][:, :args.n_initial], paired_prefix):
                raise SystemExit(
                    f"{target}/{name}: paired random diverges from BO's initial "
                    f"design — simulate_random_on_grid(seeds=...) no longer "
                    f"mirrors simulate_bo_on_grid's RNG stream")

        all_results[target] = results
        # Flush this target to disk before starting the next one.
        np.savez_compressed(
            ckpt, **{f"{t}|{m}": all_results[t][m]
                     for t in all_results for m in methods})
        print(f"  [checkpoint] {len(all_results)}/{len(targets)} targets saved",
              flush=True)

    save_curves(f"{args.out}/bo_variants_curves.npz", all_results, targets,
                methods, budget, args.n_initial, args.seeds, true_best_by)
    if os.path.exists(ckpt):
        os.remove(ckpt)  # superseded by the complete bo_variants_curves.npz
    make_outputs(all_results, targets, methods, budget, args.n_initial,
                 args.seeds, true_best_by, args.out)


def save_curves(path, all_results, targets, methods, budget, n_initial, seeds,
                true_best_by):
    """Persist the raw convergence curves.

    Written *before* the figures, because the curves are the only expensive part
    (~2 h of GP fits at 10 seeds) and everything else is derived from them. With
    this file on disk, `--from-curves` redraws every figure and table in seconds.
    """
    np.savez_compressed(
        path, budget=budget, n_initial=n_initial, seeds=seeds,
        targets=np.array(targets), methods=np.array(methods),
        true_best=np.array([true_best_by[t] for t in targets]),
        **{f"{t}|{m}": all_results[t][m] for t in targets for m in methods})
    print(f"saved: {os.path.basename(path)}")


def load_curves(path):
    """Inverse of `save_curves`."""
    z = np.load(path, allow_pickle=False)
    targets = [str(t) for t in z["targets"]]
    methods = [str(m) for m in z["methods"]]
    all_results = {t: {m: z[f"{t}|{m}"] for m in methods} for t in targets}
    true_best_by = dict(zip(targets, z["true_best"].tolist()))
    return (all_results, targets, methods, int(z["budget"]),
            int(z["n_initial"]), int(z["seeds"]), true_best_by)


def _grid(n):
    ncols = min(4, n)
    return math.ceil(n / ncols), ncols


def make_outputs(all_results, targets, methods, budget, n_initial, seeds,
                 true_best_by, out):
    """Every figure, console table and CSV — derived purely from the curves."""
    nrows, ncols = _grid(len(targets))

    # ---------------------------------------------------------- Figure 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows),
                             squeeze=False)
    for ax in axes.flat[len(targets):]:
        ax.set_visible(False)
    steps = np.arange(1, budget + 1)
    for ax, target in zip(axes.flat, targets):
        true_best = true_best_by[target]
        ax.axhline(y=true_best, color="#333333", linestyle="--", linewidth=1,
                   alpha=0.6, label=f"Best possible ({true_best:.3f})")
        for name in PLOT_METHODS:
            curves = all_results[target][name]
            is_random = name == RANDOM
            ax.plot(steps, curves.mean(axis=0), color=METHOD_COLORS[name],
                    label=method_label(name),
                    linewidth=1.6 if is_random else 2.2,
                    alpha=0.8 if is_random else 1.0)
        ax.axvline(x=n_initial + 0.5, color="#BBBBBB", linestyle=":", linewidth=1)
        ax.annotate("GP starts", xy=(n_initial + 0.5, 0.02), xycoords=("data", "axes fraction"),
                    fontsize=7, color="#888780", rotation=90, va="bottom", ha="right")
        ax.set_xlabel("Mixtures evaluated", fontsize=10)
        ax.set_ylabel("Best transfer accuracy found", fontsize=10)
        ax.set_title(display_name(target), fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        ax.grid(True, alpha=0.12)
        ax.set_xlim(1, budget)
        # Zoom to where the curves live; a line chart has no zero-baseline duty,
        # and anchoring at 0 flattened every difference into one band.
        lo = min(all_results[target][m].mean(axis=0)[0] for m in methods)
        ax.set_ylim(lo - 0.02 * (true_best - lo + 1e-9), true_best + 0.03 * (true_best - lo + 1e-9) + 1e-3)
        ax.tick_params(labelsize=9)
    fig.suptitle("How fast does each GP feature set find the best source mixture?",
                 fontsize=16, fontweight="bold", y=1.005)
    fig.tight_layout()
    for ext, kw in (("png", dict(dpi=150)), ("pdf", {})):
        fig.savefig(f"{out}/bo_convergence_all_variants.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("saved: bo_convergence_all_variants.png")

    # ---------------------------------------------------------- Figure 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows),
                             squeeze=False)
    for ax in axes.flat[len(targets):]:
        ax.set_visible(False)
    for ax, target in zip(axes.flat, targets):
        true_best = true_best_by[target]
        means = [all_results[target][m][:, -1].mean() for m in methods]
        stds = [all_results[target][m][:, -1].std() for m in methods]
        ax.barh(range(len(methods)), means, xerr=stds, height=0.62, capsize=3,
                alpha=0.9, color=[METHOD_COLORS[m] for m in methods])
        ax.axvline(x=true_best, color="#333333", linestyle="--", linewidth=1,
                   alpha=0.6, label=f"Best possible ({true_best:.3f})")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([method_label(m) for m in methods], fontsize=9)
        ax.set_xlabel("Best transfer accuracy found", fontsize=10)
        ax.set_title(display_name(target), fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        ax.invert_yaxis()
        ax.tick_params(labelsize=9)
        ax.grid(True, axis="x", alpha=0.12)
    fig.suptitle(f"Best mixture found after {budget} evaluations "
                 f"({seeds} seeds, error bars = 1 sd)",
                 fontsize=16, fontweight="bold", y=1.005)
    fig.tight_layout()
    for ext, kw in (("png", dict(dpi=150)), ("pdf", {})):
        fig.savefig(f"{out}/bo_summary_bars.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("saved: bo_summary_bars.png")

    # ---------------------------------------------------- tables + CSVs
    label = {t: display_name(t) for t in targets}
    w = max(12, max(len(v) for v in label.values()))
    print("\n" + "=" * 80 + "\nSUMMARY: mean best accuracy at budget (+- std)\n" + "=" * 80)
    print(f"{'method':<{LABEL_WIDTH}s}" + "".join(f" | {label[t]:>{w}s}" for t in targets))
    rows = []
    for m in methods:
        line, row = f"{method_label(m):<{LABEL_WIDTH}s}", dict(method=m)
        for t in targets:
            c = all_results[t][m]
            mean, std = c[:, -1].mean(), c[:, -1].std()
            line += f" | {mean:>{w-7}.4f}+-{std:.3f}"
            row[f"{t}_mean"], row[f"{t}_std"] = mean, std
        print(line)
        rows.append(row)
    print(f"{'Best possible':<{LABEL_WIDTH}s}" + "".join(f" | {true_best_by[t]:>{w}.4f}" for t in targets))
    pd.DataFrame(rows).to_csv(f"{out}/bo_variants_summary.csv", index=False)
    print("\nsaved: bo_variants_summary.csv")

    wins_by = {m: {t: (all_results[t][m][:, -1] >= true_best_by[t] - 1e-6).mean()
                   for t in targets} for m in methods}
    speed_by = {m: {t: float(np.mean([
        next((i + 1 for i, v in enumerate(c) if v >= true_best_by[t] - 0.01), budget)
        for c in all_results[t][m]])) for t in targets} for m in methods}

    print("\n" + "=" * 80 + "\nWINS: how often is the true best mixture found?\n" + "=" * 80)
    for m in methods:
        detail = "  ".join(f"{label[t]}={wins_by[m][t]:.0%}" for t in targets)
        print(f"  {method_label(m):<{LABEL_WIDTH}s} avg={np.mean(list(wins_by[m].values())):.0%}  ({detail})")

    print("\n" + "=" * 80 + "\nSPEED: mean evaluations to within 1% of the best\n" + "=" * 80)
    for m in methods:
        detail = "  ".join(f"{label[t]}={speed_by[m][t]:.1f}" for t in targets)
        print(f"  {method_label(m):<{LABEL_WIDTH}s} avg={np.mean(list(speed_by[m].values())):.1f}  ({detail})")

    ws = []
    for m in methods:
        row = {"method": m, "wins_avg": np.mean(list(wins_by[m].values())),
               "evals_to_1pct_avg": np.mean(list(speed_by[m].values()))}
        for t in targets:
            row[f"{t}_wins"], row[f"{t}_evals_to_1pct"] = wins_by[m][t], speed_by[m][t]
        ws.append(row)
    pd.DataFrame(ws).to_csv(f"{out}/bo_variants_wins_speed.csv", index=False)
    print("saved: bo_variants_wins_speed.csv")
    print(f"\ndone — figures in {out}")


if __name__ == "__main__":
    main()
