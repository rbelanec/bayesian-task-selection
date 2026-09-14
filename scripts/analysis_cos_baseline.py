#!/usr/bin/env python3
"""Does cosine similarity to the target pick the right source mixture?

Head-to-head of the cheap task-selection signals against the brute-force answer
in figures/target/target_summary.csv. For each target task the grid holds the
transfer accuracy of all 4083 source mixtures (at each mixture's best
coefficient), so "which mixture would signal X have chosen, and how much
accuracy does that cost versus the mixture that actually wins?" is answerable
exactly, with no new evaluation.

Signals compared (all need only single-task fine-tunes, never a merged-model
evaluation — the property that makes them worth having):

  cos_mix   cos(summed mixture task vector, target task vector)   [compute_task_cos.py]
  dot_mix   <summed mixture task vector, target task vector>      [compute_task_cos.py]
  ts_cos    DTVG Target Similarity, cosine flavor                 [compute_task_cos.py]
  ts_dot    DTVG Target Similarity, their dot product (Eq. 2)     [compute_task_cos.py]
  kc_cos    DTVG Knowledge Consistency, cosine flavor             [compute_task_cos.py]
  kc_dot    DTVG Knowledge Consistency, their dot product         [compute_task_cos.py]
  sar       target's Subspace Alignment Ratio vs the mixture      [compute_target_sar.py]
  sar_iso   same against the isotropic spectrum                   [compute_target_sar.py]

Cosine and the dot product are crossed with two aggregations (summed mixture vs
per-member mean) so the comparison isolates the similarity choice from the
aggregation choice; see compute_task_cos.py for why all four orderings differ.

Reference rows put those numbers on a scale:

  oracle    the accuracy-best mixture (regret 0 by construction)
  random    a uniformly random mixture = the grid mean, i.e. what picking
            blind gets you; the honest floor any signal has to beat
  all       the all-12-sources mixture — "AllForOne", merge everything, the
            thing you would do with no selection method at all

Per (target, signal) it reports the selected mixture, its accuracy and NAI, the
regret against the oracle, where the pick lands in the true ranking, the
Spearman correlation of the signal with accuracy over the whole grid, and the
top-k overlap.

Usage:
    PYTHONPATH=src python scripts/analysis_cos_baseline.py \
        --summary figures/target/target_summary.csv \
        --cos figures/target/target_cos.csv \
        --out figures/analysis/cos_baseline.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from combo_names import display_combo, display_name, source_tasks

COS_SIGNALS = ["cos_mix", "dot_mix", "ts_cos", "ts_dot", "kc_cos", "kc_dot"]
SAR_SIGNALS = ["sar", "sar_iso"]


def load(summary_path, cos_path, method):
    df = pd.read_csv(summary_path)
    df = df[df["method"] == method].copy()
    if df.empty:
        raise SystemExit(f"no rows for method {method!r} in {summary_path}")

    if os.path.exists(cos_path):
        cos = pd.read_csv(cos_path)
        cos = cos[cos["method"] == method]
        keep = ["combo", "target"] + [c for c in COS_SIGNALS if c in cos.columns]
        before = len(df)
        df = df.merge(cos[keep], on=["combo", "target"], how="left")
        assert len(df) == before, "cos join changed the row count"
        missing = df[COS_SIGNALS[0]].isna().sum() if COS_SIGNALS[0] in df else len(df)
        if missing:
            print(f"note: {missing}/{len(df)} rows have no cosine signal yet\n")
    else:
        print(f"note: {cos_path} not found — cosine signals skipped "
              f"(run scripts/compute_task_cos.py first)\n")
    return df


def overlap_at_k(tdf, signal, k):
    """|top-k by signal  ∩  top-k by accuracy| / k."""
    a = set(tdf.nlargest(k, signal)["combo"])
    b = set(tdf.nlargest(k, "best_acc")["combo"])
    return len(a & b) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="figures/target/target_summary.csv")
    ap.add_argument("--cos", default="figures/target/target_cos.csv")
    ap.add_argument("--method", default="base")
    ap.add_argument("--out", default="figures/analysis/cos_baseline.csv")
    ap.add_argument("--k", type=int, default=10, help="k for the top-k overlap")
    args = ap.parse_args()

    df = load(args.summary, args.cos, args.method)
    signals = [s for s in COS_SIGNALS + SAR_SIGNALS
               if s in df.columns and df[s].notna().any()]
    if not signals:
        raise SystemExit("no usable signal columns found")

    all_combo = "+".join(source_tasks())
    targets = list(dict.fromkeys(df["target"]))
    rows = []

    for target in targets:
        tdf = df[(df["target"] == target) & df["best_acc"].notna()]
        if tdf.empty:
            continue
        n = len(tdf)
        oracle = tdf.loc[tdf["best_acc"].idxmax()]

        def record(name, sel, **extra):
            # rank 1 = the accuracy-best mixture; ties resolved pessimistically
            rank = int((tdf["best_acc"] > sel["best_acc"]).sum()) + 1
            rows.append(dict(
                target=target, signal=name, n_combos=n,
                selected=sel["combo"], n_tasks=int(sel["n_tasks"]),
                acc=float(sel["best_acc"]), nai=float(sel["nai"]),
                best_coef=float(sel["best_coef"]),
                oracle_combo=oracle["combo"], oracle_acc=float(oracle["best_acc"]),
                regret=float(oracle["best_acc"]) - float(sel["best_acc"]),
                rank=rank, pct=rank / n, **extra))

        record("oracle", oracle, rho=np.nan, overlap_at_k=np.nan)

        # "random" has no single mixture behind it: report the grid mean as the
        # accuracy, with the mean-regret that implies.
        rows.append(dict(
            target=target, signal="random", n_combos=n,
            selected="(mean over all mixtures)", n_tasks=-1,
            acc=float(tdf["best_acc"].mean()), nai=float(tdf["nai"].mean()),
            best_coef=np.nan,
            oracle_combo=oracle["combo"], oracle_acc=float(oracle["best_acc"]),
            regret=float(oracle["best_acc"] - tdf["best_acc"].mean()),
            rank=int((tdf["best_acc"] > tdf["best_acc"].mean()).sum()) + 1,
            pct=((tdf["best_acc"] > tdf["best_acc"].mean()).sum() + 1) / n,
            rho=np.nan, overlap_at_k=np.nan))

        hit_all = tdf[tdf["combo"] == all_combo]
        if not hit_all.empty:
            record("all", hit_all.iloc[0], rho=np.nan, overlap_at_k=np.nan)

        for sig in signals:
            sdf = tdf[tdf[sig].notna()]
            if sdf.empty:
                continue
            sel = sdf.loc[sdf[sig].idxmax()]
            record(sig, sel,
                   rho=float(spearmanr(sdf[sig], sdf["best_acc"])[0]),
                   overlap_at_k=overlap_at_k(sdf, sig, min(args.k, len(sdf))))

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    # ---- per-target detail: what did each signal pick, and what did it cost? ----
    for target in targets:
        t = out[out["target"] == target]
        if t.empty:
            continue
        print(f"\n=== {display_name(target)} "
              f"({int(t['n_combos'].iloc[0])} mixtures) ===")
        print(f"{'signal':<9} {'acc':>7} {'regret':>8} {'rank':>7} {'pct':>7} "
              f"{'rho':>6} {f'ov@{args.k}':>6}  selected")
        for _, r in t.iterrows():
            rho = "  --  " if pd.isna(r["rho"]) else f"{r['rho']:+.3f}"
            ov = "  -- " if pd.isna(r["overlap_at_k"]) else f"{r['overlap_at_k']:.2f}"
            sel = (r["selected"] if r["selected"].startswith("(")
                   else display_combo(r["selected"]))
            print(f"{r['signal']:<9} {r['acc']:>7.4f} {r['regret']:>8.4f} "
                  f"{int(r['rank']):>7d} {r['pct']:>6.1%} {rho:>6} {ov:>6}  "
                  f"{sel[:60]}")

    # ---- aggregate: the summary line per signal ----
    print("\n\n=== aggregate over targets ===")
    print(f"{'signal':<9} {'mean acc':>9} {'mean reg':>9} {'med reg':>8} "
          f"{'med pct':>8} {'mean rho':>9} {f'mean ov@{args.k}':>11} {'wins':>5}")
    order = ["oracle"] + signals + ["all", "random"]
    for sig in order:
        s = out[out["signal"] == sig]
        if s.empty:
            continue
        wins = int((s["regret"] <= 0).sum())
        rho = s["rho"].mean()
        ov = s["overlap_at_k"].mean()
        print(f"{sig:<9} {s['acc'].mean():>9.4f} {s['regret'].mean():>9.4f} "
              f"{s['regret'].median():>8.4f} {s['pct'].median():>7.1%} "
              f"{'   --   ' if pd.isna(rho) else f'{rho:>+9.3f}'} "
              f"{'   --   ' if pd.isna(ov) else f'{ov:>11.2f}'} {wins:>5d}")

    print(f"\n  regret = oracle accuracy - selected accuracy (lower is better)")
    print(f"  pct    = selected mixture's percentile in the true ranking (lower is better)")
    print(f"  wins   = targets where the signal picked an accuracy-best mixture")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
