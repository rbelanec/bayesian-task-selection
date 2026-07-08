"""Bayesian task selection: BO over source-task subsets with target SAR
as the objective.

For each target task, find the source-task subset whose summed task-vector
subspace best aligns with the target's own task vector (utils.sar with probe
vectors — the same objective as scripts/compute_target_sar.py). With 5 source
tasks (31 subsets) exhaustive search is cheap, so BO is validated against the
true optimum plus greedy forward selection (DTVG-style) and random search —
the proof-of-concept before scaling to 15+ tasks, where exhaustive search is
no longer an option. The search/objective primitives live in
src/bayesian_task_selection.py.

If the ground-truth transfer ranking exists (figures/target/target_summary.csv
from scripts/summarize_target.py), each method's chosen subset is also scored
by its actual merged transfer accuracy, so "BO finds the max-SAR subset" and
"the max-SAR subset transfers well" stay separate questions.

Writes per-target convergence and SAR-landscape plots plus bo_summary.csv to
--out (default figures/bo). Run via SLURM (scripts/slurm/bo_selection.sh),
not on the login node.

Usage:
    PYTHONPATH=src python scripts/bo_task_selection.py \
        --model llama-3.2-1b-instruct --seed 42 --method lora --iso
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np

import utils
from bayesian_task_selection import (SOURCE_TASKS, TARGET_TASKS, SarObjective,
                                     exhaustive_search,
                                     run_bayesian_optimization,
                                     run_greedy_search, run_random_search)

# dataviz palette
C_BO, C_GREEDY, C_NEUTRAL = "#4a3aa7", "#e34948", "#888780"


# ============================================================
# VISUALIZATION
# ============================================================

def plot_convergence(bo_result, random_curves, exhaustive_best, greedy_result,
                     target, sar_name, out_dir):
    """Best SAR found vs number of evaluations for BO, random (mean over
    trials) and greedy, against the exhaustive optimum."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    n_steps = len(bo_result["best_so_far"])
    steps = np.arange(1, n_steps + 1)

    ax.axhline(y=exhaustive_best, color=C_NEUTRAL, linestyle="--", linewidth=1,
               label=f"Exhaustive optimum ({exhaustive_best:.4f})")

    random_mean = np.array(random_curves).mean(axis=0)
    ax.plot(steps, random_mean, color=C_NEUTRAL, linewidth=1.5,
            label="Random search (mean)")

    if greedy_result["history"]:
        greedy_evals = [h["n_evals"] for h in greedy_result["history"]]
        greedy_sars = [h["sar"] for h in greedy_result["history"]]
        ax.step([0] + greedy_evals + [n_steps],
                [0] + greedy_sars + [greedy_sars[-1]], where="post",
                color=C_GREEDY, linewidth=1.5, label="Greedy (DTVG-style)")

    ax.plot(steps, bo_result["best_so_far"], color=C_BO, linewidth=2,
            label="Bayesian optimization")

    for i, val in enumerate(bo_result["best_so_far"]):
        if abs(val - exhaustive_best) < 1e-6:
            ax.scatter([i + 1], [val], color=C_BO, s=60, zorder=5,
                       edgecolors="white", linewidth=1.5)
            ax.annotate(f"Found optimum\nat eval {i + 1}", xy=(i + 1, val),
                        xytext=(i + 3, val - 0.02), fontsize=9, color=C_BO,
                        arrowprops=dict(arrowstyle="->", color=C_BO))
            break

    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel(f"Best {sar_name} found")
    ax.set_title(f"Task selection convergence — target: {target}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(1, n_steps)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/convergence_{target}.png", dpi=150)
    fig.savefig(f"{out_dir}/convergence_{target}.pdf")
    plt.close(fig)


def plot_subset_landscape(exhaustive_results, target, sar_name, out_dir):
    """Sorted bar chart of SAR for all subsets — the landscape BO searches."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    sars = [r["sar"] for r in exhaustive_results]
    labels = [r["subset_str"] for r in exhaustive_results]
    colors = [C_BO if i == 0 else "#D3D1C7" for i in range(len(sars))]
    ax.barh(range(len(sars)), sars, color=colors, height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(sar_name)
    ax.set_title(f"{sar_name} landscape, all {len(sars)} subsets — target: {target}")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(f"{out_dir}/landscape_{target}.png", dpi=150)
    fig.savefig(f"{out_dir}/landscape_{target}.pdf")
    plt.close(fig)


# ============================================================
# GROUND-TRUTH TRANSFER ACCURACY (optional join)
# ============================================================

def load_transfer_acc(path, method):
    """{(combo, target): best_acc} from scripts/summarize_target.py's CSV."""
    if not os.path.exists(path):
        print(f"note: {path} not found — transfer-accuracy columns will be "
              "empty (run scripts/summarize_target.py first)\n")
        return {}
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] == method:
                out[(r["combo"], r["target"])] = float(r["best_acc"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--method", default="lora")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iso", action="store_true",
                    help="use iso-SAR (equal singular values) as the objective")
    ap.add_argument("--device", default="cpu", help="device for SAR SVDs")
    ap.add_argument("--rank-threshold", type=float, default=0.95)
    ap.add_argument("--n-initial", type=int, default=5,
                    help="random evaluations before BO starts")
    ap.add_argument("--n-bo-iterations", type=int, default=15)
    ap.add_argument("--n-random-trials", type=int, default=50)
    ap.add_argument("--acc", default="figures/target/target_summary.csv",
                    help="ground-truth transfer ranking to score subsets against")
    ap.add_argument("--out", default="figures/bo")
    ap.add_argument("--verbose", action="store_true", help="per-matrix SAR logs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.out, exist_ok=True)

    n_sources = len(SOURCE_TASKS)
    sar_name = "SAR (iso)" if args.iso else "SAR"
    budget = args.n_initial + args.n_bo_iterations

    print(f"loading task vectors ({args.method}/{args.model}, seed {args.seed})...")
    source_tvs = dict(zip(SOURCE_TASKS, utils.build_task_vectors(
        args.model, args.method, args.seed, SOURCE_TASKS)))
    target_tvs = dict(zip(TARGET_TASKS, utils.build_task_vectors(
        args.model, args.method, args.seed, TARGET_TASKS)))
    objective = SarObjective(SOURCE_TASKS, source_tvs, target_tvs,
                             rank_threshold=args.rank_threshold,
                             device=args.device, iso=args.iso)
    acc = load_transfer_acc(args.acc, args.method)

    summary = []
    for target in TARGET_TASKS:
        print(f"\n{'=' * 60}\nTARGET: {target}\n{'=' * 60}")

        print(f"\n  exhaustive search ({2 ** n_sources - 1} subsets)...")
        exhaustive = exhaustive_search(target, objective)
        exhaustive_best = exhaustive[0]["sar"]
        print(f"  best subset: {exhaustive[0]['subset_str']} "
              f"({sar_name}={exhaustive_best:.4f})")

        print("\n  greedy search...")
        greedy = run_greedy_search(target, objective)
        print(f"  greedy: {'+'.join(greedy['best_tasks'])} "
              f"({sar_name}={greedy['best_sar']:.4f}, evals={greedy['n_evals']})")

        print("\n  Bayesian optimization...")
        bo = run_bayesian_optimization(target, objective,
                                       n_initial=args.n_initial,
                                       n_iterations=args.n_bo_iterations,
                                       seed=args.seed)
        print(f"  BO: {'+'.join(bo['best_tasks'])} "
              f"({sar_name}={bo['best_sar']:.4f}, evals={bo['n_evaluations']})")

        print(f"\n  {args.n_random_trials} random search trials...")
        random_curves = [
            run_random_search(target, objective, n_sources, budget,
                              seed=args.seed + trial)
            for trial in range(args.n_random_trials)
        ]
        random_best = float(np.mean([c[-1] for c in random_curves]))
        print(f"  random avg best: {sar_name}={random_best:.4f}")

        plot_convergence(bo, random_curves, exhaustive_best, greedy, target,
                         sar_name, args.out)
        plot_subset_landscape(exhaustive, target, sar_name, args.out)

        found_optimum = abs(bo["best_sar"] - exhaustive_best) < 1e-6
        evals_to_optimum = next(
            (i + 1 for i, v in enumerate(bo["best_so_far"])
             if abs(v - exhaustive_best) < 1e-6), None)

        # score the chosen subsets by actual transfer accuracy (if available)
        bo_combo = "+".join(bo["best_tasks"])
        sar_combo = exhaustive[0]["subset_str"]
        evaluated = {c for c, t in acc if t == target}
        acc_best_combo = (max(evaluated, key=lambda c: acc[(c, target)])
                          if evaluated else None)

        summary.append(dict(
            target=target, objective=sar_name,
            exhaustive_best=exhaustive_best, exhaustive_subset=sar_combo,
            bo_best=bo["best_sar"], bo_subset=bo_combo,
            bo_found_optimum=found_optimum, bo_evals_to_optimum=evals_to_optimum,
            greedy_best=greedy["best_sar"],
            greedy_subset="+".join(greedy["best_tasks"]),
            greedy_evals=greedy["n_evals"], random_avg_best=random_best,
            bo_subset_acc=acc.get((bo_combo, target)),
            sar_opt_subset_acc=acc.get((sar_combo, target)),
            acc_best_subset=acc_best_combo,
            acc_best=acc.get((acc_best_combo, target)) if acc_best_combo else None,
        ))

    csv_path = f"{args.out}/bo_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {csv_path}")

    fmt = lambda v, p=4: "--" if v is None else f"{v:.{p}f}"
    print(f"\n{'=' * 80}\nSUMMARY ({sar_name} objective)\n{'=' * 80}")
    print(f"{'target':<8} {'exhaustive':>11} {'BO':>9} {'greedy':>9} "
          f"{'random':>9} {'BO evals':>9}")
    for s in summary:
        print(f"{s['target']:<8} {s['exhaustive_best']:>11.4f} "
              f"{s['bo_best']:>9.4f} {s['greedy_best']:>9.4f} "
              f"{s['random_avg_best']:>9.4f} {str(s['bo_evals_to_optimum']):>9}")

    print("\nchosen subsets (and their actual transfer accuracy):")
    for s in summary:
        hit = "==" if s["bo_found_optimum"] else "!="
        print(f"  {s['target']:6s}: SAR-opt {s['exhaustive_subset']:28s} "
              f"acc={fmt(s['sar_opt_subset_acc'], 3)} | BO {hit} "
              f"{s['bo_subset']:28s} acc={fmt(s['bo_subset_acc'], 3)}")
        if s["acc_best_subset"]:
            print(f"  {'':6s}  acc-opt {s['acc_best_subset']:28s} "
                  f"acc={fmt(s['acc_best'], 3)}")


if __name__ == "__main__":
    main()
