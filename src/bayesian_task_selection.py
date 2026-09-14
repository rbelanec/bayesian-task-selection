"""Bayesian task-selection primitives shared by the scripts.

Search-space convention: a source subset is a {0,1}^n_sources vector over the
sweep's SOURCE_TASKS (read via combo_names, relaxed to [0,1]^n for the GP,
rounded for evaluation).

Retrospective simulation on the precomputed transfer grid from
scripts/summarize_target.py (used by scripts/analysis_correlations.py and
scripts/analysis_bo_variants.py) — the GP predicts transfer accuracy and
candidates come from the already-evaluated combos:
  simulate_bo_on_grid, simulate_random_on_grid, simulate_greedy_by

The online-search routines that optimized a live SAR objective
(SarObjective, exhaustive_search, run_bayesian_optimization, ...) were
removed together with their driver scripts/bo_task_selection.py after the
§11 proof-of-concept; see git history if they are ever needed again.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# combo_names reads the task lists straight out of src/eval/eval_target.py, so the
# indicator vector cannot go stale against the sweep. It lives in scripts/, which
# is sys.path[0] when the analyses are run as `python scripts/analysis_*.py`, but
# not when this module is imported with only PYTHONPATH=src.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import combo_names  # noqa: E402


def combo_to_binary(combo_str: str, source_tasks: list[str] | None = None):
    """"mnli+qqp" -> a {0,1} indicator over source_tasks (default: the sweep's).

    Hardcoding the vocabulary here is what made this silently wrong once the
    sources grew from 5 to 12: every combo built from a new task encoded as the
    all-zero vector, so unrelated mixtures collapsed onto identical GP inputs.
    An unknown task now raises instead.
    """
    vocab = combo_names.source_tasks() if source_tasks is None else source_tasks
    tasks = combo_str.split("+")
    unknown = [t for t in tasks if t not in vocab]
    if unknown:
        raise ValueError(
            f"combo {combo_str!r} has task(s) {unknown} outside the source "
            f"vocabulary {vocab} — the indicator would silently drop them")
    return [1.0 if t in tasks else 0.0 for t in vocab]


def fit_ei(train_X: torch.Tensor, train_Y: torch.Tensor) -> ExpectedImprovement:
    """Fit a SingleTaskGP on (X, Y) and return its Expected Improvement."""
    gp = SingleTaskGP(train_X, train_Y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return ExpectedImprovement(model=gp, best_f=train_Y.max())


# ============================================================
# Retrospective simulation on the precomputed transfer grid
# ============================================================

def simulate_bo_on_grid(target_df, feature_cols, n_initial=3, n_iterations=10,
                        seed=42, fallback_col=None):
    """Retrospective BO over the evaluated grid: GP input = feature_cols
    ("binary" expands to the subset indicator list column), GP output =
    transfer accuracy. If GP fitting fails, fall back to the unevaluated row
    maximizing fallback_col (random if None)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    accs = target_df["best_acc"].values
    features = []
    for _, row in target_df.iterrows():
        feat = []
        for col in feature_cols:
            if col == "binary":
                feat.extend(row["binary"])
            else:
                feat.append(row[col])
        features.append(feat)
    all_X = torch.tensor(features, dtype=torch.double)
    all_Y = torch.tensor(accs, dtype=torch.double).unsqueeze(-1)

    available = list(range(len(target_df)))
    np.random.shuffle(available)
    evaluated = available[:n_initial]
    available = available[n_initial:]

    best_so_far = [max(accs[i] for i in evaluated[:step + 1])
                   for step in range(n_initial)]

    n_fallback = 0
    for _ in range(n_iterations):
        if not available:
            break
        try:
            ei = fit_ei(all_X[evaluated], all_Y[evaluated]) # fits the expected improvement acquisition function
            with torch.no_grad():
                # One batched forward pass over every remaining candidate. EI at
                # q=1 is independent per design, so this is identical to scoring
                # them one at a time — but the per-candidate loop it replaces cost
                # a full acquisition call each, and at 4083 candidates x 60
                # iterations x 360 runs that dominated everything else.
                ei_values = ei(all_X[available].unsqueeze(1))
            next_idx = available[int(torch.argmax(ei_values))]
        except Exception:
            n_fallback += 1
            if fallback_col is not None:
                next_idx = max(available,
                               key=lambda i: target_df.iloc[i][fallback_col])
            else:
                next_idx = available[np.random.randint(len(available))]

        evaluated.append(next_idx)
        available.remove(next_idx)
        best_so_far.append(max(accs[i] for i in evaluated))

    # A silent fallback makes BO look exactly like its fallback strategy, which
    # would read as "BO does not beat random" rather than "the GP never fit".
    if n_fallback:
        warnings.warn(
            f"simulate_bo_on_grid: GP fitting failed on {n_fallback}/"
            f"{n_iterations} iterations; those picks came from "
            f"{fallback_col or 'random'} instead", RuntimeWarning, stacklevel=2)

    best_idx = max(evaluated, key=lambda i: accs[i])
    return {
        "best_so_far": best_so_far,
        "best_combo": target_df.iloc[best_idx]["combo"],
        "best_acc": float(accs[best_idx]),
    }


def simulate_random_on_grid(target_df, budget, n_trials=50, seed=42, seeds=None):
    """Random-order evaluation of the grid; one running-best curve per trial.

    Two modes:

    `seeds=None` (default) — `n_trials` independent draws from a `default_rng`
    stream. A precise estimate of random search in absolute terms, but it shares
    nothing with `simulate_bo_on_grid`.

    `seeds=<iterable>` — **paired**: trial i replays the exact permutation
    `simulate_bo_on_grid` draws for `seed=seeds[i]`, so the random walk *is* BO's
    initial design followed by more of the same order. Over the first
    `n_initial` steps the two are then identical by construction, not merely
    equal in expectation.

    That pairing matters because up to `n_initial` both methods are the same
    estimator — running max over uniform draws without replacement — so an
    unpaired baseline puts pure Monte-Carlo noise in the region of the plot
    where BO has not yet fit a GP, which reads as BO leading before it has done
    anything. Common random numbers remove that and tighten the comparison.
    """
    accs = target_df["best_acc"].values
    curves = []
    if seeds is not None:
        for s in seeds:
            # Mirror simulate_bo_on_grid's stream exactly: legacy global RNG,
            # shuffling a plain range list. Do not "modernise" this to
            # default_rng without changing it there too.
            np.random.seed(s)
            order = list(range(len(accs)))
            np.random.shuffle(order)
            curves.append(np.maximum.accumulate(accs[order[:budget]]))
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_trials):
            order = rng.permutation(len(accs))[:budget]
            curves.append(np.maximum.accumulate(accs[order]))
    return np.array(curves)


def simulate_greedy_by(target_df, col="sar"):
    """Evaluate grid rows in descending order of `col` (pure score-rank
    strategy). Returns (running-best accuracy curve, top-ranked combo)."""
    sorted_df = target_df.sort_values(col, ascending=False)
    return (np.maximum.accumulate(sorted_df["best_acc"].values),
            sorted_df.iloc[0]["combo"])
