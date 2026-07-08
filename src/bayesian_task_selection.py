"""Bayesian task-selection primitives shared by the scripts.

Search-space convention: a source subset is a {0,1}^n_sources vector over
SOURCE_TASKS (relaxed to [0,1]^n for the GP, rounded for evaluation).

Two families of routines:

  Online search (used by scripts/bo_task_selection.py) — optimize a live
  objective, typically `SarObjective` (utils.sar of the target's task vector
  against a source subset's summed subspace):
    exhaustive_search, run_bayesian_optimization, run_random_search,
    run_greedy_search

  Retrospective simulation on the precomputed transfer grid from
  scripts/summarize_target.py (used by scripts/analysis_correlations.py and
  scripts/analysis_bo_variants.py) — the GP predicts transfer accuracy and
  candidates come from the already-evaluated combos:
    simulate_bo_on_grid, simulate_random_on_grid, simulate_greedy_by
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import utils

SOURCE_TASKS = ["mnli", "qnli", "qqp", "sst2", "record"]
TARGET_TASKS = ["mrpc", "boolq", "rte", "cola"]


def combo_to_binary(combo_str: str, source_tasks: list[str] = SOURCE_TASKS):
    """"mnli+qqp" -> [1.0, 0.0, 1.0, 0.0, 0.0] over source_tasks."""
    tasks = combo_str.split("+")
    return [1.0 if t in tasks else 0.0 for t in source_tasks]


def fit_ei(train_X: torch.Tensor, train_Y: torch.Tensor) -> ExpectedImprovement:
    """Fit a SingleTaskGP on (X, Y) and return its Expected Improvement."""
    gp = SingleTaskGP(train_X, train_Y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return ExpectedImprovement(model=gp, best_f=train_Y.max())


class SarObjective:
    """SAR of the target's task vector against a source subset's summed
    subspace, cached per (subset, target) — the objective is deterministic, so
    repeated queries (BO revisits, greedy prefixes) are free."""

    def __init__(self, source_tasks, source_tvs, target_tvs,
                 rank_threshold=0.95, device="cpu", iso=False):
        self.source_tasks = source_tasks
        self.source_tvs = source_tvs      # {task: TaskVector}
        self.target_tvs = target_tvs      # {task: TaskVector}
        self.rank_threshold = rank_threshold
        self.device = device
        self.iso = iso
        self._cache = {}

    def __call__(self, subset_binary, target) -> float:
        selected = tuple(t for t, b in zip(self.source_tasks, subset_binary)
                         if b > 0.5)
        if not selected:
            return 0.0
        key = (selected, target)
        if key not in self._cache:
            self._cache[key] = utils.sar(
                [self.source_tvs[t] for t in selected], list(selected),
                rank_threshold=self.rank_threshold,
                device=self.device, iso=self.iso,
                probe_vectors=[self.target_tvs[target]], probe_tasks=[target],
            )[target]
        return self._cache[key]


# ============================================================
# Online search over a live objective
# ============================================================

def exhaustive_search(target, objective, source_tasks=SOURCE_TASKS):
    """Evaluate the objective for all non-empty subsets; sorted best-first."""
    n_sources = len(source_tasks)
    results = []
    for size in range(1, n_sources + 1):
        for subset in combinations(range(n_sources), size):
            binary = torch.zeros(n_sources)
            binary[list(subset)] = 1.0
            task_names = [source_tasks[i] for i in subset]
            results.append({
                "binary": binary,
                "tasks": task_names,
                "subset_str": "+".join(task_names),
                "sar": objective(binary, target),
            })
    results.sort(key=lambda r: r["sar"], reverse=True)
    return results


def run_bayesian_optimization(target, objective, source_tasks=SOURCE_TASKS,
                              n_initial=5, n_iterations=15, seed=42):
    """BO over {0,1}^n_sources, relaxed to [0,1]^n for the GP; candidates are
    rounded to binary for evaluation. Returns best subset + convergence curve."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_sources = len(source_tasks)

    all_X, all_Y = [], []

    # --- Phase 1: initial random evaluations ---
    for _ in range(n_initial):
        while True:
            x = torch.bernoulli(0.5 * torch.ones(n_sources))
            if x.sum() > 0:
                break
        all_X.append(x)
        all_Y.append(objective(x, target))

    # --- Phase 2: BO iterations ---
    for _ in range(n_iterations):
        train_X = torch.stack(all_X).double()
        train_Y = torch.tensor(all_Y).unsqueeze(-1).double()

        y_mean = train_Y.mean()
        y_std = train_Y.std().clamp(min=1e-6)
        ei = fit_ei(train_X, (train_Y - y_mean) / y_std)

        bounds = torch.stack([torch.zeros(n_sources, dtype=torch.double),
                              torch.ones(n_sources, dtype=torch.double)])
        candidate, _ = optimize_acqf(
            acq_function=ei, bounds=bounds, q=1,
            num_restarts=10, raw_samples=256,
        )

        x_new = (candidate.squeeze() > 0.5).float()
        if x_new.sum() == 0:
            x_new[candidate.squeeze().argmax()] = 1.0

        # already-evaluated subset: flip a random bit to keep exploring
        if any(torch.equal(x_new, x_prev) for x_prev in all_X):
            flip_idx = torch.randint(0, n_sources, (1,)).item()
            x_new[flip_idx] = 1.0 - x_new[flip_idx]
            if x_new.sum() == 0:
                x_new[flip_idx] = 1.0

        all_X.append(x_new)
        all_Y.append(objective(x_new, target))

    best_so_far = np.maximum.accumulate(all_Y).tolist()
    best_idx = int(np.argmax(all_Y))
    best_tasks = [source_tasks[i] for i in range(n_sources)
                  if all_X[best_idx][i] > 0.5]
    return {
        "best_sar": all_Y[best_idx],
        "best_tasks": best_tasks,
        "best_so_far": best_so_far,
        "n_evaluations": len(all_Y),
    }


def run_random_search(target, objective, n_sources, budget, seed=42):
    """Random search with the same evaluation budget as BO."""
    rng = np.random.default_rng(seed)
    all_Y = []
    for _ in range(budget):
        while True:
            x = torch.tensor(rng.integers(0, 2, n_sources), dtype=torch.float)
            if x.sum() > 0:
                break
        all_Y.append(objective(x, target))
    return np.maximum.accumulate(all_Y).tolist()


def run_greedy_search(target, objective, source_tasks=SOURCE_TASKS):
    """Greedy forward selection (DTVG-style): repeatedly add the task that
    increases the objective the most; stop when nothing improves."""
    n_sources = len(source_tasks)
    selected, remaining, history = [], list(range(n_sources)), []
    n_evals, current_best_sar = 0, 0.0

    while remaining:
        best_addition, best_sar = None, current_best_sar
        for idx in remaining:
            binary = torch.zeros(n_sources)
            binary[selected + [idx]] = 1.0
            sar = objective(binary, target)
            n_evals += 1
            if sar > best_sar:
                best_sar, best_addition = sar, idx
        if best_addition is None:
            break
        selected.append(best_addition)
        remaining.remove(best_addition)
        current_best_sar = best_sar
        history.append({
            "step": len(selected),
            "added": source_tasks[best_addition],
            "subset": [source_tasks[i] for i in selected],
            "sar": best_sar,
            "n_evals": n_evals,
        })

    return {
        "best_sar": current_best_sar,
        "best_tasks": [source_tasks[i] for i in selected],
        "history": history,
        "n_evals": n_evals,
    }


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

    for _ in range(n_iterations):
        if not available:
            break
        try:
            ei = fit_ei(all_X[evaluated], all_Y[evaluated])
            with torch.no_grad():
                ei_values = [ei(all_X[i].unsqueeze(0).unsqueeze(0)).item()
                             for i in available]
            next_idx = available[int(np.argmax(ei_values))]
        except Exception:
            if fallback_col is not None:
                next_idx = max(available,
                               key=lambda i: target_df.iloc[i][fallback_col])
            else:
                next_idx = available[np.random.randint(len(available))]

        evaluated.append(next_idx)
        available.remove(next_idx)
        best_so_far.append(max(accs[i] for i in evaluated))

    best_idx = max(evaluated, key=lambda i: accs[i])
    return {
        "best_so_far": best_so_far,
        "best_combo": target_df.iloc[best_idx]["combo"],
        "best_acc": float(accs[best_idx]),
    }


def simulate_random_on_grid(target_df, budget, n_trials=50, seed=42):
    """Random-order evaluation of the grid; one running-best curve per trial."""
    rng = np.random.default_rng(seed)
    accs = target_df["best_acc"].values
    curves = []
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
