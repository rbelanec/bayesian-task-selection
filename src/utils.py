import torch
import numpy as np
import glob
import json
import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from task_vector import TaskVector

from typing import Optional

logger = logging.getLogger(__name__)

METHODS_TO_TARGET_MODULES = {
    "lora": ["lora_A", "lora_B"],
    # Exclude embeddings (tied to lm_head) and layernorms from the base task vector:
    # the embedding delta is a task-agnostic shared drift that amplifies destructively
    # under a single global merge coefficient. See figures/amplification/README.md.
    "base": ["self_attn", "mlp"],
    # `freeze` = full FT with embeddings frozen (lever B). Same filter as base so the two
    # are an apples-to-apples ablation (differ only in training, not in what's merged).
    # Embeddings have an exactly-zero delta here, so `None` (merge everything trained) would
    # be equivalent up to the negligible per-block layernorm deltas this filter drops.
    "freeze": ["self_attn", "mlp"],
}


def nai(datasets, zsh, ft, merged):
    # caluclate normalized accuracy improvement (from Marczak et al.)
    return {ds: (merged[ds] - zsh[ds]) / (ft[ds] - zsh[ds]) for ds in datasets}


# scripts/compute_metrics.py stores these tasks' scores on `evaluate`'s 0-100
# scale while every other task is 0-1, so reading one back without the /100 puts
# it two orders of magnitude above the merged accuracy it is meant to be the
# ceiling for. compute_metrics.py applies the same correction when it builds
# results.tex; src/metrics.py normalizes at source for the merge sweep.
PERCENT_SCALE_TASKS = frozenset({"squad_v2"})


def _eval_exact_match(pattern: str, task: str | None = None) -> float:
    """`exact_match` from the first compute_metrics.jsonl matching `pattern`.

    Each line is a JSON object; we merge them (a task may log macro_f1/f1 on one
    line and exact_match on another). `exact_match` is the correct accuracy for
    every task, including ReCoRD — whose predict_results.json holds a wrong,
    naive per-row number instead (see src/tests/record_metric.py).

    `task` selects the scale correction: pass it for anything in
    PERCENT_SCALE_TASKS, otherwise the value is returned as stored.
    """
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no compute_metrics.jsonl matching: {pattern}")
    metrics: dict[str, float] = {}
    with open(hits[0]) as f:
        for line in f:
            if line.strip():
                metrics.update(json.loads(line))
    if "exact_match" not in metrics:
        raise KeyError(f"no 'exact_match' in {hits[0]} (have {list(metrics)})")
    score = float(metrics["exact_match"])
    return score / 100 if task in PERCENT_SCALE_TASKS else score


def zero_shot_acc(model: str, seed: int, task: str) -> float:
    # Pretrained-model accuracy (no fine-tuning); method-independent.
    return _eval_exact_match(
        f"saves_bts_preliminary/zero-shot/{model}/eval_{task}_{seed}_*/compute_metrics.jsonl",
        task,
    )


def finetuned_acc(method: str, model: str, seed: int, task: str) -> float:
    # Single-task fine-tuned accuracy (NAI's per-task ceiling).
    return _eval_exact_match(
        f"saves_bts_preliminary/{method}/{model}/eval_{task}_{seed}_*/compute_metrics.jsonl",
        task,
    )


def merged_acc_at_best_coef(
    method: str, model: str, seed: int, tasks: list[str]
) -> tuple[dict[str, float], float]:
    """Per-task merged accuracy at the shared coef maximizing mean accuracy.

    Matches the "best shared coef" selection in scripts/summarize_merged.py.
    """
    combo = "_".join(tasks)
    csv_path = (
        f"saves_bts_merged/{method}/{model}/{combo}_{seed}_best/{combo}_{seed}_acc_coef.csv"
    )
    df = pd.read_csv(csv_path, index_col="scaling_coef").sort_index()
    best_coef = float(df.mean(axis=1).idxmax())  # coef with best mean-over-tasks accuracy
    row = df.loc[best_coef]
    return {task: float(row[task]) for task in tasks}, best_coef


def compute_nai(
    model: str, method: str, seed: int, tasks: list[str]
) -> dict[str, float]:
    """Normalized Accuracy Improvement per task, loaded straight from the pipeline.

    Pulls merged accuracy (at the best shared coef), single-task fine-tuned
    accuracy, and zero-shot accuracy from saves_bts_*, then applies `nai`.
    """
    zsh = {task: zero_shot_acc(model, seed, task) for task in tasks}
    ft = {task: finetuned_acc(method, model, seed, task) for task in tasks}
    merged, _ = merged_acc_at_best_coef(method, model, seed, tasks)
    return nai(tasks, zsh, ft, merged)


def calc_rank(S, norm_thresh=0.95):
    # Rank based on approximation error (Eq. 6) in the paper
    rank = np.argmax(np.sqrt(np.cumsum(S.pow(2) / S.pow(2).sum())) > norm_thresh)
    return rank


def alignment_ratio(S, S_proj):
    # Subspace alignment ratio based on norms of projected task matrix vs norm of the original one (Eq. 5) in the paper
    #
    # NOTE: S/S_proj are 1-D singular-value arrays, and for a 1-D input
    # np.linalg.norm(..., ord=2) is the Euclidean norm — sqrt(sum of squares of
    # the singular values), which is exactly the Frobenius norm of the matrix
    # they came from. So this ratio never needed the singular values at all;
    # _sar_modes computes it straight from the matrices. Kept for callers that
    # already hold spectra (and as the reference the fast path is checked
    # against).
    return np.linalg.norm(S_proj, ord=2) / np.linalg.norm(S, ord=2)


@torch.no_grad()
def sar(
    task_vectors: list[TaskVector],
    tasks: list[str],
    rank_threshold: float = 0.95,
    device: str = "cpu",
    iso: bool = False,
    probe_vectors: list[TaskVector] | None = None,
    probe_tasks: list[str] | None = None,
) -> dict[str, float]:
    """Subspace Alignment Ratio (Marczak et al., iso-merging fig_3a.py).

    For each 2D weight matrix, build the merged subspace from the top-k left
    singular vectors of the summed task vector (k chosen by `calc_rank` at
    `rank_threshold`), project each individual task vector onto that subspace,
    and measure how much of its spectral norm survives via `alignment_ratio`
    (Eq. 5/6). Returns the per-task mean alignment ratio over all matrices.

    The reference gates on `key.startswith("model.visual")`; our task vectors
    are already filtered to self_attn/mlp at creation, so we keep every 2D
    matrix instead. Set `iso=True` to align against the isotropic spectrum
    (all singular values equal to their mean) rather than the raw sum spectrum.

    By default the projected vectors are `task_vectors` themselves (in-mixture
    SAR). Pass `probe_vectors`/`probe_tasks` to instead project *other* task
    vectors (e.g. held-out target tasks) onto the mixture subspace — the
    subspace is still built from `task_vectors` only.
    """
    return _sar_modes(
        task_vectors, tasks, rank_threshold, device,
        modes=("iso",) if iso else ("sum",),
        probe_vectors=probe_vectors, probe_tasks=probe_tasks,
    )["iso" if iso else "sum"]


@torch.no_grad()
def sar_both(
    task_vectors: list[TaskVector],
    tasks: list[str],
    rank_threshold: float = 0.95,
    device: str = "cpu",
    probe_vectors: list[TaskVector] | None = None,
    probe_tasks: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """`(sar, sar_iso)` in one pass — same values as two `sar()` calls.

    Both modes build their subspace from the *same* summed task vector, and
    `iso` only flattens the spectrum used to choose the rank; the left singular
    vectors are identical. Calling `sar()` twice therefore paid for that SVD
    (and every probe's device transfer) twice over.
    """
    out = _sar_modes(task_vectors, tasks, rank_threshold, device,
                     modes=("sum", "iso"),
                     probe_vectors=probe_vectors, probe_tasks=probe_tasks)
    return out["sum"], out["iso"]


@torch.no_grad()
def _sar_modes(task_vectors, tasks, rank_threshold, device, modes,
               probe_vectors=None, probe_tasks=None):
    """Shared engine for `sar` / `sar_both`; see `sar` for the semantics.

    The per-probe work here is one thin matmul and two Frobenius norms. The
    original formulation SVD'd both the probe and its projection only to reduce
    each spectrum to its Euclidean norm — see `alignment_ratio`. Dropping those
    two SVDs removes 24 of the 25 decompositions per (matrix, mixture) and
    leaves the one that actually defines the subspace.

    ||U_k U_k^T tv||_F == ||U_k^T tv||_F because U_k has orthonormal columns, so
    the projection is never formed at full size either.
    """
    assert len(task_vectors) == len(tasks), "need one task name per task vector"
    if probe_vectors is None:
        probe_vectors, probe_tasks = task_vectors, tasks
    assert len(probe_vectors) == len(probe_tasks), "need one name per probe vector"

    # SAR is only defined for 2D weight matrices (skip biases / norms / 1D).
    keys_2d = [k for k in task_vectors[0].vector if task_vectors[0].vector[k].dim() == 2]
    logger.info(
        "sar[%s]: %d matrices, subspace from %d tasks, %d probes on %s",
        "+".join(modes), len(keys_2d), len(tasks), len(probe_tasks), device,
    )
    t0 = time.perf_counter()

    ratios = {m: {task: [] for task in probe_tasks} for m in modes}
    for i, key in enumerate(keys_2d, 1):
        tk = time.perf_counter()
        merge_by_sum = sum(tv.vector[key].to(device) for tv in task_vectors)
        U, S, _ = torch.linalg.svd(merge_by_sum, full_matrices=False)

        # Moved once per matrix and reused by every mode, not once per (mode, probe).
        probes = [(task, ptv.vector[key].to(device))
                  for task, ptv in zip(probe_tasks, probe_vectors)]
        probe_fro = {task: torch.linalg.matrix_norm(pm, "fro")
                     for task, pm in probes}

        ranks = {}
        for mode in modes:
            S_mode = torch.ones_like(S) * S.mean() if mode == "iso" else S
            rel_rank = calc_rank(S_mode.cpu(), norm_thresh=rank_threshold)
            ranks[mode] = rel_rank
            U_k = U[:, :rel_rank]
            for task, pm in probes:
                num = torch.linalg.matrix_norm(U_k.T @ pm, "fro")
                ratios[mode][task].append(float(num / probe_fro[task]))

        logger.info(
            "sar[%s] %d/%d %s shape=%s rank=%s (%.1fs, total %.1fs)",
            "+".join(modes), i, len(keys_2d), key, tuple(merge_by_sum.shape),
            ",".join(f"{m}={ranks[m]}" for m in modes),
            time.perf_counter() - tk, time.perf_counter() - t0,
        )

    logger.info("sar[%s]: done in %.1fs", "+".join(modes), time.perf_counter() - t0)
    return {m: {task: float(np.mean(ar)) for task, ar in d.items() if ar}
            for m, d in ratios.items()}


def get_task_combinations(tasks: list[str]):
    from itertools import combinations

    return [c for r in range(2, len(tasks) + 1) for c in combinations(tasks, r)]


def create_task_vector(
    pretrained_checkpoint: str,
    finetuned_checkpoint: str,
    target_modules: Optional[list[str] | None] = None,
) -> TaskVector:
    task_vector = TaskVector(
        pretrained_checkpoint=pretrained_checkpoint,
        finetuned_checkpoint=finetuned_checkpoint,
        target_modules=target_modules,
    )

    return task_vector


def plot_acc_coef_csv(csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, index_col="scaling_coef").sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    for task in df.columns:
        ax.plot(df.index, df[task], label=task)
    ax.plot(df.index, df.mean(axis=1), linestyle="--", color="black", label="avg")
    ax.set_xlabel("scaling coefficient")
    ax.set_ylabel("accuracy")
    ax.set_title(csv_path.stem)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(csv_path.with_suffix(".png"), dpi=150)
    fig.savefig(csv_path.with_suffix(".pdf"))
    plt.close(fig)


#: A training run only counts as a usable checkpoint once one of these exists at
#: the top level of its output dir. A run that was killed (or OOM'd) mid-training
#: leaves behind the dir with just train.yaml, which loads fine as a *path* but
#: blows up inside from_pretrained several minutes later.
_WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")


def _has_weights(checkpoint: str) -> bool:
    return any(Path(checkpoint, f).exists() for f in _WEIGHT_FILES)


def pretrained_checkpoint(model, method, seed) -> str:
    """The merge origin: the pretrained weights every task vector is measured from."""
    return f"saves_pretrained_weights/{method}/{model}/pretrained_weights_{seed}"


def find_finetuned_checkpoint(model, method, seed, task) -> str:
    """The single-task fine-tune to subtract the pretrained weights from.

    Split out of `build_task_vectors` so consumers that need the checkpoint
    *path* rather than a materialized TaskVector — scripts/compute_task_cos.py
    reads the tensors key-by-key, because 24 full task vectors do not fit in
    RAM — resolve it by exactly the same rules, instead of reimplementing the
    "several runs, some without weights" handling below and drifting from it.
    """
    pattern = f"saves_bts_preliminary/{method}/{model}/train_{task}_{seed}_*"
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for task '{task}' matching: {pattern}"
        )

    # A task can have several runs (e.g. a failed one plus its rerun). glob
    # returns them in filesystem order, so skip the ones that never saved
    # weights and take the newest of what's left.
    usable = sorted(filter(_has_weights, matches), key=lambda p: Path(p).stat().st_mtime)
    if not usable:
        raise FileNotFoundError(
            f"No checkpoint with saved weights for task '{task}'; "
            f"{len(matches)} dir(s) matched {pattern} but none contain any of {_WEIGHT_FILES}: "
            f"{sorted(matches)}"
        )
    if len(usable) < len(matches):
        logger.warning(
            "Task '%s': ignoring %d checkpoint dir(s) without saved weights (%s)",
            task, len(matches) - len(usable), sorted(set(matches) - set(usable)),
        )

    return usable[-1]


def build_task_vectors(model, method, seed, tasks) -> list[TaskVector]:
    """Per-task TaskVectors for a combo (individual, not summed)."""
    return [
        create_task_vector(
            pretrained_checkpoint(model, method, seed),
            find_finetuned_checkpoint(model, method, seed, task),
            target_modules=METHODS_TO_TARGET_MODULES[method],
        )
        for task in tasks
    ]


def create_vector_combination(model, method, seed, tasks) -> dict[str, TaskVector]:
    return {"_".join(tasks): sum(build_task_vectors(model, method, seed, tasks))}
