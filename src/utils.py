import torch
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from task_vector import TaskVector

from typing import Optional, Any

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


def calc_rank(S, norm_thresh=0.95):
    # Rank based on approximation error (Eq. 6) in the paper
    rank = np.argmax(np.sqrt(np.cumsum(S.pow(2) / S.pow(2).sum())) > norm_thresh)
    return rank


def alignment_ratio(S, S_proj):
    # Subspace alignment ratio based on norms of projected task matrix vs norm of the original one (Eq. 5) in the paper
    return np.linalg.norm(S_proj, ord=2) / np.linalg.norm(S, ord=2)


@torch.no_grad()
def sar():
    pass


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


def create_vector_combination(model, method, seed, tasks) -> dict[str, TaskVector]:
    pretrained_checkpoint = (
        f"saves_pretrained_weights/{method}/{model}/pretrained_weights_{seed}"
    )

    task_vectors = []
    for task in tasks:
        pattern = f"saves_bts_preliminary/{method}/{model}/train_{task}_{seed}_*"
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(
                f"No checkpoint found for task '{task}' matching: {pattern}"
            )

        finetuned_checkpoint = matches[0]
        task_vectors.append(
            create_task_vector(
                pretrained_checkpoint,
                finetuned_checkpoint,
                target_modules=METHODS_TO_TARGET_MODULES[method],
            )
        )

    return {"_".join(tasks): sum(task_vectors)}
