"""Check scripts/compute_task_cos.py's Gram shortcut against a direct computation.

compute_task_cos.py never forms a mixture vector: it claims

    cos(sum_{i in M} s_i, t) = (sum_i <s_i,t>)
                               / (sqrt(sum_{i,j in M} <s_i,s_j>) * ||t||)

and reads every term off a Gram matrix accumulated key-by-key in float64. If
that identity or the chunked accumulation were wrong, the baseline would still
produce plausible-looking numbers, so it is checked here against the obvious
implementation: materialize the deltas densely, add them up, take the cosine.

Restricted to a couple of weight keys so the dense side fits in memory — the
identity is per-inner-product and does not care how many keys are concatenated,
and --chunk-elems is set below the key size so the chunked path is exercised.

Usage:
    PYTHONPATH=src:scripts python src/tests/task_cos_gram.py
"""

import numpy as np
import torch
from safetensors import safe_open

import utils
from compute_task_cos import gram_over_checkpoints

MODEL, METHOD, SEED = "llama-3.2-1b-instruct", "base", 42
SOURCES = ["mnli", "qnli", "sst2"]
TARGET = "cb"
KEYS = ["model.layers.0.self_attn.q_proj.weight",
        "model.layers.5.mlp.gate_proj.weight"]


def dense_vector(task):
    """Flat float64 delta for `task` over KEYS, the obvious way."""
    pre = utils.pretrained_checkpoint(MODEL, METHOD, SEED)
    ft = utils.find_finetuned_checkpoint(MODEL, METHOD, SEED, task)
    parts = []
    with safe_open(f"{pre}/model.safetensors", framework="pt") as p, \
         safe_open(f"{ft}/model.safetensors", framework="pt") as f:
        for k in KEYS:
            parts.append((f.get_tensor(k) - p.get_tensor(k))
                         .reshape(-1).to(torch.float64))
    return torch.cat(parts).numpy()


def main():
    tasks = SOURCES + [TARGET]

    # Restrict the key filter to KEYS so both sides cover the same parameters.
    saved = utils.METHODS_TO_TARGET_MODULES[METHOD]
    utils.METHODS_TO_TARGET_MODULES[METHOD] = KEYS
    try:
        # chunk below the smallest key (4.2M elems) so chunking really happens
        G, n_params = gram_over_checkpoints(
            MODEL, METHOD, SEED, tasks,
            chunk_elems=1_000_000, device="cpu", verbose=False)
    finally:
        utils.METHODS_TO_TARGET_MODULES[METHOD] = saved

    dense = {t: dense_vector(t) for t in tasks}
    assert n_params == dense[TARGET].size, (n_params, dense[TARGET].size)

    # 1. every Gram entry
    worst = 0.0
    for i, a in enumerate(tasks):
        for j, b in enumerate(tasks):
            ref = float(dense[a] @ dense[b])
            rel = abs(G[i, j] - ref) / max(abs(ref), 1e-300)
            worst = max(worst, rel)
    print(f"Gram entries: worst relative error {worst:.3e}")
    assert worst < 1e-12, f"Gram mismatch (rel {worst:.3e})"

    # 2. the identity the script actually relies on, for every mixture
    from itertools import combinations
    t_vec = dense[TARGET]
    worst_cos = 0.0
    for r in range(2, len(SOURCES) + 1):
        for M in combinations(range(len(SOURCES)), r):
            mix = sum(dense[SOURCES[i]] for i in M)
            ref = float(mix @ t_vec / (np.linalg.norm(mix) * np.linalg.norm(t_vec)))

            k = len(tasks) - 1  # TARGET's index
            got = float(G[list(M), k].sum()
                        / (np.sqrt(G[np.ix_(M, M)].sum()) * np.sqrt(G[k, k])))
            worst_cos = max(worst_cos, abs(got - ref))
            print(f"  cos(sum{[SOURCES[i] for i in M]}, {TARGET}) "
                  f"gram={got:+.9f} dense={ref:+.9f}")
    print(f"cos identity: worst absolute error {worst_cos:.3e}")
    assert worst_cos < 1e-10, f"cos mismatch ({worst_cos:.3e})"

    print("\nOK: Gram shortcut matches the dense computation")


if __name__ == "__main__":
    main()
