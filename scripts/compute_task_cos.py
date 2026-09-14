#!/usr/bin/env python3
"""Cosine-similarity task selection: rank source mixtures by weight-space geometry.

A cheap baseline for the question figures/target/target_summary.csv answers by
brute force ("which source mixture transfers best to target t?"), in the spirit
of DTVG (Zhang et al., Findings of ACL 2025, https://aclanthology.org/2025.findings-acl.1374/):
score every mixture against the target's own task vector and pick the argmax,
with no merged-model evaluation anywhere in the loop.

Signals written per (combo, target) — all four are pure geometry, so none of
them needs the target sweep that produced the ground truth they are compared
against:

Two independent choices are crossed, so cosine and DTVG's own metric can be
compared like for like rather than one standing in for the other:

  * the similarity   -- cosine (SPoT's metric, and the scale-invariant one) vs
                        the dot product (DTVG Eq. 2, which they argue beats
                        SPoT's cosine between soft prompts). Their 1/r^2
                        token-length factor is a fixed scale here and is
                        dropped: it cannot change a ranking.
  * what is compared -- the summed mixture vector (`_mix`, the object
                        eval_target.py actually merges, since
                        utils.create_vector_combination sums the members) vs
                        the mean over members of the per-source score (`ts_`,
                        DTVG's Target Similarity aggregation).

  cos_mix   cos(sum of the mixture's source task vectors, target task vector)
  dot_mix   <sum of the mixture's source task vectors, target task vector>
  ts_cos    mean over members of cos(source, target)
  ts_dot    mean over members of <source, target>   -- DTVG Target Similarity

Those four are genuinely different orderings, not reparametrizations:

  * cos_mix vs ts_cos differ whenever members are not mutually orthogonal. Cos
    of the sum rewards members whose deltas reinforce *along the target
    direction* and penalizes a mixture whose members inflate ||sum|| by
    pulling elsewhere; the mean of cos cannot see either effect.
  * dot_mix = |M| * ts_dot, i.e. they agree on any fixed mixture size and
    disagree across sizes -- dot_mix rewards adding members almost
    unconditionally (an unrelated source with a positive projection still
    raises the sum), while ts_dot has to beat the running average. With
    mixtures of size 2..12 in one ranking, that is the whole ballgame.
  * cosine normalizes by ||sum||, so it is the only one of the four that
    charges a mixture for growing. It is also *blind to the merge
    coefficient*: one number per (combo, target), against a ground truth that
    is itself maximized over the coefficient grid.

DTVG's second grouping metric, in both flavors -- target-independent, a
conflict measure among the members rather than a relevance measure against the
target. Constant across targets for a given combo; carried on every row so it
joins like n_tasks:

  kc_cos    DTVG "Knowledge Consistency": mean pairwise cos among members
  kc_dot    the same with DTVG's own metric: mean pairwise <s_i, s_j>

Why a Gram matrix instead of 4083 merges
----------------------------------------
    cos(sum_{i in M} s_i, t) = (sum_{i in M} <s_i,t>)
                               / (sqrt(sum_{i,j in M} <s_i,s_j>) * ||t||)

Every quantity on the right is an entry of the Gram matrix of the 12 source and
12 target task vectors. So one pass over the checkpoints yields a 24x24 matrix,
and all 4083 mixtures x 12 targets follow in closed form -- no mixture vector is
ever formed, and the cost is independent of how many mixtures are scored.

Why key-by-key rather than utils.build_task_vectors
---------------------------------------------------
These task vectors are the `base` method's self_attn+mlp delta: 973M fp32
params, 3.89 GB each. The 24 needed here would be ~93 GB resident, so the
tensors are streamed one key at a time straight out of the safetensors files
(24 slices of one key at a time instead), and the Gram matrix is accumulated in
float64. Checkpoint resolution still goes through utils, so "which fine-tune is
task t's?" cannot drift from the merge sweep's answer.

Usage:
    PYTHONPATH=src python scripts/compute_task_cos.py \
        --model llama-3.2-1b-instruct --seed 42 \
        --out figures/target/target_cos.csv

The Gram matrix is cached to --gram; a rerun with the same task lists reuses it
and re-emits the CSV in about a second (the ~118 GB of checkpoint reads is the
entire cost of this script).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from itertools import combinations

import numpy as np
import torch
from safetensors import safe_open

import utils
from combo_names import source_tasks, target_tasks


def gram_over_checkpoints(model, method, seed, tasks, chunk_elems, device, verbose=True):
    """Gram matrix <tv_i, tv_j> of `tasks`' task vectors, streamed key by key.

    Returns (G, n_params) with G float64 (len(tasks) x len(tasks)) computed over
    the concatenation of every key `METHODS_TO_TARGET_MODULES[method]` keeps —
    i.e. the same parameters the merge touches, so the geometry matches what the
    sweep evaluated.
    """
    pre_path = utils.pretrained_checkpoint(model, method, seed)
    ft_paths = [utils.find_finetuned_checkpoint(model, method, seed, t) for t in tasks]
    target_modules = utils.METHODS_TO_TARGET_MODULES[method]

    n = len(tasks)
    G = np.zeros((n, n), dtype=np.float64)

    def st(path):
        f = os.path.join(path, "model.safetensors")
        if not os.path.exists(f):
            raise FileNotFoundError(f"expected an unsharded {f}")
        return f

    pre_f = safe_open(st(pre_path), framework="pt")
    ft_f = [safe_open(st(p), framework="pt") for p in ft_paths]

    keys = [k for k in pre_f.keys()
            if any(m in k for m in target_modules)]
    keys.sort()
    if not keys:
        raise SystemExit(f"no keys matched {target_modules}")

    n_params = 0
    t0 = time.perf_counter()
    for ki, key in enumerate(keys, 1):
        base = pre_f.get_tensor(key).reshape(-1)
        # 24 fp32 deltas of one key: 1.6 GB at the widest (2048x8192), versus
        # 93 GB for the same 24 vectors held whole.
        A = torch.empty((n, base.numel()), dtype=torch.float32)
        for i, f in enumerate(ft_f):
            A[i] = f.get_tensor(key).reshape(-1) - base
        n_params += base.numel()
        del base

        # float64 accumulation: these are inner products over ~1e9 terms, where
        # float32 summation drifts enough to reorder near-tied mixtures.
        for c0 in range(0, A.shape[1], chunk_elems):
            Ac = A[:, c0:c0 + chunk_elems].to(device=device, dtype=torch.float64)
            G += (Ac @ Ac.T).cpu().numpy()
            del Ac
        del A

        if verbose:
            print(f"[gram {ki}/{len(keys)}] {key} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    for f in ft_f:
        f.__exit__(None, None, None)
    pre_f.__exit__(None, None, None)
    return G, n_params


def cos_from_gram(G, idx_a, idx_b):
    """cos between task vectors, from Gram entries."""
    return G[idx_a, idx_b] / np.sqrt(G[idx_a, idx_a] * G[idx_b, idx_b])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", default="base",
                    help="which task-vector flavor to measure (its entry in "
                         "utils.METHODS_TO_TARGET_MODULES picks the keys)")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="default: SOURCE_TASKS from src/eval/eval_target.py")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="default: TARGET_TASKS from src/eval/eval_target.py")
    ap.add_argument("--device", default="cpu",
                    help="device for the Gram matmuls (cuda is fine but this "
                         "script is I/O bound, not compute bound)")
    ap.add_argument("--chunk-elems", type=int, default=4_000_000,
                    help="elements per float64 matmul chunk (memory knob)")
    ap.add_argument("--out", default="figures/target/target_cos.csv")
    ap.add_argument("--gram", default=None,
                    help="cache path for the Gram matrix "
                         "(default: --out with _gram.npz)")
    ap.add_argument("--recompute", action="store_true",
                    help="ignore any cached Gram matrix and re-read checkpoints")
    args = ap.parse_args()

    sources = args.sources or source_tasks()
    targets = args.targets or target_tasks()
    tasks = list(sources) + list(targets)
    if len(set(tasks)) != len(tasks):
        raise SystemExit("a task appears in both --sources and --targets")

    gram_path = args.gram or (os.path.splitext(args.out)[0] + "_gram.npz")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    G = None
    if os.path.exists(gram_path) and not args.recompute:
        z = np.load(gram_path, allow_pickle=False)
        cached = list(z["tasks"])
        if cached == tasks and str(z["method"]) == args.method:
            G = z["gram"]
            print(f"[cache] reusing Gram matrix from {gram_path} "
                  f"({z['n_params']} params)")
        else:
            print(f"[cache] {gram_path} is for different tasks/method — recomputing")

    if G is None:
        print(f"[gram] streaming {len(tasks)} task vectors "
              f"({args.method}) key by key; this reads "
              f"~{4.94 * (len(tasks) + 1):.0f} GB")
        G, n_params = gram_over_checkpoints(
            args.model, args.method, args.seed, tasks,
            args.chunk_elems, args.device)
        np.savez(gram_path, gram=G, tasks=np.array(tasks), n_params=n_params,
                 method=args.method, model=args.model, seed=args.seed)
        print(f"[gram] wrote {gram_path}")

    ns = len(sources)
    si = {t: i for i, t in enumerate(sources)}
    ti = {t: ns + i for i, t in enumerate(targets)}

    # Degenerate task vector = a fine-tune that never moved: cosine is 0/0.
    for t, i in list(si.items()) + list(ti.items()):
        if G[i, i] <= 0:
            raise SystemExit(f"task {t!r} has a zero-norm task vector "
                             f"(<tv,tv> = {G[i, i]}) — check its checkpoint")

    # Pairwise source cosines, for kc_cos.
    src_cos = {(a, b): cos_from_gram(G, si[a], si[b])
               for a in sources for b in sources}
    st_cos = {(a, k): cos_from_gram(G, si[a], ti[k])
              for a in sources for k in targets}

    combos = [c for r in range(2, len(sources) + 1)
              for c in combinations(sources, r)]
    print(f"[emit] {len(combos)} combinations x {len(targets)} targets")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "combo", "n_tasks", "target",
            "cos_mix", "dot_mix", "ts_cos", "ts_dot", "kc_cos", "kc_dot"])
        w.writeheader()
        for tasks_m in combos:
            m = [si[t] for t in tasks_m]
            # ||sum_i s_i||^2 = sum_{i,j} <s_i,s_j>
            mix_sq = G[np.ix_(m, m)].sum()
            mix_norm = np.sqrt(mix_sq)
            pairs = list(combinations(tasks_m, 2))
            kc_cos = float(np.mean([src_cos[p] for p in pairs])) if pairs else float("nan")
            kc_dot = (float(np.mean([G[si[a], si[b]] for a, b in pairs]))
                      if pairs else float("nan"))
            combo = "+".join(tasks_m)
            for target in targets:
                k = ti[target]
                dots = G[m, k]
                w.writerow(dict(
                    method=args.method, combo=combo, n_tasks=len(tasks_m),
                    target=target,
                    cos_mix=float(dots.sum() / (mix_norm * np.sqrt(G[k, k]))),
                    dot_mix=float(dots.sum()),
                    ts_cos=float(np.mean([st_cos[(t, target)] for t in tasks_m])),
                    ts_dot=float(dots.mean()),
                    kc_cos=kc_cos,
                    kc_dot=kc_dot,
                ))

    print(f"wrote {args.out} ({len(combos) * len(targets)} rows)")


if __name__ == "__main__":
    main()
