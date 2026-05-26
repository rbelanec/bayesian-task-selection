"""Visualize and compare task-vector "amplification" for full FT (base / freeze) vs LoRA.

Amplification of a set of task vectors S for a parameter group g is

    A_g(S) = || sum_{t in S} tau_t^g ||  /  sqrt( sum_{t in S} || tau_t^g ||^2 )

It measures how *constructively* per-task updates add up under task arithmetic:
  * A == 1                -> task vectors are mutually orthogonal (clean addition)
  * A == sqrt(|S|)        -> task vectors are perfectly aligned (one shared direction)
A single global scaling coefficient is tuned for the bulk of the merged delta, so a
group whose amplification is much higher than the others gets systematically
over-driven. For full fine-tuning that group is the (tied) embeddings; LoRA never
touches them.

Methods compared (all reduce to weight-space deltas of the same shape per tensor):
  * base   : full FT.                tau_t = theta_finetuned(t) - theta_pretrained
  * freeze : full FT, embeddings frozen (lever B). Same as base but embed/lm_head/final
             norm have an *exactly-zero* delta, so their amplification is undefined
             (0/0) and is reported as "frozen" rather than plotted.
  * lora   : tau_t = (alpha / r) * B_t @ A_t  (effective dW per adapted module).

All amplifications are derived from per-group Gram matrices G[i, j] = <tau_i, tau_j>
(Frobenius inner product), so any task subset is cheap to evaluate without re-reading
the checkpoints.

Usage:
    PYTHONPATH=src python scripts/visualize_amplification.py \
        --tasks mnli qnli qqp sst2 record \
        --methods base freeze \
        --model llama-3.2-1b-instruct --seed 42 \
        --out-dir figures/amplification
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open

# Parameter groups for a *full-model* (base / freeze) task vector. A key may belong to
# several groups (e.g. k_proj is in both "attn" and the "attn(k/v)" subset that lines
# up with the LoRA target modules).
BASE_GROUPS = ["attn(k/v)", "attn", "mlp", "embed", "norm"]
CHUNK_ROWS = 8192  # row-chunk size for streaming large 2D tensors (keeps RAM bounded)

# Stable colors per parameter group + how each method is distinguished from the others
# (full-FT methods share group colors; linestyle / hatch separate base from freeze).
GROUP_COLOR = {
    "embed": "#e6550d", "attn(k/v)": "#3182bd", "attn": "#6baed6",
    "mlp": "#08519c", "norm": "#a1d99b",
}
GROUP_MARKER = {"embed": "o", "attn(k/v)": "s", "attn": "s", "mlp": "^", "norm": "v"}
METHOD_LS = {"base": "-", "freeze": "-.", "lora": "-"}
METHOD_HATCH = {"base": "", "freeze": "//", "lora": ""}
LORA_COLOR = "#d62728"


def series_style(method: str, group: str):
    """(color, linestyle, marker) for one (method, group) series."""
    if method == "lora":
        return LORA_COLOR, "-", "D"
    return GROUP_COLOR[group], METHOD_LS.get(method, "-"), GROUP_MARKER[group]


def base_groups_of(key: str) -> list[str]:
    groups: list[str] = []
    if "embed_tokens" in key:
        groups.append("embed")
    elif "layernorm" in key or key.endswith("model.norm.weight"):
        groups.append("norm")
    elif "self_attn" in key:
        groups.append("attn")
        if "k_proj" in key or "v_proj" in key:
            groups.append("attn(k/v)")
    elif "mlp" in key:
        groups.append("mlp")
    return groups


def _iter_chunks(numel_dim0: int):
    for start in range(0, numel_dim0, CHUNK_ROWS):
        yield start, min(start + CHUNK_ROWS, numel_dim0)


def base_gram(pre_path: str, ft_paths: list[str]) -> dict[str, np.ndarray]:
    """Per-group Gram matrices of full-model task vectors, streamed chunk-by-chunk."""
    T = len(ft_paths)
    fpre = safe_open(pre_path, framework="pt")
    ffts = [safe_open(p, framework="pt") for p in ft_paths]
    gram = {g: np.zeros((T, T), dtype=np.float64) for g in BASE_GROUPS}

    for key in fpre.keys():
        groups = base_groups_of(key)
        if not groups:
            continue
        pre_slice = fpre.get_slice(key)
        ft_slices = [f.get_slice(key) for f in ffts]
        n0 = pre_slice.get_shape()[0]
        for a, b in _iter_chunks(n0):
            base_chunk = pre_slice[a:b].float()
            deltas = [s[a:b].float() - base_chunk for s in ft_slices]
            flat = [d.reshape(-1).double() for d in deltas]
            for i in range(T):
                for j in range(i, T):
                    v = torch.dot(flat[i], flat[j]).item()
                    for g in groups:
                        gram[g][i, j] += v
                        if i != j:
                            gram[g][j, i] += v
    return gram


def lora_gram(ft_dirs: list[str]) -> dict[str, np.ndarray]:
    """Gram matrix of LoRA effective weight deltas dW = (alpha/r) * B @ A, summed
    over all adapted modules. Returned under the single group 'attn(k/v)'."""
    T = len(ft_dirs)
    G = np.zeros((T, T), dtype=np.float64)

    # per-task: module -> dW
    dW_per_task: list[dict[str, torch.Tensor]] = []
    for d in ft_dirs:
        cfg = json.load(open(os.path.join(d, "adapter_config.json")))
        scaling = cfg["lora_alpha"] / cfg["r"]
        f = safe_open(os.path.join(d, "adapter_model.safetensors"), framework="pt")
        a_keys = [k for k in f.keys() if k.endswith("lora_A.weight")]
        dW: dict[str, torch.Tensor] = {}
        for ak in a_keys:
            module = ak[: -len(".lora_A.weight")]
            A = f.get_tensor(ak).float()                      # [r, in]
            B = f.get_tensor(module + ".lora_B.weight").float()  # [out, r]
            dW[module] = scaling * (B @ A)                     # [out, in]
        dW_per_task.append(dW)

    modules = sorted(dW_per_task[0].keys())
    for i in range(T):
        for j in range(i, T):
            v = sum(
                torch.dot(dW_per_task[i][m].reshape(-1).double(),
                          dW_per_task[j][m].reshape(-1).double()).item()
                for m in modules
            )
            G[i, j] = v
            G[j, i] = v
    return {"attn(k/v)": G}


def amplification(G: np.ndarray, idxs: tuple[int, ...]) -> float:
    sub = G[np.ix_(idxs, idxs)]
    num2 = sub.sum()
    den2 = np.trace(sub)
    return float(np.sqrt(num2 / den2)) if den2 > 0 else float("nan")


def amp_vs_n(G: np.ndarray, T: int):
    """For each subset size N in 2..T return (mean, lo, hi) amplification over all
    combinations of that size."""
    out = {}
    for N in range(2, T + 1):
        vals = [amplification(G, c) for c in itertools.combinations(range(T), N)]
        out[N] = (float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)))
    return out


def cosine_matrix(G: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    return G / np.outer(d, d)


def iter_series(full_grams: dict[str, dict[str, np.ndarray]], lora_g):
    """Yield (method, group, G) across all full-FT methods then lora, in display order."""
    for method, gd in full_grams.items():
        for g in BASE_GROUPS:
            yield method, g, gd[g]
    if lora_g is not None:
        yield "lora", "attn(k/v)", lora_g["attn(k/v)"]


# ---------------------------------------------------------------------------- #
# discovery
# ---------------------------------------------------------------------------- #
def find_full(method: str, model: str, seed: int, tasks: list[str]):
    """Locate the pretrained reference + per-task checkpoints for a full-model method
    (base or freeze). Raises FileNotFoundError if anything is missing so the caller can
    skip the method gracefully."""
    pre = f"saves_pretrained_weights/{method}/{model}/pretrained_weights_{seed}/model.safetensors"
    if not os.path.exists(pre):
        raise FileNotFoundError(f"no {method} pretrained reference at {pre}")
    ft = []
    for t in tasks:
        m = sorted(glob.glob(f"saves_bts_preliminary/{method}/{model}/train_{t}_{seed}_*/model.safetensors"))
        if not m:
            raise FileNotFoundError(f"no {method} checkpoint for task {t}")
        ft.append(m[0])
    return pre, ft


def find_lora(model: str, seed: int, tasks: list[str]):
    dirs = []
    for t in tasks:
        m = sorted(glob.glob(f"saves_bts_preliminary/lora/{model}/train_{t}_{seed}_*"))
        m = [d for d in m if os.path.exists(os.path.join(d, "adapter_model.safetensors"))]
        if not m:
            raise FileNotFoundError(f"no lora adapter for task {t}")
        dirs.append(m[0])
    return dirs


# ---------------------------------------------------------------------------- #
# plotting
# ---------------------------------------------------------------------------- #
def plot_bars(full_grams, lora_g, tasks, out_dir):
    T = len(tasks)
    full = tuple(range(T))
    rows = [
        (f"{method} / {g}", method, g, amplification(G, full))
        for method, g, G in iter_series(full_grams, lora_g)
    ]
    labels = [r[0] for r in rows]
    colors = [series_style(r[1], r[2])[0] for r in rows]
    # frozen groups (freeze/embed, freeze/norm-if-zero) have NaN amplification: draw a
    # zero-height bar and annotate, so the category is visibly present but marked frozen.
    heights = [r[3] if np.isfinite(r[3]) else 0.0 for r in rows]

    fig, ax = plt.subplots(figsize=(max(9, 1.0 * len(rows)), 5))
    bars = ax.bar(range(len(rows)), heights, color=colors)
    for b, r in zip(bars, rows):
        h = METHOD_HATCH.get(r[1], "")
        if h:
            b.set_hatch(h)
    ax.axhline(1.0, ls="--", c="gray", lw=1, label="1.0 = orthogonal (clean addition)")
    ax.axhline(np.sqrt(T), ls=":", c="black", lw=1, label=f"√{T} = fully aligned")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("amplification  ‖Στ‖ / √Σ‖τ‖²")
    ax.set_title(f"Task-vector amplification, all {T} tasks merged\n({', '.join(tasks)})")
    for b, r in zip(bars, rows):
        if np.isfinite(r[3]):
            ax.text(b.get_x() + b.get_width() / 2, r[3] + 0.01, f"{r[3]:.2f}",
                    ha="center", va="bottom", fontsize=9)
        else:
            ax.text(b.get_x() + b.get_width() / 2, 0.03, "frozen\nΔ=0",
                    ha="center", va="bottom", fontsize=8, color="gray")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"amplification_bars.{ext}"), dpi=150)
    plt.close(fig)


def plot_vs_n(full_grams, lora_g, tasks, out_dir):
    T = len(tasks)
    ns = list(range(2, T + 1))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, g, G in iter_series(full_grams, lora_g):
        data = amp_vs_n(G, T)
        mean = [data[n][0] for n in ns]
        if not np.all(np.isfinite(mean)):
            continue  # frozen / zero-delta group (e.g. freeze/embed) -> nothing to plot
        lo = [data[n][1] for n in ns]
        hi = [data[n][2] for n in ns]
        c, ls, mk = series_style(method, g)
        ax.plot(ns, mean, ls=ls, marker=mk, color=c, label=f"{method} / {g}")
        ax.fill_between(ns, lo, hi, color=c, alpha=0.10)
    ax.plot(ns, np.sqrt(ns), ls=":", c="black", lw=1, label="√N (fully aligned)")
    ax.axhline(1.0, ls="--", c="gray", lw=1, label="1.0 (orthogonal)")
    ax.set_xticks(ns)
    ax.set_xlabel("number of tasks merged (N)")
    ax.set_ylabel("amplification (mean over combinations, band = min–max)")
    ax.set_title("How constructively task vectors add as more tasks are merged")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"amplification_vs_n.{ext}"), dpi=150)
    plt.close(fig)


def plot_heatmaps(full_grams, lora_g, tasks, out_dir):
    panels = []
    for method, gd in full_grams.items():
        panels.append((f"{method} / embed", gd["embed"]))
        panels.append((f"{method} / attn(k/v)", gd["attn(k/v)"]))
    if lora_g is not None:
        panels.append(("lora / attn(k/v)", lora_g["attn(k/v)"]))
    # drop panels with no signal (e.g. freeze/embed, whose Gram is identically zero)
    panels = [(t, G) for (t, G) in panels if np.any(np.diag(G) > 0)]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, G) in zip(axes, panels):
        C = cosine_matrix(G)
        im = ax.imshow(C, vmin=-0.1, vmax=0.6, cmap="coolwarm")
        ax.set_xticks(range(len(tasks)))
        ax.set_yticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tasks, fontsize=8)
        ax.set_title(f"{title}\ncross-task cosine", fontsize=9)
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(C[i, j]) < 0.4 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"cosine_heatmaps.{ext}"), dpi=150)
    plt.close(fig)


def write_summary(full_grams, lora_g, tasks, out_dir):
    T = len(tasks)
    full = tuple(range(T))
    rows = [(method, g, amplification(G, full)) for method, g, G in iter_series(full_grams, lora_g)]

    print(f"\nAmplification with all {T} tasks merged ({', '.join(tasks)}):")
    print(f"{'method':6} {'group':12} {'amp':>6}  (1.0=orthogonal, √{T}={np.sqrt(T):.2f}=aligned)")
    for method, g, a in rows:
        if not np.isfinite(a):
            print(f"{method:6} {g:12} {'n/a':>6}  (frozen / zero delta)")
        else:
            flag = "  <-- over-amplified" if a > 1.2 else ""
            print(f"{method:6} {g:12} {a:6.3f}{flag}")

    with open(os.path.join(out_dir, "amplification_summary.csv"), "w") as f:
        f.write("method,group,n_tasks,amplification\n")
        for method, g, a in rows:
            f.write(f"{method},{g},{T},{a:.4f}\n")
        for method, gd in full_grams.items():
            for g in BASE_GROUPS:
                for n, (m, lo, hi) in amp_vs_n(gd[g], T).items():
                    f.write(f"{method},{g},{n},{m:.4f}\n")
        if lora_g is not None:
            for n, (m, lo, hi) in amp_vs_n(lora_g["attn(k/v)"], T).items():
                f.write(f"lora,attn(k/v),{n},{m:.4f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["mnli", "qnli", "qqp", "sst2", "record"])
    ap.add_argument("--methods", nargs="+", default=["base", "freeze"],
                    help="full-FT methods to include (computed like base); lora is always added")
    ap.add_argument("--model", default="llama-3.2-1b-instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="figures/amplification")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    full_grams: dict[str, dict[str, np.ndarray]] = {}
    for method in args.methods:
        try:
            pre, ft = find_full(method, args.model, args.seed, args.tasks)
        except FileNotFoundError as e:
            print(f"[skip] {method}: {e}")
            continue
        print(f"Computing {method} Gram matrices (streaming full-model deltas)...")
        full_grams[method] = base_gram(pre, ft)

    if not full_grams:
        raise SystemExit("no full-FT checkpoints found for any requested method")

    try:
        print("Computing LoRA Gram matrix (effective B@A deltas)...")
        lora_g = lora_gram(find_lora(args.model, args.seed, args.tasks))
    except FileNotFoundError as e:
        print(f"[skip] lora: {e}")
        lora_g = None

    write_summary(full_grams, lora_g, args.tasks, args.out_dir)
    plot_bars(full_grams, lora_g, args.tasks, args.out_dir)
    plot_vs_n(full_grams, lora_g, args.tasks, args.out_dir)
    plot_heatmaps(full_grams, lora_g, args.tasks, args.out_dir)
    print(f"\nSaved figures + amplification_summary.csv to {args.out_dir}/")


if __name__ == "__main__":
    main()
