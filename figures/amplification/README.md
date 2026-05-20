# Task-vector amplification: why base (full FT) merges worse than LoRA

This diagnostic explains why task arithmetic with **full fine-tuning ("base")** underperforms
**LoRA** on this project, and quantifies the mechanism. Figures and `amplification_summary.csv`
in this directory are produced by [`scripts/visualize_amplification.py`](../../scripts/visualize_amplification.py).

## TL;DR

When you merge several task vectors, the **embeddings** (which are tied to `lm_head`) pick up a
shared, task-agnostic drift that adds up *constructively* across tasks. A single global merge
coefficient then over-drives that drift and distorts the output head. Attention/MLP updates don't
have this problem, and LoRA never touches embeddings at all — so it merges cleanly.

The single number that captures this is **amplification**.

## What "amplification" means

For a set of tasks `S`, with per-task task vectors `τ_t` (restricted to a parameter group, e.g.
embeddings or attention), amplification is

```
            ‖ Σ_{t∈S} τ_t ‖
A(S) =  ─────────────────────────
          √( Σ_{t∈S} ‖τ_t‖² )
```

- **Numerator** = the length of the vector you *actually* add when merging — the real vector sum.
- **Denominator** = the length that sum *would* have if the task vectors were mutually
  perpendicular (Pythagoras).

So amplification measures **how much longer the merged update is than a clean, orthogonal
combination of the same pieces**. It is purely about *alignment/direction* — magnitude cancels
out, so large and small task vectors can have the same amplification.

### Geometry

Think tip-to-tail vector addition:

```
  uncorrelated directions                 aligned directions
     τ1 →                                    τ1 →→→
         ↑ τ2                                τ2 →→→
     sum ↗  (partial cancellation)          sum →→→→→→  (lengths just add)
```

- **Uncorrelated / perpendicular** → the pieces partly cancel; the sum grows like a random walk,
  `√N`. That's the denominator, so `A = 1.0`.
- **Aligned (same direction)** → lengths add directly; the sum grows like `N`, so `A = √N`.

Hence `A = 1` means "clean addition, every task lives in its own subspace"; `A = √N` means
"everything piled onto one shared direction."

### The closed form that builds intuition

For `N` task vectors with average pairwise cosine `c̄` (and similar norms):

```
A ≈ √( 1 + (N − 1)·c̄ )
```

- If `c̄ = 0` (uncorrelated) → `A = 1` **for any N**. Adding more tasks never inflates it.
- If `c̄ > 0` → `A` grows with **both** the correlation **and** the number of tasks merged.

That is exactly the rising `base/embed` curve in `amplification_vs_n.png`: each new task stacks
another copy of the same drift direction.

## Why it matters for the merge

Task arithmetic scales the **whole** merged delta by **one** global coefficient `λ`:

```
θ_merged = θ_0 + λ · Σ_t τ_t
```

`λ` is chosen to put the *useful, near-orthogonal* part (attention + MLP) at the right strength,
but it multiplies every parameter group equally. A group with high amplification has already
contributed a sum inflated beyond its "fair" orthogonal size, so `λ` drives it proportionally too
far. For embeddings — tied to `lm_head` — that over-shoots the output head and corrupts
predictions, while attention/MLP sit right where `λ` intended.

**Amplification is the single number that tells you which group a shared global coefficient will
systematically over-drive, and how much worse it gets as you merge more tasks.** It is *not* "how
big the update is" — it's "how coherently the per-task updates stack relative to a clean
orthogonal sum," and that coherence is precisely what one merge coefficient can't accommodate.

## Measured results (llama-3.2-1b-instruct, seed 42, tasks: mnli/qnli/qqp/sst2/record)

All 5 tasks merged (`1.0` = orthogonal, `√5 ≈ 2.24` = fully aligned):

| method | group       | amplification |
|--------|-------------|---------------|
| base   | attn(k/v)   | 1.06          |
| base   | attn        | 1.05          |
| base   | mlp         | 1.03          |
| base   | **embed**   | **1.41** ⚠    |
| base   | norm        | 1.07          |
| lora   | attn(k/v)   | 1.02          |

- **`amplification_bars.png`** — the bar chart above; only `base/embed` lifts off the orthogonal line.
- **`amplification_vs_n.png`** — `base/embed` climbs with N (1.11 → 1.41 for N=2→5) toward the `√N`
  line, while every other group (base attn/mlp/norm **and** all of LoRA) stays pinned at ~1.0.
- **`cosine_heatmaps.png`** — the *why*: base embeddings are correlated across tasks (cosine
  0.13–0.56), whereas base attn(k/v) and LoRA attn(k/v) are near-zero off-diagonal.

### What the embedding drift actually is

It is **not** that embeddings change a lot — in magnitude the embed delta is *smaller* than
attn/mlp (rel. ≈ 0.25–0.5% vs 0.4–0.8% of the weight norm). The correlation comes from a
**task-agnostic global mean-shift**: the whole 128k-token embedding table slides in one common
direction (5–46% of the embed-delta energy per task), and those mean-shift vectors are nearly
identical across tasks (cosine **0.62–0.88**). The *task-specific* embedding movement is small and
scattered (top-200 changed-token Jaccard 0.01–0.14). The correlated part carries no task signal
but stacks constructively under summation — that's the amplification.

## The fix

Two levers (see also the project memory note):

- **A — no retraining:** restrict the base task vector to attention + MLP in
  `src/utils.py` (`METHODS_TO_TARGET_MODULES["base"] = ["self_attn", "mlp"]`). This drops
  embeddings (tied to `lm_head`) and layernorms from the merge while keeping each single-task
  model's own embedding adaptation. Re-run the `eval.py` coefficient sweep afterward.
- **B — retrain with frozen embeddings:** freeze `embed_tokens` (also freezes the tied `lm_head`)
  during base fine-tuning. More principled — removes the body↔embedding train/merge mismatch and
  matches LoRA's setup (LoRA freezes embeddings and still gets equal-or-better single-task
  accuracy, so embedding training isn't needed for these tasks).

## Regenerating

```bash
PYTHONPATH=src python scripts/visualize_amplification.py \
    --tasks mnli qnli qqp sst2 record \
    --model llama-3.2-1b-instruct --seed 42 \
    --out-dir figures/amplification
```

The script builds per-group Gram matrices once (streamed, low memory) so any task subset is cheap
to evaluate, and compares base in weight space against LoRA's *effective* delta
`ΔW = (α/r)·B·A`, so the comparison is apples-to-apples.
