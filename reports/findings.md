# Task-Vector Merging — Findings So Far

*Llama-3.2-1B-Instruct, seed 42. Living document; numbers regenerated from the
artifacts listed at the end.*

## TL;DR

1. **Embedding amplification is a real but irrelevant pathology.** Full fine-tuning
   merges over-amplify the (tied) token embeddings; freezing them (`freeze`) removes the
   amplification entirely — and changes merge accuracy essentially **not at all**
   (`freeze` ≈ `base`, marginally worse). So the embedding story does *not* explain the
   merge-quality gap.
2. **There is no universal best method.** Across 26 task combos, **LoRA wins on mean
   accuracy** (13/26) but **base wins on worst-case (min-task) retention** (16/26). This
   is a mean-vs-min trade-off, not a clear ranking.
3. **`record` breaks LoRA merges.** Whenever ReCoRD is in the merge, LoRA sacrifices it
   (2–21% of its single-task accuracy retained) while base keeps it partly alive
   (38–53%). Without `record`, LoRA dominates. The earlier "LoRA always wins" read came
   from a record-free pair and does not generalize.
4. **Implication for BTS:** task-vector *geometry* (Gram / cosine / amplification) does
   **not** predict merge accuracy — body cosines are uniformly ~0.02–0.08 (near
   orthogonal) yet one task can still be annihilated. A selection criterion needs more
   than geometry.

---

## 1. Setup

- **Model:** `llama-3.2-1b-instruct`, seed 42.
- **Tasks:** four GLUE-style sentence classification tasks (`mnli` 3-way, `qnli` binary,
  `qqp` binary, `sst2` binary) plus **`record`** (ReCoRD, span/QA — the odd task out:
  different format and a low accuracy ceiling ~0.5).
- **Methods:**
  - `base` — full fine-tuning. Task vector τ = θ_ft − θ_pre over all params.
  - `freeze` — full FT with `embed_tokens`/tied `lm_head`/final norm **frozen** (lever B
    of the amplification fix); embeddings have an exactly-zero delta.
  - `lora` — rank-8 adapter on `k_proj,v_proj` only (α=8, scaling=1.0).
- **Merging:** task arithmetic — sum the per-task vectors, scale by a single global
  coefficient, sweep the coefficient, evaluate every task (`eval.py`,
  `*_acc_coef.csv`). 26 combos: 10 pairs, 10 triples, 5 quads, 1 five-way.
- **Selection protocol used here:** *best shared coefficient* = the coef maximizing mean
  accuracy across the combo's tasks (standard task-arithmetic protocol).

### Single-task ceilings (`predict_accuracy`)

| method | mnli | qnli | qqp | sst2 | record |
|--------|------|------|-----|------|--------|
| base   | 0.898 | 0.929 | 0.905 | 0.948 | 0.510 |
| freeze | 0.898 | 0.930 | 0.906 | 0.954 | 0.513 |
| lora   | 0.886 | 0.927 | 0.896 | 0.956 | **0.473** |

All three train all five tasks well individually (record's ceiling is intrinsically low;
LoRA's record adapter is the weakest of the three).

---

## 2. Diagnostic: embedding amplification (Gram-matrix analysis)

Per parameter group we form the Gram matrix `G[i,j] = ⟨τ_i, τ_j⟩` and the
**amplification** `A = ‖Σ τ‖ / √(Σ‖τ‖²)` (1.0 = orthogonal, √N = perfectly aligned). A
single global scaling coef is tuned for the bulk of the delta, so any group whose
amplification sticks out gets systematically over-driven.

- **base/embed amplification = 1.52** (grows 1.13 → 1.52 with N=2→5), vs all other groups
  (attn/mlp/norm) ~1.03–1.08. Driven by a **task-agnostic global mean-shift** of the
  tied 128k-token embedding table (cross-task cosine 0.5–0.6), not by task signal.
- **Body groups are near-orthogonal** for all methods (cross-task cosine ~0.02–0.08).
- **LoRA never touches embeddings**, so it never has the amplification.

*Artifacts:* `figures/amplification/` (bars, amplification-vs-N, cosine heatmaps).

---

## 3. The amplification fix works on norms, not on accuracy

`freeze` zeroes the embedding delta, so embed amplification is **gone** (undefined, 0/0).
But the merged accuracy is statistically indistinguishable from `base`, in fact slightly
worse:

| method | best-min-retention wins | best-mean-acc wins |
|--------|------------------------:|-------------------:|
| base   | **16 / 26** | 9 / 26 |
| freeze | 2 / 26 | 4 / 26 |
| lora   | 8 / 26 | **13 / 26** |

`freeze` almost never wins. **Conclusion: embedding amplification is not the operative
cause of the merge-quality gap** — confirmed across all 26 combos, not just one pair.

---

## 4. Full comparison — mean vs worst-case trade-off

Aggregated over all 26 combos (at each combo's best shared coef). `ret` = merged ÷
single-task; `mean_ret` averages over the combo's tasks, `min_ret` is the worst task;
`oracle_min_ret` lets each task pick its own best coef (upper bound):

| method | mean_acc | min_acc | mean_ret | min_ret | oracle_min_ret |
|--------|---------:|--------:|---------:|--------:|---------------:|
| base   | 0.641 | 0.434 | 0.755 | **0.627** | **0.680** |
| freeze | 0.636 | 0.429 | 0.746 | 0.615 | 0.664 |
| lora   | **0.657** | 0.416 | **0.758** | 0.572 | 0.623 |

LoRA optimizes the **average**; base is more robust to the **weakest** task. Even with an
oracle per-task coefficient, base retains the worst task better than LoRA — so this is not
purely a single-coef artifact.

Retention degrades with arity for everyone (interference compounds), and **LoRA's worst
task degrades fastest**:

| arity | base min_ret | freeze min_ret | lora min_ret |
|------:|-------------:|---------------:|-------------:|
| 2 | 0.79 | 0.79 | 0.78 |
| 3 | 0.56 | 0.56 | 0.50 |
| 4 | 0.48 | 0.45 | 0.37 |
| 5 | 0.39 | 0.35 | **0.21** |

---

## 5. `record` breaks LoRA merges

Splitting the 26 combos by whether they contain ReCoRD reverses the ranking:

| subset | base min_ret | freeze min_ret | lora min_ret |
|--------|-------------:|---------------:|-------------:|
| **without** record (11) | 0.645 | 0.645 | **0.806** |
| **with** record (15)    | **0.613** | 0.593 | 0.401 |

In record-containing merges, **`record` is the task LoRA sacrifices** (the merged model
keeps the classification tasks and discards record):

| combo | base record ret | lora record ret |
|-------|----------------:|----------------:|
| mnli+qqp+record | 46% | **2%** |
| mnli+qnli+qqp+record | 38% | 16% |
| mnli+qqp+sst2+record | 53% | 21% |
| mnli+qnli+qqp+sst2+record (5-way) | 39% | 21% |

**Mechanism.** `record` has the lowest ceiling and the weakest LoRA adapter. LoRA's best
*shared* coef sits high (0.4–0.7) to activate the strong classification tasks; at that
coef record's small contribution is swamped and collapses to chance. `base` peaks at a
*low* coef (0.15–0.25), which spreads the damage: it keeps record partly alive at the cost
of under-fitting the strong tasks (e.g. mnli 43–55%). So `base`'s "robustness" is really
**uniform degradation**, while LoRA's "high mean" is **protect-the-strong, drop-the-weak**.

The earlier `mnli+qnli` result (LoRA retains both at ~0.8–0.84, base/freeze pin qnli at
chance) is a **record-free, LoRA-favorable** case — representative of the "without record"
column, not of merging in general.

---

## 6. Implications for Bayesian Task Selection

- **Geometry alone is not a merge-quality predictor.** Body task vectors are uniformly
  near-orthogonal (cosine ~0.02–0.08) across *all* methods and combos, yet outcomes range
  from "both tasks retained" to "one task annihilated." Amplification/Gram/cosine cannot
  distinguish these — a BTS criterion built only on task-vector geometry will mis-rank.
- **Compatibility is asymmetric and task-dependent.** A low-ceiling / different-format
  task (record) acts as a *victim* under LoRA but a *drag* under base. Selection should
  encode per-task properties (ceiling, format, adapter strength), not just pairwise
  similarity.
- **The selection objective matters.** "Best mean" and "best worst-case" pick different
  methods and different coefficients. BTS needs to state which it optimizes.
- **Single global coef is leaving value on the table** — but only some: oracle per-task
  coef improves min-retention modestly (base 0.63→0.68, lora 0.57→0.62), so per-task /
  TIES / DARE-style merging is worth trying but is not a complete fix.

---

## 7. Open questions / next steps

1. **Recover the sacrificed task** — per-task coefficients or TIES/DARE trimming,
   especially for record under LoRA.
2. **Why does record's LoRA contribution vanish** — measure adapter/effective-dW norms
   relative to the classification tasks; is it magnitude or direction?
3. **Build & test the BTS predictor** — can a model using task ceilings + geometry +
   method recover the observed merge ranking? Geometry alone provably cannot.
4. **Generalize** — more seeds, a larger model, more task families (currently 4 GLUE + 1
   QA).

---

## 8. Scaling to 3B / 8B — training-stability note

Extending the study to `llama-3.2-3b-instruct` and `llama-3.1-8b-instruct` surfaced two
infrastructure issues (orthogonal to the merging science, but they gate the 8B data):

- **Memory.** Full FT (`base`/`freeze`) of 8B OOMs on a single 140GB H200 (~128GB static
  in bf16: 16 weights + 16 grads + 96 AdamW). Resolved with DeepSpeed ZeRO-2 across 2 GPUs
  (`config_templates/deepspeed/ds_z2_config.json`). 3B fits on one GPU.
- **bf16 numerical instability (8B only).** 8B full FT in bf16 **diverges to NaN** a few
  steps in (training loss logged as exactly `0.0`, `eval_loss: NaN`). The cause is *not*
  data, optimization, or attention: halving LR + doubling warmup left the blow-up at the
  identical step; eager (fp32-softmax) attention gave a bit-identical loss curve and still
  died; a **seed change moved the failure step** (seed 42 → step 100, seed 43 → step 15),
  and the **3B trains clean through the identical data/batch**. So a *specific* mnli
  example tips the **8B's bf16 forward non-finite** — an 8B-only numerical knife-edge —
  and the resulting NaN gradient poisons all weights. **Fix:** train the 8B in **fp32**
  (`bf16: false`) with **ZeRO-3** (`config_templates/deepspeed/ds_z3_config.json`,
  ~80GB/GPU of the 160GB fp32 state across 2 GPUs). Verified: 8B `base` mnli now clears
  the old failure zone (step 100 loss ≈ 0.22 vs NaN), 0 zero-loss steps over 1105 logged
  steps, `eval_loss` ≈ 0.10.
- **Confound to footnote:** the 8B trains in fp32 while 1B/3B are bf16. Justified by the
  bf16 instability, but worth stating when comparing across model sizes.

---

## Artifacts & how to regenerate

All plotting/analysis runs with the `pf` conda env python (matplotlib is not in `base`):

```bash
PY=/mnt/data/home/robeke797/miniconda3/envs/pf/bin/python
PYTHONPATH=src $PY scripts/visualize_amplification.py --tasks mnli qnli qqp sst2 record --methods base freeze
PYTHONPATH=src $PY scripts/visualize_merged.py        # -> figures/merged/merged_coef_sweeps_{base,freeze,lora}.{png,pdf}
PYTHONPATH=src $PY scripts/summarize_merged.py        # -> figures/merged/merge_summary.csv + console aggregates
```

- `figures/amplification/` — amplification bars, amplification-vs-N, cosine heatmaps.
- `figures/merged/merged_coef_sweeps_{method}.{png,pdf}` — per-method coef sweeps, one
  subplot per combo, dotted = single-task ceiling.
- `figures/merged/merge_summary.csv` — per-(method,combo) best-coef stats used above.
