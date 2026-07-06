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
5. **Target transfer is real and mixture-dependent (§10).** Merged source vectors can
   match fine-tuning a held-out target directly (rte: NAI 1.00 from `mnli+qqp+sst2`;
   mrpc: 0.93 from `mnli+qnli`) or barely help (cola: 0.19) — and each target has a
   *different* best mixture, which is exactly the ground truth a task-selection method
   must predict.

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

## 9. Subspace alignment (SAR) and the isotropic ("iso") baseline

**What SAR measures.** When we merge by summing task vectors, the sum spans a
*subspace* of weight-space — a set of directions the merged update can move along. A
task is well-served by the merge only if *its* important directions lie inside that
shared subspace; if they don't, summing simply discards them and that task degrades. The
**subspace alignment ratio (SAR)**, from Marczak et al. (iso-merging, arXiv:2502.04959),
quantifies this per 2D weight matrix.

Concretely, for each task vector τ:
- **Define the merged subspace.** SVD the summed task vector and keep its top-k left
  singular vectors `U_k`, where k is the smallest rank capturing 95% of the spectral
  energy (Eq. 6). `U_k` is an orthonormal basis for "the directions the merge actually
  uses." (Why the top-k and not all of `U`? The tail directions carry negligible energy
  and are mostly noise; truncating asks the sharper question "do τ's *strong* directions
  fall in the merge's *strong* directions.")
- **Project τ onto it.** `proj = U_k U_kᵀ τ` keeps only the part of τ that lives in the
  merged subspace and zeroes the rest.
- **Measure how much survived.** `SAR = ‖S_proj‖₂ / ‖S_τ‖₂` (Eq. 5), the ratio of the
  projected matrix's spectral norm (largest singular value) to the original's. It is a
  fraction in [0, 1]: **SAR ≈ 1** ⇒ τ lives almost entirely inside the shared subspace
  (the merge can represent it, so it should be retained); **low SAR** ⇒ the merge subspace
  misses τ's dominant directions (the sum is structurally unable to keep that task).

We average SAR over all matrices to get one number per task. Intuitively SAR is *not*
pairwise similarity between tasks — it asks whether a task fits in the **collective**
subspace the sum produces, which is exactly the quantity a merge either preserves or
throws away.

**What "iso" means.** The singular values in `S` are the per-direction "stretch factors"
of the summed task matrix along its orthogonal singular directions. A raw sum has an
**anisotropic** spectrum — a few directions dominate, the rest are tiny — so when task
vectors are summed those few dominant directions collide and interfere (the merge
pathology iso-merging targets). The fix is to **flatten the spectrum**: replace every
singular value with their common mean, `S_iso = ones_like(S) · mean(S)`. The reconstructed
operator `U · diag(mean) · Vᵀ` then stretches *every* direction equally — no direction
dominates. A map that scales all directions the same is **isotropic** (a sphere vs. an
ellipsoid); hence the name. It is **not** literally "ones" — overall scale is preserved
via the mean; only the *shape* of the spectrum is made uniform.

In our `sar()` (`src/utils.py`) the `iso=True` branch uses this flattened spectrum when
choosing the merged subspace's rank, so `sar_iso` reports alignment against the *isotropic*
merge's subspace rather than the raw sum's. We pair SAR with **NAI** (normalized accuracy
improvement, `(merged − zero_shot) / (finetuned − zero_shot)`) the same way the paper's
Fig. 3a does — higher subspace alignment should track higher retained accuracy.

*Artifacts:* `scripts/compute_nai_sar.py` (→ `figures/nai_sar.csv`),
`scripts/visualize_nai_sar.py` and `scripts/visualize_nai_sar_compare.py` (NAI-vs-SAR
scatters, the latter overlaying `sar` vs `sar_iso` per task, one figure per method).

---

## 10. Target-task transfer — the ground truth for task selection

**Question.** Bayesian task selection must pick, for a *target* task, the source-task
subset whose merged vector transfers best. To score any selection method we need the
ground truth: evaluate **every** source mixture on every target and rank them.

**Setup.** Four held-out targets: `mrpc` (paraphrase), `boolq` (yes/no QA), `rte`
(NLI), `cola` (acceptability). `src/eval/eval_target.py` (launched via
`scripts/slurm/eval_target_array.sh`, same 0–25 array indexing as the source sweep)
rebuilds each of the 26 source-combo task vectors and sweeps the same 40-point
coefficient grid, but evaluates the merged model on the four targets →
`saves_bts_merged/{method}/{model}/{tasks}_{seed}_target/…_target_acc_coef.csv`.
`scripts/summarize_target.py` ranks the combos per target by accuracy at the
**per-target best** coefficient (`df[target].idxmax()`) and reports **NAI**
(`(merged − zero_shot) / (finetuned − zero_shot)`; 1.0 = matches fine-tuning the
target, 0 = no gain over pretrained). Note this uses target eval data to pick the
coefficient — it is the transfer *ceiling*, the right ground truth for ranking
mixtures, not a zero-target-data protocol.

**Results (lora, 1B, seed 42)** — best mixture per target:

| target | best mixture | coef | acc | NAI | zero-shot | single-task FT |
|--------|--------------------|------|-------|-------|-----------|------|
| rte    | mnli+qqp+sst2      | 0.60 | 0.755 | **1.000** | 0.256 | 0.755 |
| mrpc   | mnli+qnli          | 0.60 | 0.762 | 0.934 | 0.005 | 0.816 |
| boolq  | sst2+record        | 0.40 | 0.556 | 0.405 | 0.387 | 0.803 |
| cola   | qnli+qqp+sst2      | 0.30 | 0.691 | 0.193 | 0.664 | 0.803 |

**Findings.**

- **Transfer can be total.** For rte, the best mixture *equals* fine-tuning rte
  directly (NAI 1.00) without ever seeing rte training data. Every top-rte and
  top-mrpc mixture contains `mnli` — sensible (rte is NLI; mrpc benefits from
  entailment-style supervision) and a pattern a selection method should recover.
- **Or nearly nothing.** cola's raw accuracy (0.691) looks fine but NAI exposes it:
  zero-shot is already 0.664, so mixtures recover <20% of the fine-tuning gain.
  boolq sits in between (NAI 0.41).
- **The coefficient is critical for transfer.** boolq collapses to exact-match ≈ 0
  past coef ~0.5–0.75 (the usual lora output-format collapse, §5) — *below* its own
  0.387 zero-shot. The coefficient the *source* sweep selects often lands past that
  cliff, so a mixture that transfers fine at coef 0.4 scores 0.000 at the
  source-selected 0.85. Per-target rankings are only meaningful at per-target coefs.
- **Distinct winners per target** (and 26-way full rankings in
  `figures/target/target_summary.csv`) give the comparison baseline for the BTS
  selection method: top-1 hit, top-k overlap, or rank correlation against the
  method's predicted ordering.

*Artifacts:* `figures/target/target_summary.{csv,tex}` — full per-target rankings
(CSV: 26 combos × 4 targets per method) and the paper-ready booktabs table.

---

## Reproducing the pipeline end-to-end

Everything runs on SLURM with the `pf` conda env. Order matters — later stages read
the earlier stages' outputs from `saves_*` directories.

```bash
# 0. Pretrained-weight snapshots (EPOCHS=0 "fine-tune" = the task-vector origin θ_pre)
#    -> saves_pretrained_weights/{method}/{model}/pretrained_weights_{seed}/
bash scripts/pretrained_weights.sh          # edit peft_methods/models at the top

# 1. Single-task fine-tunes + their evals
#    -> saves_bts_preliminary/{method}/{model}/{train,eval}_{task}_{seed}_{ts}/
bash scripts/slurm/preliminary_source.sh    # sources: mnli qnli qqp sst2 record
bash scripts/slurm/preliminary_target.sh    # targets: mrpc boolq rte cola
bash scripts/slurm/zero_shot.sh             # zero-shot refs on all 9 tasks

# 2. Correct per-eval metrics + gathered reference table
#    -> compute_metrics.jsonl in each eval dir; results.csv at repo root
python scripts/compute_metrics.py           # per-dir metrics (edit lists at top)
python scripts/compute_metrics.py --gather_results   # -> results.csv

# 3. Source coefficient sweeps (one array index per combo; edit MODELS/METHODS
#    constants at the top of src/eval/eval.py first)
#    -> saves_bts_merged/{method}/{model}/{tasks}_{seed}_best/…_acc_coef.csv
sbatch --array=0-25 scripts/slurm/eval_array.sh

# 4. Target transfer sweeps (same indexing; constants in src/eval/eval_target.py)
#    -> saves_bts_merged/{method}/{model}/{tasks}_{seed}_target/…_target_acc_coef.csv
sbatch --array=0-25 scripts/slurm/eval_target_array.sh

# 5. Analysis / figures / tables — see the block below
```

Steps 0–2 must complete before 3–4 (the sweeps read the fine-tuned checkpoints and
pretrained snapshots); step 2's zero-shot/fine-tuned references are needed for every
NAI number (steps 5's `compute_nai_sar.py` and `summarize_target.py`). Re-running
step 1 creates a *new* timestamped `train_*` dir per task and the task-vector builder
globs the first match — keep exactly one train dir per (method, task).

## Artifacts & how to regenerate

All plotting/analysis runs with the `pf` conda env python (matplotlib is not in `base`):

```bash
PY=/mnt/data/home/robeke797/miniconda3/envs/pf/bin/python
PYTHONPATH=src $PY scripts/visualize_amplification.py --tasks mnli qnli qqp sst2 record --methods base freeze
PYTHONPATH=src $PY scripts/visualize_merged.py        # -> figures/merged/merged_coef_sweeps_{base,freeze,lora}.{png,pdf}
PYTHONPATH=src $PY scripts/summarize_merged.py        # -> figures/merged/merge_summary.csv + console aggregates
PYTHONPATH=src $PY scripts/compute_nai_sar.py --device cuda    # -> figures/nai_sar.csv (NAI + SAR/SAR_iso per task)
PYTHONPATH=src $PY scripts/visualize_nai_sar.py               # -> figures/nai_sar/nai_vs_sar.{png,pdf}
PYTHONPATH=src $PY scripts/visualize_nai_sar_compare.py       # -> figures/nai_sar/nai_vs_sar_compare_{method}.{png,pdf}
PYTHONPATH=src $PY scripts/compare_best_coef.py --models llama-3.2-1b-instruct llama-3.2-3b-instruct \
    --labels 1B 3B --plot                                     # -> figures/merged/best_coef_compare.{tex,png,pdf}
$PY scripts/summarize_target.py --model llama-3.2-1b-instruct # -> figures/target/target_summary.{csv,tex} + ranking
```

- `figures/amplification/` — amplification bars, amplification-vs-N, cosine heatmaps.
- `figures/merged/merged_coef_sweeps_{method}.{png,pdf}` — per-method coef sweeps, one
  subplot per combo, dotted = single-task ceiling.
- `figures/merged/merge_summary.csv` — per-(method,combo) best-coef stats used above
  (per-model snapshots in `merge_summary_llama-3.2-{1b,3b}.csv`).
- `figures/merged/best_coef_compare.{tex,png,pdf}` — 1B-vs-3B best-coef table +
  grouped barplot (best coef and accuracy per combo, bars = model × method).
- `figures/nai_sar{,_iso}/`, `figures/nai_sar.csv` — NAI-vs-SAR scatters (§9).
- `figures/target/target_summary.{csv,tex}` — per-target source-mixture rankings (§10).
- `results.csv` — gathered zero-shot / single-task-FT exact-match references.
