# Task-Vector Merging for Task Selection — Findings

*Llama-3.2-1B-Instruct, seed 42, **full fine-tuning (`base`) only**. Living document;
every number below is regenerated from the artifacts listed at the end.*

> **Scope.** This document covers the `base` method on the 12-source × 12-target
> grid. Earlier sections comparing `base`/`freeze`/`lora` on a 5-task grid (merge
> quality, embedding amplification, the ReCoRD/LoRA interaction, 3B/8B training
> stability) were **removed on 2026-08-25**: their source sweeps had been deleted,
> their numbers no longer matched the artifacts they cited, and one headline
> conclusion inverted when checked. They are recoverable from git history, and are
> not cited here. LoRA may be re-added later as a second method.

## TL;DR

1. **Transfer from merged source vectors is real, and often total.** Of 12 held-out
   targets, **4 reach NAI ≥ 1.0** — the merged sources match or beat fine-tuning
   directly on the target, having never seen its training data (`cb` 1.25, `rte`
   1.12, `copa` 1.04, `rotten_tomatoes` 1.01). Only `cola` barely moves (0.06).
2. **The best mixture is small, target-specific, and not guessable.** Winners span
   2–8 of the 12 sources (**median 3**), and no source dominates — `record`,
   `hellaswag` and `winogrande` each appear in 6 of the 12 winners. Merging
   everything is not the answer, which is exactly why selection is a problem worth
   solving.
3. **Subspace alignment (SAR) does not solve it.** Per-target ρ(SAR, transfer) has
   median 0.17 and is **negative for 3 of 12** targets; top-5 overlap with the
   accuracy ranking is ~0/5 almost everywhere. SAR can track the broad trend and
   still miss the argmax, and selection only needs the argmax.
4. **Bayesian optimization does solve it, on plain subset indicators.** With a
   63-evaluation budget over **4083 candidate mixtures** (1.5% of the space), a GP
   over the binary subset vector — `BOTAS (binary)` — finds the exact optimum in
   **83%** of runs in ~24 evaluations, against **7%** for random search. Alignment
   features are worse than the indicator, and adding iso-SAR to it changes nothing
   measurable.
5. **Cosine similarity to the target is the best *cheap* signal, and still no
   substitute for search.** Scoring all 4083 mixtures against the target's own task
   vector costs **4 CPU-minutes** (one 24×24 Gram matrix; no GPU, no merged-model
   evaluation) and reaches median regret **0.034** at the **3.6th percentile** of the
   true ranking, beating raw SAR 10–0–2 and edging iso-SAR 5–6–1. But it lands on the
   best mixture for **1 of 12** targets against BO's **83%**, and — like SAR — it
   needs a **fine-tune on the target task**, which BO does not. Both limits belong to
   the geometric baselines, not to (4). DTVG's unnormalized dot product, applied to a
   summed mixture, degenerates into "merge everything".

---

## 1. Setup

- **Model:** `llama-3.2-1b-instruct`, seed 42.
- **Method:** `base` — full fine-tuning. Task vector τ = θ_ft − θ_pre over all
  parameters, relative to an EPOCHS=0 snapshot (`saves_pretrained_weights/`).
- **12 source tasks:** `mnli`, `qnli`, `qqp`, `sst2`, `record`, `snli`, `anli_r1`,
  `paws`, `imdb`, `squad_v2`, `hellaswag`, `winogrande`.
- **12 held-out target tasks:** `mrpc`, `boolq`, `rte`, `cola`, `cb`, `scitail`,
  `stsb`, `cr`, `rotten_tomatoes`, `multirc`, `copa`, `piqa`.
- **Merging:** task arithmetic — sum the per-task vectors, scale by a single global
  coefficient, sweep that coefficient over 40 points (0.05 → 2.0), evaluate.
- **Search space:** every source subset of size ≥ 2 — **4083 mixtures**, each
  evaluated on all 12 targets (`src/eval/eval_target.py`).
- **Metrics:** exact match, except `stsb` (Pearson correlation). Single-task
  fine-tuned and zero-shot references come from `results.csv`.

---

## 2. Target-task transfer — the ground truth for selection

**Question.** Task selection must pick, for a *target* task, the source subset whose
merged vector transfers best. Scoring any selection method needs the ground truth:
evaluate **every** mixture on every target and rank them. That is what this grid is.

**Setup.** `src/eval/eval_target.py` rebuilds each of the 4083 source-mixture task
vectors, sweeps the same 40-point coefficient grid, and evaluates the merged model on
all 12 held-out targets. `scripts/summarize_target.py` ranks mixtures per target by
accuracy at the **per-target best** coefficient (`df[target].idxmax()`) and reports
**NAI** = (merged − zero_shot) / (finetuned − zero_shot): 1.0 means the mixture matches
fine-tuning on the target itself, 0 means no gain over the pretrained model.

Note this picks the coefficient using target eval data, so it is the transfer
*ceiling* — the right ground truth for **ranking mixtures**, not a zero-target-data
protocol. A deployable method needs a coefficient rule that does not see the target.

**Best mixture per target** (base, 1B, seed 42; best of 4083):

| target | best mixture | size | coef | acc | NAI | zero-shot | single-task FT |
|--------|--------------|-----:|-----:|----:|----:|----------:|---------------:|
| `cb` | `mnli+sst2+record+snli+anli_r1+paws+imdb+winogrande` | 8 | 0.65 | 0.893 | **1.250** | 0.000 | 0.714 |
| `rte` | `mnli+qnli+qqp+sst2+anli_r1+paws+squad_v2+winogrande` | 8 | 0.25 | 0.841 | **1.117** | 0.256 | 0.780 |
| `copa` | `hellaswag+winogrande` | 2 | 0.35 | 0.770 | **1.044** | 0.300 | 0.750 |
| `rotten_tomatoes` | `sst2+record+imdb` | 3 | 0.55 | 0.929 | **1.011** | 0.231 | 0.921 |
| `cr` | `sst2+record+anli_r1` | 3 | 0.40 | 0.934 | **0.996** | 0.184 | 0.936 |
| `scitail` | `mnli+qqp+anli_r1+hellaswag` | 4 | 0.30 | 0.865 | **0.899** | 0.012 | 0.960 |
| `piqa` | `imdb+hellaswag+winogrande` | 3 | 0.35 | 0.721 | **0.820** | 0.368 | 0.799 |
| `mrpc` | `record+hellaswag` | 2 | 0.80 | 0.684 | **0.805** | 0.005 | 0.848 |
| `multirc` | `mnli+qnli+anli_r1+paws` | 4 | 0.20 | 0.757 | **0.766** | 0.440 | 0.854 |
| `stsb` | `qnli+paws+winogrande` | 3 | 0.45 | 0.705 | **0.707** | 0.265 | 0.887 |
| `boolq` | `qnli+record+paws+hellaswag+winogrande` | 5 | 0.35 | 0.699 | **0.707** | 0.387 | 0.828 |
| `cola` | `record+imdb+hellaswag` | 3 | 0.30 | 0.674 | **0.057** | 0.664 | 0.832 |

**Findings.**

- **Transfer can be total.** 4 of 12 targets reach **NAI ≥ 1.0**
  (`cb` 1.25, `rte` 1.12, `copa` 1.04, `rotten_tomatoes` 1.01) — the merged sources
  match or *beat* fine-tuning directly on the target, without ever seeing its training
  data. Merging acts as transfer, not merely as compression.
- **Or nearly nothing.** `cola` reaches 0.674 accuracy, which looks respectable until
  NAI exposes it: zero-shot is already 0.664, so the best mixture recovers **6%** of the
  fine-tuning gain. Raw accuracy is not interpretable across targets; NAI is.
- **Winning mixtures are small.** Sizes span 2–8 of 12 sources,
  **median 3**. Merging all 12 is never the answer, and the arity
  bias in raw SAR (§3) therefore points the wrong way by construction.
- **No source dominates.** The most frequent members of a winning mixture are `record` 6, `hellaswag` 6, `winogrande` 6
  (of 12 winners). Each target wants a *different* subset — which is precisely the
  ground truth a selection method has to predict, and the reason the answer cannot be
  hardcoded.
- **The coefficient matters as much as the subset.** Winning coefficients span
  0.20–0.80. A mixture evaluated at the wrong coefficient can
  score near zero, so per-target rankings are only meaningful at per-target coefficients.

*Artifacts:* `figures/target/target_summary.{csv,tex}` — full rankings, 4083 mixtures ×
12 targets with NAI and the SAR columns of §3; `target_matrix.{csv,tex}` — the full
accuracy matrix (best mixture per target in bold); `target_best_per_target.{csv,tex}`;
`target_sar_scatter.{png,pdf}`.

---

## 3. Subspace alignment (SAR) and the isotropic ("iso") baseline

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
- **Measure how much survived.** `SAR = ‖S_proj‖₂ / ‖S_τ‖₂` (Eq. 5). It is a fraction
  in [0, 1]: **SAR ≈ 1** ⇒ τ lives almost entirely inside the shared subspace (the merge
  can represent it, so it should be retained); **low SAR** ⇒ the merge subspace misses
  τ's dominant directions (the sum is structurally unable to keep that task).

  **What `‖·‖₂` means here.** `S_proj` and `S_τ` are the *vectors of singular values*, and
  `np.linalg.norm(v, ord=2)` on a 1-D input is the Euclidean norm — √(Σσ²) — which is the
  **Frobenius** norm of the matrix those singular values came from, not its spectral norm.
  So as implemented, `SAR = ‖proj‖_F / ‖τ‖_F`: **the fraction of τ's total energy retained
  by the projection**, summed over all directions, rather than the ratio of largest
  singular values. Read the ratio that way when interpreting any SAR number in this
  document. (`src/utils.py:alignment_ratio`. If the intent was a spectral-norm ratio, as
  a reading of Eq. 5 as ‖·‖₂-on-a-matrix would suggest, then the implementation deviates
  from the reference and every SAR number here is the energy-fraction variant.)

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

**Two SAR variants are used in this document.** *In-mixture* SAR projects a source
task's own vector onto the mixture it belongs to (`scripts/compute_nai_sar.py` →
`figures/nai_sar.csv`) and needs the **source** sweeps. *Target* SAR projects a
held-out target's vector onto the source mixture's subspace
(`scripts/compute_target_sar.py` → `figures/target/target_sar.csv`) and is the variant
§4 evaluates as a selection signal — it needs no merged-model evaluation of the target,
only the target's own single-task fine-tune.

*Artifacts:* `figures/target/target_sar.csv` (target SAR, 4083 mixtures × 12 targets,
current). `figures/nai_sar.csv` and the `scripts/visualize_nai_sar*.py` scatters are
**pending**: the existing file is from the deleted 5-task source sweeps, and the
12-source `base` re-run is in progress — do not cite it until regenerated.

---

## 4. Selecting mixtures: SAR vs Bayesian optimization at scale

**The pilot this replaces.** An earlier version of this experiment ran on 5 source
tasks (26 mixtures) and 4 targets, with `lora`. This section re-answers its questions
on the full `base` grid — **12 sources → 4083 mixtures × 12 targets**, a 157× larger
search space — and supersedes it throughout. Several of the pilot's headline numbers
turn out to be artifacts of the small grid rather than findings about the method, and
they are called out below so the earlier write-up is not cited by mistake.

The one pilot result *not* superseded here, because this section does not repeat it:
Bayesian optimization was validated against exhaustive search over a **live SAR
objective** on the 5-source space (31 subsets), where it located the exhaustive optimum
for all 4 targets in 8–14 of 20 evaluations, beating greedy forward selection and random
search. That established the BO machinery works; the question this section answers is
what to give the GP as *features*. The driver (`scripts/bo_task_selection.py`) and its
launcher were removed after the proof of concept — see git history.

**Protocol.** Retrospective BO over the evaluated grid (no new training runs): GP
+ EI, budget **63 evaluations** (3 initial + 60), **10 seeds**, method `base`,
`scripts/analysis_bo_variants.py`. The budget is 1.5% of the space; the pilot's 13
evaluations were *half* of its 26.

The method is **BOTAS**, and its variants are named for the features given to the GP —
`BOTAS (binary)`, `BOTAS (SAR)`, and so on — in the table below and in every figure
under `figures/bo_variants/`. Those labels live in `DISPLAY_METHODS` in
`analysis_bo_variants.py`; the internal keys differ because they index the cached
convergence curves. Note that **`Random` is not a BOTAS variant** — it is the baseline
the variants are read against. Where §3 and §5 write "SAR" or "iso-SAR" they mean the
alignment *signal*, not a BOTAS configuration.

The random baseline (labelled simply **Random** in the figures) is **paired** — it
replays each BO run's own initial design and then keeps drawing in the same order. Up to `n_initial` the two are the same
estimator (running max over uniform draws), so an unpaired baseline puts sampling
noise exactly where BO has not yet fit a GP; in the 5-seed pilot that noise showed
BO ~0.06 ahead before it had done anything. `analysis_bo_variants.py` asserts the
pairing holds — every BO variant's first `n_initial` steps must equal the baseline's
exactly — and hard-fails if the two sampling paths ever drift apart.

*Validation (not plotted):* an independent unpaired 50-trial random baseline scored
8% wins / 53.0 evals against the paired baseline's 7% / 51.1, i.e. pairing tightened
the comparison without shifting the baseline's level.

### Which GP features find high-transfer mixtures?

| method (GP features)             | wins (found true best) | evals to within 1% | pilot (n=26) |
|----------------------------------|------------------------|--------------------|--------------|
| **`BOTAS (binary)`**             | **83%**                | **23.6**           | 100%         |
| `BOTAS (binary + iso-SAR)`       | 82%                    | 24.1               | 90%          |
| `BOTAS (iso-SAR)`                | 48%                    | 31.4               | 65%          |
| `BOTAS (iso-SAR, norm)`          | 22%                    | 32.4               | —            |
| `BOTAS (SAR)`                    | 15%                    | 47.9               | 40%          |
| `BOTAS (SAR, norm)`              | 12%                    | 38.8               | —            |
| `Random` (10 seeds, baseline)    | 7%                     | 51.1               | 57%          |

1. **`BOTAS (binary)` still dominates, and this is the result that scales.** The pilot
   worried that a subset indicator could not cover 2^n as n grew. At 12 sources it
   still finds the exact optimum in 83% of runs, in ~24 of 4083 candidates.
2. **Random search collapses, 57% → 7%.** This is the largest correction to the pilot.
   At n=26 a 13-evaluation budget covered half the space, so random looked
   competitive for purely combinatorial reasons. The BO-vs-random gap is real and
   large; the pilot's near-tie was an artifact of the toy grid.
3. **Adding iso-SAR to the indicator does nothing measurable.** The pilot reported
   `BOTAS (binary + iso-SAR)` just behind `BOTAS (binary)` (90% vs 100%). At 5 seeds
   it led (83% vs 82%); at 10 paired seeds the ordering *reverses* to 83% vs 82% the
   other way, and speed
   flips with it (23.6 vs 24.1). Doubling the seeds and removing a noise source
   moved the ranking instead of sharpening it, so the honest statement is that the
   two are indistinguishable — not that iso-SAR adds a little.
4. **`BOTAS (SAR)` is weak but no longer worse than random** (15% vs 7%). The pilot's
   "raw SAR as a GP feature is worse than random search" does not survive the larger
   grid.
5. **iso-SAR remains the only alignment feature with real signal** (`BOTAS (iso-SAR)`,
   48%), consistent
   with §3.

### Does SAR predict transfer?

Compared like-for-like (per target, as the pilot reported), correlations are **more
spread out, not uniformly stronger**: ρ(SAR, acc) ranges −0.38…+0.94 with **median
0.17**, and **3 of 12 targets are negative** (`cola` −0.38, `mrpc` −0.16, `cb`
−0.10). Two targets carry the positive end (`rotten_tomatoes` 0.94, `cr` 0.89).
iso-SAR beats raw on 7 of 12.

**Do not quote the pooled number.** `correlations.csv`'s `overall` row (0.53 raw /
0.59 iso) is a single Spearman over all 12 targets stacked together, and it is
roughly double the per-target median purely because targets differ in level — mean
accuracy spans 0.314–0.863 and mean SAR 0.561–0.698 across targets, so pooling
manufactures between-target correlation that says nothing about ranking mixtures
*within* a target, which is the only thing selection does.

**But top-5 overlap with the accuracy ranking is 0/5 on almost every target.** SAR
now tracks the broad trend across thousands of mixtures while still failing to
identify the *best* one — and selection only needs the argmax. The pilot's conclusion
("SAR alone does not select the right mixture") holds for a different reason than
it gave: not because SAR is uninformative, but because being globally correlated
and being right at the top are different properties.

### Per-target failures worth chasing

- **`cr` defeats every method** — best win rate 20% (both binary variants), despite
  ρ(SAR, acc) = 0.89. Plausibly a near-tie among the top mixtures rather than a
  search failure; needs the top-k accuracy spread checked before drawing a
  conclusion.
- **`mrpc` is where raw SAR actively hurts** — 0.5578 mean vs 0.6838 true best,
  *below* the paired random baseline (0.6426), 0% wins. It is the one target with a
  clearly negative ρ(SAR, acc) = −0.16, and SAR-guided BO inherits that.
- **`boolq` splits the alignment scores**: raw SAR reaches 0.6904 while iso-SAR gets
  0.6570 — the one target where the iso variant is clearly worse.

### Two implementation faults that would have invalidated this

Both were found while scaling and are fixed; they are recorded because either one
silently produces a plausible-looking table.

- **`combo_to_binary` hardcoded the 5-task vocabulary.** Every mixture built from
  one of the 7 new sources encoded as the all-zero vector, collapsing unrelated
  mixtures onto identical GP inputs. It now reads the list from
  `src/eval/eval_target.py` and raises on an unknown task.
- **`simulate_bo_on_grid` swallowed GP-fitting failures** and fell back to random
  without a word. That failure mode is indistinguishable in the output from "BO does
  not beat random" — the exact claim at stake. It now warns; this run reported **0
  fallbacks in 43,200 fits** (12 targets x 6 variants x 10 seeds x 60 iterations),
  so these numbers are real BO.

A third fault was only a cost, not a correctness problem: EI was evaluated one
candidate at a time (~4080 acquisition calls per iteration). Batched into a single
forward pass it is **612× faster**, identical to 9.3e-10.

*Artifacts:* `figures/bo_variants/bo_variants_summary.csv` (mean±std at budget),
`bo_variants_wins_speed.csv` (the wins/speed tables above), `bo_variants_curves.npz`
(raw convergence curves), `bo_convergence_all_variants.{png,pdf}`,
`bo_summary_bars.{png,pdf}`; `figures/analysis/` (correlations.csv, per-target
scatters, ranking comparison).

Produced by SLURM jobs 79135 + 79421 (`ONLY=bo_variants N_ITERATIONS=60 SEEDS=10
sbatch scripts/slurm/analysis.sh`; the first job timed out at 9 of 12 targets and
the second resumed from the per-target checkpoint). All CSVs are written natively
at full precision. `bo_variants_curves.npz` holds the raw curves, so every figure
and table here redraws in seconds via `analysis_bo_variants.py --from-curves` — no
GP refit needed for restyling or for a new metric.

One caveat on exact reproduction: BLAS thread count perturbs the last floating-point
digits of the GP fits, which can flip an EI argmax. Against an earlier unpinned run
this moved the aggregate by a single run in 120 (`BOTAS (binary)` 100/120 -> 99/120 wins,
i.e. 83% either way at the reported precision). The rankings and every conclusion
below are unaffected, but do not expect bit-identical numbers across machines.

---

## 5. Cosine similarity to the target — the cheapest selection signal

**What this tests.** §3–§4 ask whether subspace alignment picks the right mixture.
This asks the same question of the other obvious piece of geometry: how similar is a
mixture's task vector to the **target's own** task vector? That is the selection rule
behind DTVG ([Zhang et al., Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1374/)),
moved from their soft-prompt setting into weight-space task arithmetic. Like SAR it
needs no merged-model evaluation — only the target's single-task fine-tune, which
`saves_bts_preliminary/` already holds — so it is a real selection signal and not an
oracle over the transfer grid.

**Two choices, crossed.** DTVG's similarity is the *dot product* of mean-pooled prompt
vectors (their Eq. 2); they explicitly position cosine as the weaker SPoT metric they
beat. Their Target Similarity **averages** per-source scores rather than scoring the
merged mixture. Both choices are varied independently, so any difference can be
attributed to one of them:

|                                | cosine    | dot product (DTVG Eq. 2) |
|--------------------------------|-----------|--------------------------|
| summed mixture vector          | `cos_mix` | `dot_mix`                |
| mean over members (DTVG TS)    | `ts_cos`  | `ts_dot`                 |

plus DTVG's second grouping metric, Knowledge Consistency — mean pairwise similarity
*among* the members, target-independent — as `kc_cos` / `kc_dot`. Their 1/r²
token-length factor is dropped: it is a fixed scale and cannot change a ranking.

**Cost: 4 minutes on one CPU node, no GPU.** Because

    cos(Σ_{i∈M} s_i, t) = (Σ_{i∈M} ⟨s_i,t⟩) / ( √(Σ_{i,j∈M} ⟨s_i,s_j⟩) · ‖t‖ )

every signal is a function of the 24×24 Gram matrix of the 12 source and 12 target
task vectors. One pass over 25 checkpoints yields it and all 4083 mixtures × 12
targets follow in closed form — no mixture vector is ever formed, and the cost does
not grow with the number of mixtures scored. The task vectors are 3.89 GB each (973M
fp32 params), so ~93 GB would not fit in RAM; they are streamed key-by-key with the
Gram accumulated in float64. Both halves are checked against direct dense computation
(`src/tests/task_cos_gram.py`, `task_cos_emit.py`) — worst relative error 9.5e-13 on
real checkpoints, 3e-16 on the signal algebra.

### Results

Regret = (best accuracy available for that target) − (accuracy of the selected
mixture). `pct` is where the pick lands in the true ranking of 4083 mixtures. ρ is the
**median per-target** Spearman correlation with accuracy — never the pooled value, for
the reason given in §4. `ov@10` is mean top-10 overlap with the accuracy ranking.
"vs iso-SAR" is the per-target win–tie–loss record on regret.

| signal            | mean reg | med reg | med pct | median ρ | ov@10 | vs iso-SAR |
|-------------------|----------|---------|---------|----------|-------|------------|
| *oracle*          | 0        | 0       | 0%      | —        | —     | —          |
| **`ts_cos`**      | **0.122**| **0.034**| **3.6%**| **+0.493**| 0.125 | **5–6–1**  |
| `ts_dot`          | 0.147    | 0.069   | 11.4%   | +0.416   | 0.067 | 5–4–3      |
| `sar_iso` (§3)    | 0.155    | 0.110   | 19.5%   | +0.421   | 0.117 | —          |
| `cos_mix`         | 0.163    | 0.108   | 23.6%   | +0.398   | 0.125 | 4–6–2      |
| `sar` (§3)        | 0.181    | 0.142   | 21.8%   | +0.174   | 0.025 | 5–0–7      |
| *random mixture*  | 0.223    | 0.222   | 55.7%   | —        | —     | —          |
| `dot_mix` ≡ *all* | 0.257    | 0.234   | 50.0%   | +0.335   | 0.000 | —          |
| `kc_cos`          | 0.414    | 0.378   | 79.6%   | −0.052   | 0.008 | —          |
| `kc_dot`          | 0.414    | 0.378   | 79.6%   | −0.145   | 0.000 | —          |

Per-target regret for the signals worth comparing:

| target            | `ts_cos` | `ts_dot` | `cos_mix` | `sar_iso` | `sar`  | random | *all*  | oracle size |
|-------------------|----------|----------|-----------|-----------|--------|--------|--------|-------------|
| `mrpc`            | 0.2328   | 0.3676   | 0.3627    | 0.3676    | 0.3603 | 0.3263 | 0.3676 | 2           |
| `boolq`           | 0.0731   | 0.0731   | 0.1086    | 0.1086    | 0.0920 | 0.2190 | 0.5917 | 5           |
| `rte`             | 0.1119   | 0.3141   | 0.1588    | 0.1119    | 0.1444 | 0.2242 | 0.2599 | 8           |
| `cola`            | 0.0086   | 0.4688   | 0.1074    | 0.1323    | 0.1390 | 0.1227 | 0.1419 | 3           |
| `cb`              | 0.5536   | 0.0893   | 0.5536    | 0.5536    | 0.2857 | 0.2905 | 0.0893 | 8           |
| `scitail`         | 0.0452   | 0.0640   | 0.2611    | 0.2775    | 0.2686 | 0.1706 | 0.1091 | 4           |
| `stsb`            | 0.3932   | 0.3716   | 0.3542    | 0.2260    | 0.5059 | 0.3038 | 0.3542 | 3           |
| `cr`              | 0.0133   | 0.0133   | 0.0133    | 0.0133    | 0.0080 | 0.0710 | 0.0505 | 3           |
| `rotten_tomatoes` | 0.0019   | 0.0019   | 0.0019    | 0.0019    | 0.0047 | 0.0917 | 0.0572 | 3           |
| `multirc`         | 0.0227   | 0.0037   | 0.0227    | 0.0227    | 0.0565 | 0.1291 | 0.2182 | 4           |
| `copa`            | **0**    | **0**    | 0.0100    | 0.0400    | 0.1200 | 0.3248 | 0.2500 | 2           |
| `piqa`            | 0.0016   | 0.0016   | 0.0016    | 0.0016    | 0.1877 | 0.4071 | 0.5892 | 3           |

1. **Cosine-to-target is the strongest cheap signal, but it does not beat iso-SAR
   target by target.** `ts_cos` roughly halves iso-SAR's median regret (0.034 vs
   0.110) and lands at the 3.6th percentile of the true ranking against iso-SAR's
   19.5th, with the best median per-target ρ of any signal here (+0.493, negative on
   only 1 of 12 targets vs raw SAR's 3). But the head-to-head record is **5 wins, 6
   ties, 1 loss** — on half the targets the two pick mixtures of identical accuracy,
   and the mean-regret gap is carried by `scitail` (+0.232) and `mrpc` (+0.135),
   partly given back by its one loss on `stsb` (−0.167). The defensible claim is
   that cosine is *at least as good as* iso-SAR and much better than raw SAR (10–0–2),
   not that it dominates. It is also the only signal that ever hits the exact optimum
   (`copa`, shared with `ts_dot`).
2. **Normalization is the decision that matters — not cosine vs dot product.** On the
   summed mixture, cosine beats DTVG's dot product 9–1–2, and `dot_mix` is not merely
   weak but **degenerate: it selects the all-12 mixture for all 12 targets**, so its
   row is numerically identical to *all*. Unnormalized dot product on a sum rewards
   every added member with a positive projection, so it maximizes at the full set;
   dividing by ‖Σ s_i‖ is what makes the signal a selector at all. At DTVG's own
   averaging aggregation, by contrast, cosine vs dot is **4–5–3** — no established
   difference. The mean gap there (0.122 vs 0.147) is two targets swinging in opposite
   directions (`cola` 0.009 vs 0.469 for cosine; `cb` 0.554 vs 0.089 against it), not
   a broad effect. **DTVG's metric is fine; applying it to a summed mixture is not.**
   Note this does not contradict their paper: mean-pooled soft prompts of fixed token
   length carry little norm information, whereas a sum of variable-size weight deltas
   is almost all norm.
3. **Averaging over members beats scoring the merged mixture**, in both metrics
   (`ts_cos` 0.122 < `cos_mix` 0.163; `ts_dot` 0.147 < `dot_mix` 0.257). DTVG's
   aggregation choice is the right one here, which is mildly surprising: `cos_mix`
   scores the object the sweep actually merges, and still loses to a signal that never
   forms it.
4. **Every cheap signal still misses the top of the ranking.** `ts_cos` averages 0.125
   top-10 overlap and its **median is 0.0** — the same failure §3 documents for SAR:
   these signals rank the field broadly while rarely resolving its top few, and
   selection only needs the argmax. See the limitations below for what that costs
   against §4.
5. **The small-mixture prior is not the explanation.** `ts_cos` picks a size-2 mixture
   on all 12 targets, so its advantage could have been a size bias rather than
   selection skill. It is not — and the bias would be a *handicap*: size-2 is the
   **worst** mixture size on average (grid mean accuracy 0.505 at size 2 vs 0.576 at
   size 7), so a random size-2 pick scores mean regret **0.284**, worse than a random
   mixture of any size (0.223). `ts_cos` reaches 0.122 from inside its own worst
   stratum, beating random-size-2 by 0.16 accuracy.
6. **Knowledge Consistency alone is not a selector**, as expected: it is
   target-independent by construction, so it picks `mnli+snli` for every target, with
   median ρ near zero and slightly negative (`kc_cos` −0.052, `kc_dot` −0.145) and
   mean regret 0.414 — far worse than random. This is a sanity check passing, not a
   negative result: DTVG uses KC as a
   second-stage filter *after* ranking by Target Similarity, never on its own.

### What the baselines require, and what they still miss

Two limitations apply to every signal in this section *and* to SAR in §3, and §4's
Bayesian source-task selection shares neither of them.

**1. They do not find the best mixture; BO does.** `ts_cos` and `ts_dot` land on the
accuracy-best mixture for **1 of 12** targets (8.3%, both on `copa`); `cos_mix`, `sar`
and `sar_iso` for **none**. §4's `BOTAS (binary)` finds the exact
optimum in **83% of 120 (target, seed) runs**, and reaches within 1% of it in ~24 of
4083 evaluations. The median top-10 overlap of 0.0 says the same thing from the other
side. A cheap geometric signal narrows the field; it does not answer the question, and
on this grid nothing in §3 or §5 comes close to the search method.

**2. They require a fine-tune on the target task.** Every signal here — and SAR — is
computed against the **target's own task vector**, which does not exist until the
target has been fine-tuned. §4's BO needs no such thing: it scores candidate mixtures
by evaluating *merged* models on the target, so it needs target **evaluation** data and
63 merge-and-evaluate cycles, and never a target training run. The baselines are
therefore both more demanding in what they need and less accurate in what they find
than the method they are baselines for.

That asymmetry is a cost of transplanting DTVG into weight space, not a flaw in their
design. In their setting the target "fine-tune" is a soft prompt at 0.035% of
parameters, re-estimated during the target training they are already doing — which is
what makes their *dynamic* re-grouping possible at all. The weight-space equivalent is
a full fine-tune of the entire model. The requirement is nearly free where they propose
it and expensive here.

### Two failures, one of them suspicious

`cb` (86.7th percentile) and `stsb` (79.1st) are the only targets where `ts_cos` picks
*worse than random*, and they fail for different reasons.

- **`cb` is the mechanism working against itself.** Its oracle is a size-8 mixture and
  merging all 12 sources is near-optimal there (regret 0.089, its best result of any
  method in the table) — a target that wants *many* sources. Cosine's norm penalty
  charges a mixture for growing, so it picks a pair and loses 0.55. `ts_dot`, which has
  no such penalty, is the best signal on `cb` at 0.089. Across all 12 targets `ts_cos`
  regret does trend with oracle mixture size (ρ = +0.485; mean regret 0.093 for
  oracle size ≤ 3 vs 0.161 for ≥ 4), but at n = 12 that is **not significant**
  (p = 0.11) and iso-SAR shows a weaker version of the same trend (+0.215). Treat the
  size-penalty story as a hypothesis this grid cannot settle, not a result.
- **`stsb` is not explained by size** — its oracle is a size-3 mixture, well inside
  what cosine can express, and every signal does badly on it (best is iso-SAR at
  0.226; raw SAR is worst in the whole table at 0.506). It is also the one target
  scored by Pearson correlation rather than exact match, and the one whose column
  needed the `REQUIRE_NONZERO=stsb` metric fix mid-sweep. **Verify the `stsb` column
  before reading anything into its numbers**; a target whose metric was repaired
  partway through the grid is the wrong place to draw a conclusion about geometry.

*Artifacts:* `figures/target/target_cos.csv` (48996 rows, six signals),
`target_cos_gram.npz` (the cached 24×24 Gram matrix — re-emitting every signal from it
takes seconds and touches no checkpoint), `figures/analysis/cos_baseline.{csv,txt}`
(per-target detail and the aggregate above). Produced by SLURM job 83290
(`sbatch scripts/slurm/task_cos.sh`, 4 min) then
`scripts/analysis_cos_baseline.py`.

---

## Reproducing the pipeline end-to-end

Everything runs on SLURM. Order matters — later stages read the earlier stages'
outputs from `saves_*` directories. See [`../README.md`](../README.md) for the
script-by-script map and the cluster gotchas (`conda activate` is broken; arrays
cap at 48; `sacct` is unusable).

> **The commands below regenerate the current 12-source / 12-target / 4083-combination
> `base` sweep** — the one every number in §1–§4 comes from.

```bash
PY=/mnt/home/$USER/miniconda3/envs/pf/bin/python   # conda activate does not work here

# 0. Pretrained-weight snapshots (EPOCHS=0 "fine-tune" = the task-vector origin θ_pre)
#    -> saves_pretrained_weights/{method}/{model}/pretrained_weights_{seed}/
bash scripts/pretrained_weights.sh          # edit peft_methods/models at the top

# 1. Single-task fine-tunes + their evals. Task lists are per-script; the
#    canonical SOURCE_TASKS/TARGET_TASKS live in src/eval/eval_target.py.
#    -> saves_bts_preliminary/{method}/{model}/{train,eval}_{task}_{seed}_{ts}/
bash scripts/slurm/preliminary_source.sh    # 12 sources
bash scripts/slurm/preliminary_target.sh    # 12 targets
bash scripts/slurm/zero_shot.sh             # zero-shot refs

# 2. Correct per-eval metrics + gathered reference table
#    -> compute_metrics.jsonl in each eval dir; results.csv at repo root
$PY scripts/compute_metrics.py <eval_dir>          # per-dir metrics (edit lists at top)
# Note the hyphen, and the positional root dir — `--gather_results` is rejected:
$PY scripts/compute_metrics.py saves_bts_preliminary --gather-results  # -> results.{csv,tex}

# 3. Source coefficient sweeps (one array index per combo; edit MODELS/METHODS
#    constants at the top of src/eval/eval.py first)
#    -> saves_bts_merged/{method}/{model}/{tasks}_{seed}_best/…_acc_coef.csv
#    4083 combos exceed MaxArraySize and the queued-job cap, so go through the wrapper:
scripts/slurm/run_eval_batched.sh eval_array.sh

# 4. Target transfer sweeps (same indexing; constants in src/eval/eval_target.py)
#    -> saves_bts_merged/{method}/{model}/{tasks}_{seed}_target/…_target_acc_coef.csv
scripts/slurm/run_eval_batched.sh eval_target_array.sh
scripts/combo_status.py target --count      # 0 = every combination wrote a CSV

# 5. Target SAR (needed before summarize_target's sar columns and the §4 analyses)
#    -> figures/target/target_sar.csv     ~2 h sharded; ~26 h in a single process
sbatch --array=0-47 scripts/slurm/target_sar_array.sh
bash scripts/slurm/target_sar_array.sh --merge

# 5b. Cosine / DTVG selection signals (§5; also fills summarize_target's cos columns)
#     -> figures/target/target_cos.csv + target_cos_gram.npz    ~4 min, CPU-only
sbatch scripts/slurm/task_cos.sh

# 6. The transfer grid the analyses read -> figures/target/target_summary.csv
PYTHONPATH=src $PY scripts/summarize_target.py --model llama-3.2-1b-instruct --seed 42

# 7. §4-§5 selection analyses — CPU-only, run on the cpu_short partition:
mkdir -p logs_analysis
sbatch scripts/slurm/analysis.sh       # analysis_correlations + analysis_bo_variants
PYTHONPATH=src $PY scripts/analysis_cos_baseline.py   # §5 table; seconds, reads CSVs only

# 8. Remaining analysis / figures / tables — see the block below
```

`scripts/slurm/bo_selection.sh` and `scripts/bo_task_selection.py` (the online BO
over a live SAR objective, §4's pilot) were **removed** after the proof-of-concept; the
`figures/bo/` and `figures/bo_iso/` artifacts they produced are still on disk. Use
git history to recover them if that experiment is ever revisited.

Steps 0–2 must complete before 3–4 (the sweeps read the fine-tuned checkpoints and
pretrained snapshots); step 2's zero-shot/fine-tuned references are needed for every
NAI number (`summarize_target.py` in step 6, `compute_nai_sar.py` below). Re-running
step 1 creates a *new* timestamped `train_*` dir per task and the task-vector builder
globs the first match — keep exactly one train dir per (method, task).

## Artifacts & how to regenerate

All plotting/analysis runs with the `pf` env python (matplotlib is not in `base`).
Call the interpreter directly — `conda activate pf` fails since the Aug 2026 remount:

```bash
PY=/mnt/home/$USER/miniconda3/envs/pf/bin/python
PYTHONPATH=src $PY scripts/summarize_target.py --model llama-3.2-1b-instruct --seed 42
N_ITERATIONS=60 SEEDS=10 sbatch scripts/slurm/analysis.sh   # correlations + BO variants
```

Current, `base`, 12 sources × 12 targets — safe to cite:

- `figures/target/target_summary.{csv,tex}` — per-target mixture rankings, 4083 × 12,
  with NAI, both SAR columns and (once step 5b has run) the six cosine/DTVG columns
  (§2, §4, §5). `target_matrix.{csv,tex}` — full accuracy
  matrix. `target_best_per_target.{csv,tex}` — the §2 table.
- `figures/target/target_sar.csv` — target SAR / iso-SAR, 48996 rows (§3, §4).
- `figures/target/target_cos.csv` — the six cosine / DTVG selection signals, 48996
  rows (§5). `target_cos_gram.npz` — the cached 24×24 task-vector Gram matrix every
  one of them is derived from; re-emitting the CSV from it costs seconds and reads no
  checkpoints.
- `figures/analysis/cos_baseline.{csv,txt}` — §5's per-target and aggregate
  selected-vs-best comparison, including the SAR columns and the
  oracle/random/all reference rows.
- `figures/analysis/` — `correlations.csv` (per-target ρ; **ignore its `overall` row**,
  see §4), per-target scatters, ranking comparison, simulated-BO convergence.
- `figures/bo_variants/` — `bo_variants_summary.csv` (mean±std at budget),
  `bo_variants_wins_speed.csv` (win rate + evals-to-1%), `bo_variants_curves.npz`
  (raw curves), convergence and summary-bar figures (§4).
- `results.csv` / `results.tex` — zero-shot and single-task-FT references for all 24
  tasks, source vs target shaded.

**Not currently citable:**

- `figures/nai_sar.csv`, `figures/nai_sar{,_iso}/` — in-mixture SAR + NAI (§3). From the
  deleted 5-task source sweeps; the 12-source `base` re-run is in progress.
- `figures/merged/*` — per-mixture merge-quality stats. Same deleted source sweeps, and
  the numbers disagreed with the write-up that cited them. Regenerate with
  `summarize_merged.py` once the source sweep completes.
- `figures/bo/`, `figures/bo_iso/` — online BO over the live SAR objective, from the
  5-source `lora` pilot; their driver script no longer exists.
