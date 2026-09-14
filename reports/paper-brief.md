# Bayesian Task Selection for Task-Vector Merging — Paper Brief

*Self-contained context for drafting. You do not need repository access to use
this document; every number, task list and configuration value needed to write
about the work is reproduced here.*

**Snapshot: 2026-09-01.** The **target sweep is complete** — all 4083 source
mixtures × 12 targets, all 40 coefficients. The stsb re-scoring is finished, the
12-source SAR is computed, and the BO feature comparison has been run at full
scale. The **source sweep is 634 / 4083** and is the only outstanding run; §7
says what depends on it.

**Scope: `base` (full fine-tuning) only.** The earlier `lora`/`freeze` comparison
has been withdrawn — see §8. LoRA may be added later as a second method.

**Status tags.** Every claim carries one.

- `VALIDATED` — computed from complete artifacts on disk, checked, and not
  affected by any known measurement bug. Safe to put in the paper.
- `PROVISIONAL` — correct as computed, but the underlying data is still being
  extended in a way that could change the answer.
- `WITHDRAWN` — previously reported, now known not to be supportable.
- `PENDING` — not computed yet.

---

## 1. What the paper argues

**Research question.** Given a pool of source tasks with fine-tuned task vectors,
**which subset should be merged to maximise performance on a held-out target
task** — and can that subset be found without exhaustively evaluating every
mixture?

The claim structure has three parts:

1. **Subset selection is a real problem with a non-trivial answer.** Which
   sources you merge matters enormously for a given target, the best subset
   differs per target, and it is not predictable from simple heuristics like
   "use everything" or "use the most similar task".
2. **The ground truth exists and is semantically coherent.** The exhaustive sweep
   over all 4083 source subsets × 12 targets shows targets selecting same-family
   sources, which is evidence the measurement is real rather than noise.
3. **Bayesian optimisation over the subset space finds good subsets
   sample-efficiently** — 83% exact-optimum recovery in ~24 of 4083 evaluations
   against 7% for random search — and the question of *what the GP should take as
   input features* has a negative-result answer worth reporting (§3).

A fourth, methodological contribution is worth a section: merging studies of this
kind are quietly fragile to per-task metric choice and to stale configuration,
and the default paths produce plausible-looking numbers that are wrong (§5).

---

## 2. Experimental setup

**Model.** `meta-llama/Llama-3.2-1B-Instruct`, seed 42, chat template `llama3`,
cutoff length 2048. Single seed, single model — there are **no variance
estimates anywhere** in this work, which limits how strongly any comparison can
be stated.

**Fine-tuning.** Full fine-tuning (`finetuning_type: full`), 1 epoch, per-device
batch 8, lr 2e-6, cosine schedule, warmup ratio 0.1, weight decay 1e-2, AdamW,
bf16, validation split 0.1. Labelled `base` throughout.

**Task vectors.** A task vector is the difference between the fine-tuned weights
and the pretrained weights θ_pre. θ_pre is captured explicitly as an
"EPOCHS=0 fine-tune" so the origin is exactly the checkpoint the runs started
from, not a re-downloaded copy.

**Data.** All tasks come from the `kinit/peft-factory` HuggingFace dataset.

**Source tasks (12)** — `mnli, qnli, qqp, sst2, record, snli, anli_r1, paws,
imdb, squad_v2, hellaswag, winogrande`

**Target tasks (12)** — `mrpc, boolq, rte, cola, cb, scitail, stsb, cr,
rotten_tomatoes, multirc, copa, piqa`

The two sets are **disjoint** — no target task contributes a task vector.

**Evaluation-set sizes span three orders of magnitude**, which matters for how
much weight any single per-task result can carry, and for runtime (§7):

| | total examples | mean/task | smallest | largest |
|---|---|---|---|---|
| sources | 138,955 | 11,580 | `sst2` 872 | `qqp` 40,430 |
| targets | 16,908 | 1,409 | `cb` 56 | `multirc` 4,848 |

The source tasks are **8.2× larger on average** than the targets. See §4.4 and §7.

**Merging.** Task arithmetic: sum the per-task vectors of the chosen subset,
scale by a **single shared coefficient λ**, add to θ_pre, evaluate. Each
combination is swept over λ ∈ {0.05, 0.10, …, 2.00} — 40 points. λ=0 is dropped
because it is just the pretrained model.

**Combination space.** All subsets of size 2..12 of the 12 sources = **4083
combinations**, each evaluated at 40 coefficients on all 12 targets = 48,996
(combination, target) results. That is the exhaustive ground truth any selection
method is scored against.

**Two sweeps, two different questions:**

| sweep | evaluated on | question | state |
|---|---|---|---|
| target | the 12 held-out targets | transfer | **complete** (4083/4083) |
| source | the sources it merged | retention / interference | running (634/4083) |

**Metrics.** Per-task, not uniform: exact match for classification, ReCoRD's
grouped-by-`idx` metric for `record`, official SQuAD v2 EM for `squad_v2`,
Pearson correlation for `stsb`. §5 is about what happens when this is got wrong.

**Reference points.** Zero-shot (pretrained) and single-task fine-tuned scores
are on disk for all 24 tasks (`results.csv`). **NAI** (normalised accuracy
improvement, after Marczak et al.) = (merged − zero_shot) / (finetuned −
zero_shot). NAI = 1.0 means the merge matches fine-tuning directly on the target;
NAI = 0 means merging buys nothing over the pretrained model.

---
## 3. Method: Bayesian optimisation over source subsets

The selection method is BO over the {0,1}^12 subset space with a `SingleTaskGP`
surrogate and expected improvement. Candidates are the already-evaluated mixtures,
so the simulation is retrospective: no training runs happen, and every method is
scored on the same exhaustive ground truth.

Feature sets, all computable *before* any merged model is evaluated:

- **Binary subset vector** — one indicator per source task (12 dims).
- **SAR (subspace alignment ratio)** — build a subspace from the source subset's
  summed task vector, project the *target's own* task vector onto it. Computable
  from single-task checkpoints alone.
- **iso-SAR** — the isotropic variant: the singular values used to choose the
  subspace rank are replaced by **their common mean** (not by ones — overall scale
  is preserved, only the spectrum's *shape* is flattened).
- **normalised variants** — either score divided by mixture size, to remove the
  arity confound below.

`VALIDATED`. Budget **63 evaluations of 4083 candidates** (1.5% of the space),
10 seeds, paired random baseline:

| GP features | found true best | evals to within 1% |
|---|---|---|
| **Binary subset vector** | **83%** | **23.6** |
| Binary + iso-SAR | 82% | 24.1 |
| iso-SAR | 48% | 31.4 |
| iso-SAR (arity-normalised) | 22% | 32.4 |
| raw SAR | 15% | 47.9 |
| raw SAR (arity-normalised) | 12% | 38.8 |
| Random baseline | 7% | 51.1 |

**Four findings worth stating carefully:**

1. **Plain subset indicators win, and this is the result that scales.** The earlier
   5-source phase raised the worry that an indicator encoding could not cover 2^n as
   n grew. At 12 sources — 157× the search space — it still recovers the exact
   optimum in **83%** of runs within ~24 evaluations.
2. **Random search is not competitive** (7%). The 5-source phase put random at 57%,
   but its 13-evaluation budget covered *half* of a 26-mixture space; that near-tie
   was combinatorial, not a property of the method. The BO-vs-random gap is real
   and large only once the space is big enough to be interesting.
3. **Alignment features are worse than indicators, and raw SAR is much worse**
   (15% vs 83%). iso-SAR is the only alignment score with real signal (48%).
   Arity-normalising helps raw SAR's speed but not its hit rate.
4. **Adding iso-SAR to the indicator changes nothing measurable** (82% vs 83%;
   24.1 vs 23.6 evaluations). The ordering flips between seed counts, so the honest
   statement is that the two are indistinguishable — not that alignment adds a
   little. This retires the 5-source phase's "Binary + iso-SAR is close behind".

**The arity confound.** A bigger summed vector spans a wider subspace, so raw SAR
increases with mixture size and its argmax tends to the full source set — which
transfers suboptimally everywhere, since the best mixtures are small (§4.2). Any
alignment-based objective has to handle this explicitly.

**BO machinery validation.** On the 5-source space, BO over a *live* SAR objective
found the exhaustive optimum (31 subsets) for all 4 targets in 8–14 of 20
evaluations, beating greedy forward selection and random search. The optimiser was
never the problem; the objective was. That driver has since been removed from the
repository — cite it as a preliminary check, not as a current artifact.

---

## 4. Results

### 4.1 Coverage

`VALIDATED`. **4083 / 4083 target combinations complete — 100% at every mixture
size.** No coverage caveats remain, and claims about mixture size are no longer
confounded with sampling density.

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| combinations | 66 | 220 | 495 | 792 | 924 | 792 | 495 | 220 | 66 | 12 | 1 |
| complete | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

The stsb re-scoring is also complete: **0 combinations** still hold the all-zero
column the old metric produced, so stsb is now on the same footing as every other
target.

### 4.2 Transfer is task-family aligned — the core positive result

`VALIDATED` at n=4083, all targets. Best source mixture per target:

| target | best acc | best source mixture | k | λ | NAI | single-task FT |
|---|---|---|---|---|---|---|
| `cr` | 0.934 | `sst2+record+anli_r1` | 3 | 0.40 | 0.996 | 0.936 |
| `rotten_tomatoes` | 0.929 | `sst2+record+imdb` | 3 | 0.55 | **1.011** | 0.921 |
| `cb` | 0.893 | `mnli+sst2+record+snli+anli_r1+paws+imdb+winogrande` | 8 | 0.65 | **1.250** | 0.714 |
| `scitail` | 0.865 | `mnli+qqp+anli_r1+hellaswag` | 4 | 0.30 | 0.899 | 0.960 |
| `rte` | 0.841 | `mnli+qnli+qqp+sst2+anli_r1+paws+squad_v2+winogrande` | 8 | 0.25 | **1.117** | 0.780 |
| `copa` | 0.770 | `hellaswag+winogrande` | 2 | 0.35 | **1.044** | 0.750 |
| `multirc` | 0.757 | `mnli+qnli+anli_r1+paws` | 4 | 0.20 | 0.766 | 0.854 |
| `piqa` | 0.721 | `imdb+hellaswag+winogrande` | 3 | 0.35 | 0.820 | 0.799 |
| `stsb` | 0.705 | `qnli+paws+winogrande` | 3 | 0.45 | 0.707 | 0.887 |
| `boolq` | 0.699 | `qnli+record+paws+hellaswag+winogrande` | 5 | 0.35 | 0.707 | 0.828 |
| `mrpc` | 0.684 | `record+hellaswag` | 2 | 0.80 | 0.805 | 0.848 |
| `cola` | 0.674 | `record+imdb+hellaswag` | 3 | 0.30 | 0.057 | 0.832 |

**The semantic story.** Commonsense targets select commonsense sources
(`copa` → hellaswag+winogrande; `piqa` → imdb+hellaswag+winogrande). Sentiment
selects sentiment (`cr` and `rotten_tomatoes` both anchor on sst2). NLI targets
select NLI sources (`scitail` → mnli+qqp+anli_r1+hellaswag; `rte` → an
mnli/qnli/qqp/anli_r1 core). This alignment is the strongest evidence the pipeline
measures something real rather than noise.

**Best mixtures are small — with two instructive exceptions.** **Nine of twelve**
targets are best served by k ≤ 4 (median 3, and two targets by a single *pair*),
but `cb` and `rte` both want **8-source** mixtures. Because coverage is now
complete at every k, this is a real finding rather than a sampling artefact:
merging all 12 sources is never optimal, yet the optimum is not reliably tiny
either — which is exactly why a search is needed rather than a size heuristic.
(The 5-source phase reported a maximum of k=7 and 10 of 12 targets at k ≤ 4; those
figures came from a grid where k ≥ 8 was only 17–33% sampled.)

**No source dominates.** The most frequent members of a winning mixture are
`record` (6/12), `hellaswag` (6/12), `winogrande` (6/12). Each target wants a
different subset, which is precisely the ground truth a selection method must
predict and the reason the answer cannot be hardcoded.

**Two things to say explicitly rather than bury:**

- **`cola` is the informative negative.** NAI 0.057 means merging buys
  essentially nothing over the pretrained model — zero-shot is already 0.664 against
  a 0.832 fine-tuned ceiling. Consistent with grammaticality judgement being
  idiosyncratic and not recoverable from any mixture of these sources. A selection
  method cannot be blamed for failing here; there is nothing to find.
- **NAI > 1 on 4 targets** (`cb` 1.25, `rte` 1.12, `copa` 1.04, `rotten_tomatoes` 1.01)
  means the merge *beats* fine-tuning directly on the target. Plausible on
  low-ceiling targets where single-task FT overfits a small training set — `cb` has
  56 evaluation examples and a 0.714 fine-tuned ceiling — but it deserves a
  sentence of explanation rather than being presented as a straight win.

### 4.3 Subspace alignment does not predict the argmax

`VALIDATED`. Per-target Spearman ρ(SAR, transfer accuracy) across all 4083
mixtures: **median 0.17**, range
-0.38…0.94, and **negative on
3 of 12** targets. iso-SAR is stronger (median
0.42, better on 7 of 12 by |ρ|) but still misses
on `mrpc` (-0.51).

Two targets carry the positive end: `rotten_tomatoes`
0.94 and `cr`
0.89. Yet **top-5 overlap with the
accuracy ranking is ~0/5 on almost every target**, including those two. SAR can
track the broad trend across thousands of mixtures and still fail to identify the
best one — and selection only needs the argmax.

> **Do not quote the pooled correlation.** `correlations.csv` carries an `overall`
> row (0.53 raw / 0.59 iso) computed by stacking all 12 targets together. It is
> roughly triple the per-target median purely because targets differ in level —
> mean accuracy spans 0.314–0.863 across targets — so pooling manufactures
> between-target correlation that says nothing about ranking mixtures *within* a
> target. Use the per-target numbers above.

### 4.4 The merge coefficient concentrates far below the swept range

`VALIDATED` at n=48,996. Distribution of the winning λ:

| interval | share | cumulative |
|---|---|---|
| [0.05, 0.10) | 35.6% | 35.6% |
| [0.10, 0.20) | 43.9% | 79.5% |
| [0.20, 0.30) | 10.1% | 89.6% |
| [0.30, 0.50) | 4.8% | 94.4% |
| [0.50, 0.75) | 3.4% | 97.8% |
| [0.75, 1.00) | 1.6% | 99.4% |
| [1.00, 1.50) | 0.6% | 100.0% |
| [1.50, 2.00] | 0.0% | 100.0% |

**79% of all winners sit below λ = 0.2**, and the highest winner anywhere is
1.80. The top half of the swept grid — 20 of 40 points, half the compute per
combination — has produced essentially nothing.

**The 36% at the grid minimum is not clipping of good results.** Among each
target's *top-decile* mixtures only **9.3%** sit at the minimum. Low-λ pinning is
concentrated in mixtures that do not transfer, where the best available move is
trivially "merge as little as possible". Among top-decile mixtures the percentiles
are p50 0.200, p90 0.500, p99 0.750, max 1.20.

**Per-target behaviour differs sharply**, so a single shared λ range does not suit
all targets:
  - `boolq` pins 90.1% at the grid minimum, median 0.05, max 0.75
  - `multirc` pins 65.4% at the grid minimum, median 0.05, max 0.50
  - `cola` pins 65.2% at the grid minimum, median 0.05, max 1.20
  - `cb` pins 0.7% at the grid minimum, median 0.55, max 1.80
  - `piqa` pins 0.5% at the grid minimum, median 0.15, max 1.00

**Practical recommendation on file (not applied):** λ_max = 1.25 covers every good
mixture with headroom; holding the grid at 41 points would give a 0.031 step — 60%
finer resolution at identical cost. Not applied, because changing the grid makes
best-λ an argmax over different support and breaks comparability with the completed
sweep.

### 4.5 Small evaluation sets produce large ties

`VALIDATED`. Ties at each target's maximum accuracy:
  - `cb`: **67** mixtures tied at 0.892857
  - `mrpc`: **2** mixtures tied at 0.683824
  - `scitail`: **2** mixtures tied at 0.865005

`cb` has 56 evaluation examples, so accuracy is quantised to ~1.8% steps and
**"the best mixture for cb" is not identifiable** — 67 mixtures share the maximum
and an arbitrary tie-break silently picks one.

**Implication for the paper.** Any "our method recovered the optimal subset" claim
must state how ties were handled, and `cb` — and probably `copa`, n=100 — should be
excluded from or footnoted in top-1 recovery rates. Reporting "recovered a mixture
within ε of the optimum" is more honest than top-1 on these targets. Ties are
currently broken toward the smaller mixture and the tie count is reported.

---
## 5. Measurement pitfalls — a methods contribution

Eight bugs were found and fixed. Most silently corrupted numbers that looked
entirely plausible, which is the point worth making in print: **in a merging study
the default paths fail quietly.**

### 5.1 Per-task metrics

| issue | effect | status |
|---|---|---|
| `stsb` scored by exact string match on a float regression target | **every stsb value exactly 0.000** | Fixed (Pearson). Old values unrecoverable → all affected combinations re-run; now complete |
| `squad_v2` scored by naive exact match | 0.7212 vs correct EM 0.8258 — **10.5 points understated** | Fixed (official SQuAD v2 metric) |
| `squad_v2` reported on a 0–100 scale | would dominate the mean used to pick the shared λ, collapsing the choice to "whatever suits squad_v2" | Normalised to 0–1 |
| single-task ceilings read from the trainer's naive per-row EM | inflated retention: `record` ceiling 0.5097 vs correct 0.7878 | Fixed — all summaries read corrected metrics |

**Three of the 24 tasks need a task-specific metric** — `record`, `squad_v2`,
`stsb` — and the generic exact-match path produces a plausible-looking wrong answer
for each: a silent zero, a 10-point understatement, and a scale error that would
have hijacked coefficient selection. The corrected implementations were validated
to 1e-9 against the reference scorers on the full evaluation sets.

### 5.2 Stale configuration — the more dangerous class

| issue | effect | status |
|---|---|---|
| combination names split on `_` | `anli_r1` parsed as `anli`+`r1`; wrong mixture size on 1026 of 2036 combinations, and broke the SAR task-vector lookup outright | Fixed; round-trips all names |
| the GP's subset-indicator vocabulary hardcoded the old 5 sources | every mixture built from one of the 7 new sources encoded as the **all-zero vector**, collapsing unrelated mixtures onto identical GP inputs | Fixed — read from the sweep's own task list; unknown tasks now raise |
| BO silently swallowed GP-fitting failures and fell back to random | indistinguishable in the output from "BO does not beat random" — the exact claim at stake | Fixed — now warns; the reported run had **0 fallbacks in 43,200 fits** |
| figure grids hardcoded 2×2 for what became 12 targets | silently plotted 4 targets and dropped 8; a 4-entry colour map raised `KeyError` on the rest | Fixed — grids size to the target list |

**The generalisable lesson.** A metric that returns 0.000 for every input looks like
a failed merge, not a failed metric. An all-zero feature vector looks like a
converged GP. A fallback looks like a null result. Nothing in the pipeline flags any
of them. The detections that worked were cheap and worth recommending as standard
practice: check whether a metric column is *constant* across a sweep it has no
business being constant across; assert that a resumed or paired computation actually
matches what it claims to match; and make silent fallbacks warn.

### 5.3 One definitional caveat to state in the paper

`src/utils.py:alignment_ratio` computes `‖S_proj‖₂ / ‖S_τ‖₂` where both arguments
are **1-D vectors of singular values**. For 1-D input `np.linalg.norm(v, ord=2)` is
the Euclidean norm, which equals the **Frobenius** norm of the matrix those singular
values came from — not its spectral norm. So SAR as implemented is *the fraction of
the task vector's total energy retained by the projection*, summed over all
directions, rather than a ratio of largest singular values. Every SAR number in this
brief is that energy-fraction variant. If a spectral-norm ratio was intended — a
natural reading of Marczak et al.'s Eq. 5 — then the implementation deviates from
the reference and the paper should say which quantity it reports. Exploiting this
identity also made the computation 72× faster, since both inner SVDs were redundant.

---

## 6. What is NOT yet supportable

Do not write these without new data:

- **Retention or interference on the source tasks.** The source sweep is 634 / 4083
  (§7). Nothing about how merging affects the *sources* is supportable yet.
- **In-mixture SAR paired with NAI.** `figures/nai_sar.csv` is from the withdrawn
  5-source phase and joins to zero rows of the current grid; the 12-source version
  needs the source sweep.
- **Any `base` / `lora` / `freeze` comparison.** Only `base` exists at 12 sources,
  and the earlier comparison is withdrawn (§8).
- **Anything requiring variance.** Single seed, single model. No error bars exist
  anywhere, so no comparison can be called statistically significant. This is the
  single biggest reviewer risk in the work.
- **Bit-identical reproduction of the BO numbers.** BLAS thread count perturbs the
  last floating-point digits of the GP fits, which can flip an EI argmax; against an
  earlier run this moved the aggregate by one run in 120. Rankings are unaffected.

---

## 7. Open numbers and pending runs

| item | state |
|---|---|
| target sweep | **complete**, 4083 / 4083 |
| stsb re-scoring | **complete**, 0 outstanding |
| target SAR (12-source) | **complete**, 48,996 rows, no NaN |
| BO feature comparison (12-source) | **complete**, 10 seeds, 63-evaluation budget |
| source sweep | **634 / 4083 running** — see the timeline below |
| in-mixture SAR + NAI at 12 sources | `PENDING`, blocked on the source sweep |
| multiple seeds / larger model | `PENDING`, not scheduled |

**Timeline for the source sweep — this is much slower than the target sweep, and
the reason is worth understanding.** Measured throughput is ~5 combinations/hour and
falling, projecting **~50 more days** (late October). Two causes:

1. **The source tasks are 8.2× larger than the targets** (§2). Every combination
   evaluates its *own* tasks, so an average 6-task mixture is ~69,500 examples
   against the target sweep's fixed 16,908 — **4.1× more total work** (11.4 B
   generations vs 2.76 B). The target sweep took 17 days; 4.1× that is ~70 days.
2. **Progress by combination count overstates progress by work.** The sweep
   enumerates smallest-first, so 634/4083 combinations (15.5%) is only **8.9% of the
   task-evaluations**. Extrapolating from combination count will keep
   under-predicting; the observed rate falls from ~7/h to ~2.4/h through the 6–7
   task mixtures where most of the grid lives.

Two levers exist if the timeline becomes a problem, both declined so far: capping
eval samples per source task (**67% of the cost is in four tasks** — qqp 29%,
imdb 18%, record 11%, squad_v2 8.5%; a 2,000-example cap is roughly a 7× speedup at
about ±1% noise), and replacing the sweep wrapper's chunk barrier with a rolling
window (~2×, no change to any number).

---

## 8. `WITHDRAWN`: the earlier 5-source study

A previous phase used 5 sources (`mnli, qnli, qqp, sst2, record`), 4 targets, three
methods (`base`/`lora`/`freeze`) and 26 combinations. **It must not be cited**, for
two independent reasons found in a 2026-08-25 audit:

1. **Its numbers do not match its own artifacts, and one headline reverses.** The
   write-up reported `base` winning worst-case per-task retention 16/26 with
   min-retention 0.627 vs LoRA's 0.572. `figures/merged/merge_summary.csv` gives
   **LoRA 12 wins, base 7, freeze 7**, with min-retention **base 0.684, LoRA 0.729**.
   The mean-vs-min trade-off — a headline claim — points the other way. Every
   aggregate in that section disagreed with the file it cited, and the numbers match
   neither of the two surviving per-model snapshots.
2. **Its source sweeps have been deleted** and `saves*` is gitignored, so the claims
   cannot be re-derived or audited. `combo_status.py source` reports all 4083
   missing.

Two of its other reported findings are also unsupportable as stated: the `record`
retention percentages used a single-task ceiling of 0.5097 that the metric fix
raised to 0.7878, so every retention ratio involving `record` had the wrong
denominator; and the four-target transfer table was `lora`-only, with NAI values
(`mrpc` 0.934, `rte` 1.000) that under `base` become 0.369 and 0.855.

**What survives, and is worth keeping as motivation:** task-vector geometry does not
predict merge accuracy — body cosine similarities are uniformly ~0.02–0.08, i.e.
near-orthogonal, yet individual tasks are still annihilated by merging. If geometry
predicted merge quality, subset selection would be solved and no search would be
needed. That claim rests on the Gram/cosine analysis rather than on the disputed
retention aggregates, but **it should be recomputed on the 12-source `base` grid
before it goes in the paper.**

---

## 9. Framing suggestions

- **Lead with the negative-plus-positive pair.** "Alignment scores do not predict
  which mixture is best, but the search machinery works and plain subset indicators
  are a strong baseline" is more honest and more interesting than a pure method win.
- **The scaling result is the story.** Random search looks competitive at 26
  mixtures and collapses at 4083; the indicator encoding survives the 157× increase.
  Both are statements about *search-space size*, which is what makes the 12-source
  grid worth having built.
- **The `cola` result is an asset, not an embarrassment.** A target where merging
  demonstrably cannot help bounds what any selection method can promise.
- **The measurement section is publishable on its own terms.** Three of 24 tasks
  needing bespoke metrics, plus a class of stale-configuration bugs that produce
  converged-looking output, is a concrete reproducibility contribution for the
  model-merging literature.
- **Be explicit about the single seed**, once and prominently, rather than letting a
  reviewer find it.
- **State the tie-handling rule** before any top-1 recovery number (§4.5).
- **Say which SAR you report** (§5.3) — energy fraction, not spectral ratio.
