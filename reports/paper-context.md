# Bayesian Task Selection for Task-Vector Merging — Work Summary

*Context document for drafting a paper. Llama-3.2-1B-Instruct, seed 42.
Snapshot: 2026-07-30, with the target sweep still running.*

**How to read this.** Every number below is tagged with its status. `VALIDATED`
means computed from artifacts on disk and checked this session. `PARTIAL` means
computed from an incomplete sweep and safe only as a direction, not a headline.
`INVALID` means known-wrong data that must not enter the paper. `PENDING` means
not yet computed. A prior phase of this project is written up separately in
`reports/findings.md` (474 lines, the 5-source study) — this document covers the
current 12-source study and supersedes that setup, not those findings.

---

## 1. Research question

Given a pool of source tasks with fine-tuned task vectors, **which subset should
be merged to maximise performance on a target task** — and can that subset be
found without exhaustively evaluating every mixture?

The exhaustive ground truth is what the current sweep builds: every source subset
× every target, so any proposed selection method can be scored against the true
optimum. The selection method under study is Bayesian optimisation over the
subset space, with the open question being what the GP should take as input
features.

---

## 2. Experimental setup

**Model.** `meta-llama/Llama-3.2-1B-Instruct`, seed 42, template `llama3`,
cutoff 2048.

**Fine-tuning** (`config_templates/base/llama-3.2-1b-instruct/train.yaml`): full
fine-tuning (`finetuning_type: full`), 1 epoch, per-device batch 8, lr 2e-6,
cosine schedule, warmup ratio 0.1, weight decay 1e-2, AdamW, bf16, val_size 0.1.
Method label for this study is `base`. (`lora` = rank-8 on `k_proj,v_proj`, α=8,
and `freeze` = full FT with embeddings/tied `lm_head`/final norm frozen, both
used in the earlier phase.)

**Data.** All tasks come from the `kinit/peft-factory` HF dataset. Eval-set sizes
vary by three orders of magnitude, which matters for how much weight a per-task
result can carry: `cb` 56, `copa` 100, `stsb` 1500, `mnli` 9832, `squad_v2`
11873, `record` 15176.

**Source tasks (12)** — `mnli, qnli, qqp, sst2, record, snli, anli_r1, paws,
imdb, squad_v2, hellaswag, winogrande`

**Target tasks (12)** — `mrpc, boolq, rte, cola, cb, scitail, stsb, cr,
rotten_tomatoes, multirc, copa, piqa`

Source and target sets are disjoint. Declared in `src/eval/eval_target.py`.

**Merging.** Task arithmetic: sum the per-task vectors, scale by a single shared
coefficient λ, evaluate. The sweep covers λ ∈ {0.05, 0.10, …, 2.00} — 40 points,
`np.linspace(0, COEF_MAX, N_EVAL_POINTS)[1:]` with `COEF_MAX=2.0`,
`N_EVAL_POINTS=41`; the leading 0 is dropped because λ=0 is the pretrained model.

**Two sweeps, different questions:**

| sweep | script | output dir | merged model evaluated on | question |
|---|---|---|---|---|
| source | `src/eval/eval.py` | `*_42_best/` | the sources it merged | retention / interference |
| target | `src/eval/eval_target.py` | `*_42_target/` | 12 held-out targets | transfer |

Combination space: all subsets of size 2..12 of the 12 sources = **4083 combos**,
each evaluated at 40 coefficients on all 12 targets.

**Metrics.** Per-task metric from `scripts/compute_metrics.py`
(`DATASET_TO_METRIC_MAPPING`): exact match for classification, macro-F1 variants
where the label set is >2, ReCoRD's grouped-by-`idx` metric for `record`, SQuAD
v2 EM for `squad_v2`, Pearson/Spearman for `stsb`. The headline column is
exact_match, falling back to Pearson for regression tasks.

**Reference points.** `zero_shot` (pretrained) and `single-task FT` are on disk
for all 24 tasks under `saves_bts_preliminary/`. NAI (normalized accuracy
improvement, Marczak et al.) = (merged − zero_shot) / (finetuned − zero_shot);
1.0 means the merge matches fine-tuning directly on the target.

---

## 3. Pipeline

| stage | script | artifact |
|---|---|---|
| coefficient sweep, source | `src/eval/eval.py` | `saves_bts_merged/*/*/*_best/*_acc_coef.csv` |
| coefficient sweep, target | `src/eval/eval_target.py` | `saves_bts_merged/*/*/*_target/*_target_acc_coef.csv` |
| sweep progress | `scripts/combo_status.py` | — |
| source summary | `scripts/summarize_merged.py` | `merge_summary.csv` |
| target summary + winners | `scripts/summarize_target.py` | `target_summary.csv`, `target_matrix.csv`, `target_best_per_target.csv` (+ `.tex`) |
| coefficient distribution | `scripts/coef_distribution.py` | `coef_distribution.csv` |
| subspace alignment | `scripts/compute_target_sar.py` | `target_sar.csv` |
| SAR correlations | `scripts/analysis_correlations.py` | `figures/analysis/` |
| BO feature comparison | `scripts/analysis_bo_variants.py` | `figures/bo_variants/` |

Shared helpers: `scripts/combo_names.py` (parses `_`-joined combo names, reads
sweep constants from `eval_target.py` without importing torch),
`scripts/reference_scores.py` (single definition of the zero-shot / fine-tune
reference lookup). All analyses run on CPU via SLURM `cpu_short`.

---

## 4. Phase 1 — the 5-source study (`reports/findings.md`)

Sources `mnli, qnli, qqp, sst2, record`; methods `base`/`lora`/`freeze`; 26
combos. Headline results, all `VALIDATED` on that data:

1. Embedding amplification in full-FT merges is real but **does not explain merge
   quality** — freezing embeddings removes the amplification and changes accuracy
   essentially not at all.
2. **No universal best method.** LoRA wins on mean accuracy (13/26), base wins on
   worst-case per-task retention (16/26).
3. **`record` breaks LoRA merges** — LoRA retains 2–21% of its single-task
   accuracy whenever ReCoRD is in the mixture, base 38–53%.
4. **Task-vector geometry does not predict merge accuracy** — body cosines are
   uniformly ~0.02–0.08 (near-orthogonal) yet individual tasks are still
   annihilated.
5. Target transfer is real and **mixture-dependent**: each target has a different
   best mixture — the ground truth a selection method must predict.
6. **SAR alone does not select the right mixture, but the BO machinery works.**
   Raw target-SAR is near-uncorrelated with transfer accuracy (ρ ≈ −0.15…+0.32)
   and grows with mixture size, so its argmax is always the full set. Iso-SAR is
   the only alignment score with signal (cola ρ +0.66, boolq +0.47). In
   retrospective simulation, **plain binary subset features beat every SAR
   feature set** (100% win rate, ~5.6 evaluations to within 1% of the true best);
   raw SAR as a GP feature is *worse than random*.

**Caveat for the paper:** phase 1 used 5 sources / 4 targets and mixed methods.
Nothing in it has been reconfirmed at 12 sources. Point 6 is the closest thing to
a positive result and is the natural thing to re-run first on the new grid.

---

## 5. Phase 2 — the 12×12 transfer study (current)

### 5.1 Status

`PARTIAL` — **2102 / 4083 combos (51%)** written as of this snapshot; the source
sweep has **0** and has not started. Coverage is strongly skewed by arity,
because the array indexes combos in arity order:

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| done | 100% | 100% | 70% | 58% | 50% | 41% | 33% | 25% | 17% | 8% | 0% |

**This is the single most important caveat for writing anything up now.** Small
mixtures are complete, large ones barely sampled, so any statement of the form
"the best mixture has k tasks" or "accuracy declines with mixture size" is
confounded with sampling density. Statements conditioned *within* an arity are
safe; aggregates across arities are not.

### 5.2 Transfer is task-family aligned

`PARTIAL` (n=2069 combos), but the cleanest evidence that the pipeline measures
something real rather than noise:

| target | best acc | best source mixture | NAI |
|---|---|---|---|
| copa | 0.770 | hellaswag+winogrande | 1.044 |
| piqa | 0.721 | imdb+hellaswag+winogrande | 0.820 |
| rotten_tomatoes | 0.929 | sst2+record+imdb | 1.011 |
| cr | 0.934 | sst2+record+anli_r1 | 0.996 |
| scitail | 0.865 | mnli+qqp+anli_r1+hellaswag | 0.899 |
| rte | 0.838 | mnli+qnli+qqp+sst2+anli_r1+paws+imdb+winogrande | 1.110 |
| multirc | 0.757 | mnli+qnli+anli_r1+paws | 0.766 |
| boolq | 0.699 | qnli+record+paws+hellaswag+winogrande | 0.707 |
| mrpc | 0.684 | record+hellaswag | 0.805 |
| cola | 0.674 | record+imdb+hellaswag | **0.057** |
| cb | 0.893 | mnli+sst2+snli | 1.250 |

Commonsense targets select commonsense sources, sentiment selects sentiment, NLI
selects NLI. `cola` is the informative negative: NAI 0.057 means merging buys
essentially nothing over the pretrained model, consistent with grammaticality
being idiosyncratic. NAI > 1 (copa, rotten_tomatoes, rte, cb) means the merge
beats fine-tuning directly on the target — plausible on low-ceiling targets, and
worth a sentence of explanation rather than being buried.

### 5.3 Small evaluation sets produce large ties

`VALIDATED`. `cb` has **42 combinations tied at exactly 0.892857** = 50/56
examples. Accuracy on a 56-example set is quantised to ~1.8% steps, so "the best
mixture for cb" is not identifiable — an arbitrary tie-break silently picks one.
`mrpc` has 2 tied. `summarize_target.py` now breaks ties toward the smaller
mixture and reports the tie count.

**Implication for the paper:** any "our method recovered the optimal subset"
claim must state how ties were handled, and `cb` (and probably `copa`, n=100)
should be excluded from or footnoted in top-1 recovery rates.

### 5.4 The merge coefficient concentrates far below the swept range

`PARTIAL` (n=22,759 target rows). Distribution of the winning λ:

| interval | share | cumulative |
|---|---|---|
| [0.05, 0.10) | 36.0% | 36.0% |
| [0.10, 0.20) | 42.4% | 78.4% |
| [0.20, 0.30) | 9.9% | 88.3% |
| [0.30, 0.50) | 5.2% | 93.5% |
| [0.50, 0.75) | 3.3% | 96.7% |
| [0.75, 1.00) | 2.1% | 98.8% |
| [1.00, 1.50) | 1.1% | 100.0% |
| [1.50, 2.00] | 0.03% | 100% |

Observed range 0.05–1.80; **nothing ever peaks above 1.80**, so the top half of
the grid (20 of 40 points, ~half the compute per combo) has never produced a
winner.

36% of all rows peak at the smallest λ evaluated, but this is **not** clipping of
good results: among each target's *top-decile* combos only 7.6% sit there, versus
36% overall. Low-λ pinning is concentrated in mixtures that don't transfer, where
the best move is trivially "merge as little as possible" (bottom-decile combos
for boolq, mrpc, rte average λ = 0.050 exactly). Among top-decile combos the
percentiles are p50 0.200, p90 0.600, p99 0.850, max 1.250.

Per-target behaviour differs sharply — `boolq` has 85.8% of combos at the grid
minimum with a max of 0.75, while `cb` has 0.5% pinned and a median of 0.60 — so
a single shared λ range does not suit all targets.

**Recommendation on file:** `COEF_MAX = 1.25` covers every good combo with
headroom; keeping `N_EVAL_POINTS = 41` gives a 0.031 step (60% finer resolution
at identical cost). **Not applied**, because changing the grid mid-sweep makes
`best_coef` an argmax over a different support and breaks comparability with the
2102 combos already written.

---

## 6. Measurement errors found and fixed

These matter for the paper because two of them silently corrupted numbers that
looked plausible.

| issue | effect | status |
|---|---|---|
| `stsb` scored by exact string match on a float regression target | **every stsb value 0.000** across all combos | Fixed at source (`ComputeCorrelation`, Pearson, `src/metrics.py`); old values unrecoverable, so the 2041 affected combos are being **re-run** — see below |
| `squad_v2` scored by naive exact match | 0.7212 vs correct EM 0.8258 — **10.5 points understated** | Fixed (`ComputeSquadV2`); source sweep had not started, so no data lost |
| `squad_v2` metric on a 0–100 scale | would dominate the mean used to pick the shared coefficient | Normalised to 0–1 at source and on read |
| single-task ceilings read from `predict_results.json` (trainer's naive per-row EM) | inflated retention: `record` ceiling 0.5097 vs correct 0.7878, `squad_v2` 0.7356 vs 0.8258 | Fixed — all summaries read `compute_metrics.jsonl` |
| combo names split on `_` | `anli_r1` → `anli`+`r1`; wrong `n_tasks` on **1026 of 2036** combos, and would have made SAR computation fail outright | Fixed (`scripts/combo_names.py`, round-trips all 2036 names) |

`ComputeCorrelation` and `ComputeSquadV2` were validated to 1e-9 against
`scripts/compute_metrics.py` on the full eval sets (1500 and 11873 examples).

**stsb is currently a mixed column** — combos evaluated before the fix hold 0.0,
after it hold real Pearson values. `summarize_target.py` drops any target whose
column is mostly exact zeros and reports the share, so it rejoins automatically
once re-run. **stsb must still be excluded from all analysis until the re-run
below completes**, since no per-coefficient predictions are saved and post-hoc
rescoring is therefore impossible.

### stsb re-run (in progress, as of 2026-08-10)

The affected combinations are identified by the **zeros themselves**, not by file
timestamps: a pre-fix CSV has stsb `0.0` at all 40 coefficients, which the fixed
metric does not produce. The test is exact — checked against when
`ComputeCorrelation` landed (2026-07-30 12:29), the split was 2041 all-zero
before it and 0 after, out of 3508 CSVs then on disk.

| | combinations |
|---|---|
| total in the target grid | 4083 |
| stale stsb (all-zero), found 2026-08-05 | 2041 |
| never produced a CSV at all | 575 |
| valid at the start of the re-run | 1467 |

`scripts/combo_status.py --require-nonzero stsb` reports a combination as
outstanding when its CSV is missing **or** its stsb column is all zeros, and
`REQUIRE_NONZERO=stsb run_eval_batched.sh` re-runs exactly that set. The eval
overwrites the CSV in place, so nothing is deleted first, and the per-chunk
verification uses the same rule — a combination that came back still all-zero
would be reported as a failure rather than silently accepted.

Progress at the time of writing: **1818 of 4083 valid, 2265 outstanding**. The
re-run recomputes all 12 target columns per combination (the sweep evaluates
every target at every coefficient), so the other columns are refreshed too, at
no extra cost. Expect roughly 5-6 days at the cluster's sustained ~15
combinations/hour; `scripts/sweep_status.sh` reports live progress and a
measured rate.

Note the re-run does **not** make the other four issues in the table above go
away, and it does not touch the source sweep, which has not started.

---

## 7. Not yet valid / pending

- `PENDING` **SAR for this sweep.** `figures/target/target_sar.csv` is from the
  5-source/`lora` phase and joins to zero rows of the current grid.
  `compute_target_sar.py` needs a GPU re-run (2102 combos × 12 targets). Until
  then `analysis_correlations.py` has nothing to correlate.
- `PENDING` **BO feature comparison on 12 sources.** `analysis_bo_variants.py`
  additionally needs `SOURCE_TASKS` in `src/bayesian_task_selection.py:26`
  updated from the stale 5-task list — `combo_to_binary` currently drops the 7
  new sources and collapses distinct combos to identical feature vectors.
- `PENDING` **source sweep** (retention/interference at 12 sources): 0 of 4083.
- `IN PROGRESS` **stsb re-run** (§6): 1818 of 4083 combinations valid, 2265
  outstanding as of 2026-08-10. Still `PENDING` for the purpose of any claim —
  the column stays mixed until the last one lands.
- `PENDING` **`freeze` reference metrics** — `compute_metrics.py` only iterates
  `peft_methods = ["lora","base","zero-shot"]`, so `freeze` has no
  `compute_metrics.jsonl` at all, though the predictions exist. Any
  base/lora/freeze comparison needs this first, or `freeze` gets scored by a
  weaker metric than the others and appears to retain better than it does.
- Single seed (42), single model (1B). No variance estimates anywhere.

---

## 8. What can be claimed today

Safe with the current data:

- The transfer ground truth is real, mixture-dependent, and semantically coherent
  (§5.2) — supports the framing that subset selection is a genuine problem with a
  non-trivial answer.
- Selection matters: within a target, mixtures range from NAI ≈ 0 to NAI > 1.
- The merging protocol's shared coefficient lives in a narrow low range (§5.4),
  which is a practical contribution in its own right.
- The measurement pitfalls in §6 are worth a methods paragraph: three of the 24
  tasks need task-specific metrics, and the default exact-match path silently
  produces plausible-looking zeros and understatements.

Not yet supportable:

- Anything comparing mixture sizes across arities (sampling bias, §5.1).
- Any claim about SAR or BO on the 12-source grid (§7).
- Any claim involving stsb — the re-run is under way but the column is mixed
  until it finishes (§6).
- Any base/lora/freeze comparison at 12 sources (§7).
