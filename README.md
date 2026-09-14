# Bayesian Task Selection

Which **source** tasks should you merge, to transfer best to a held-out **target**
task? We fine-tune each source task separately, merge subsets of their task
vectors by task arithmetic, and evaluate the merged model on targets it never saw.
The full transfer grid is the ground truth a selection method has to predict; a GP
+ Expected Improvement loop is the selection method under test.

Research findings live in [`reports/findings.md`](reports/findings.md). This file
is the operational map: what state the pipeline is in, what each script does, and
how to run it.

Model `llama-3.2-1b-instruct`, seed 42, method `base` (full fine-tuning).

## Status

| | |
|---|---|
| Source tasks | 12 — `mnli qnli qqp sst2 record snli anli_r1 paws imdb squad_v2 hellaswag winogrande` |
| Target tasks | 12 — `mrpc boolq rte cola cb scitail stsb cr rotten_tomatoes multirc copa piqa` |
| Combinations | 4083 (all subsets of size 2..12) |
| Coefficient grid | 40 points, 0.05 → 2.0 (`COEF_MAX`, `N_EVAL_POINTS` in `eval_target.py`) |

Source sweeps and target sweeps are **complete** — all 4083 combinations wrote a
40-row CSV. Target SAR and the GP simulation were still calibrated for an earlier
5-source / 26-combination / 4-target version of this experiment and were rebuilt
in Aug 2026; see *History* below.

## Pipeline

Stages run in order — each reads the previous one's output from `saves_*`.

| # | Stage | Script | Output |
|---|---|---|---|
| 0 | Pretrained snapshots (θ_pre, the task-vector origin) | `scripts/pretrained_weights.sh` | `saves_pretrained_weights/` |
| 1 | Single-task fine-tunes + evals | `scripts/slurm/preliminary_{source,target}.sh`, `zero_shot.sh` | `saves_bts_preliminary/` |
| 2 | Per-eval metrics + reference tables | `scripts/compute_metrics.py` | `compute_metrics.jsonl`, `results.csv`, `results.tex` |
| 3 | Source coefficient sweeps | `src/eval/eval.py` via `scripts/slurm/eval_array.sh` | `saves_bts_merged/**/…_best/…_acc_coef.csv` |
| 4 | Target transfer sweeps | `src/eval/eval_target.py` via `scripts/slurm/eval_target_array.sh` | `saves_bts_merged/**/…_target/…_target_acc_coef.csv` |
| 5 | Target SAR (geometry signal) | `scripts/slurm/target_sar_array.sh` | `figures/target/target_sar.csv` |
| 6 | Transfer grid (the GP's training data) | `scripts/summarize_target.py` | `figures/target/target_summary.csv` |
| 7 | Correlations + GP simulation | `scripts/slurm/analysis.sh` | `figures/analysis/`, `figures/bo_variants/` |

## Scripts

**Task lists live in exactly one place:** `SOURCE_TASKS` / `TARGET_TASKS` in
`src/eval/eval_target.py`. Everything else reads them from there via
`scripts/combo_names.py` (by AST, so the analysis scripts never import torch).
Do not restate a task list anywhere else — every stale-list bug in this repo's
history came from a second copy.

### Evaluation (GPU)
- **`src/eval/eval.py`** — merges a combo, sweeps the coefficient, scores the
  combo's *own* tasks. Combination index comes from `$SLURM_ARRAY_TASK_ID`.
- **`src/eval/eval_target.py`** — same merges, scored on the 12 held-out targets.
  Owns the canonical task lists and sweep constants.
- **`scripts/slurm/run_eval_batched.sh`** — submits either array in chunks, since
  4083 exceeds both `MaxArraySize` and the queued-job cap.
- **`scripts/combo_status.py`** — which combinations actually produced a CSV.
  Needed because `sacct` is unusable here (see *Cluster notes*).
  `--require-nonzero COL` also catches CSVs written before a metric fix.

### Aggregation (CPU)
- **`scripts/summarize_target.py`** — reads every target sweep CSV, takes each
  combo's best coefficient *per target*, computes NAI, joins SAR. Writes
  `target_summary.csv` (the GP's input), plus the matrix and LaTeX tables.
- **`scripts/compute_target_sar.py`** — GPU. Subspace Alignment Ratio of each
  target's task vector against each mixture's summed subspace. Supports
  `--start/--limit` (sharding) and `--resume`.
- **`scripts/combo_names.py`** — parses a `mnli_anli_r1_paws` directory stem back
  into task names. A naive `split("_")` breaks on `anli_r1` / `squad_v2`.
- **`scripts/reference_scores.py`** — zero-shot and single-task-FT reference
  accuracies (NAI's floor and ceiling).

### GP / Bayesian optimization
- **`src/bayesian_task_selection.py`** — the primitives. `fit_ei` (a
  `SingleTaskGP` + Expected Improvement), `simulate_bo_on_grid` (retrospective BO
  over already-evaluated combos — no training runs), `simulate_random_on_grid`,
  `simulate_greedy_by`, `combo_to_binary`.
- **`scripts/analysis_bo_variants.py`** — **the GP simulation.** Which GP feature
  set finds high-transfer mixtures fastest: SAR, iso-SAR, either normalized by
  mixture size, the binary subset indicator, or indicator + iso-SAR — each against
  a random-search baseline.
- **`scripts/analysis_correlations.py`** — does SAR predict transfer at all, and
  how does a SAR-greedy ranking compare to BO.

## Running it

`conda activate` is broken on this cluster, so call the env interpreter directly:

```bash
PY=/mnt/home/$USER/miniconda3/envs/pf/bin/python
```

Stages 0–4 are done; you only need 5–7 unless the task lists change.

```bash
# 5. target SAR — 48 shards x ~86 combos, ~35 min per shard, ~2 h wall
sbatch --array=0-47 scripts/slurm/target_sar_array.sh
bash scripts/slurm/target_sar_array.sh --merge     # -> figures/target/target_sar.csv

# 6. the transfer grid (joins the SAR columns)
PYTHONPATH=src $PY scripts/summarize_target.py --model llama-3.2-1b-instruct --seed 42

# 7. correlations + GP simulation (CPU)
mkdir -p logs_analysis && sbatch scripts/slurm/analysis.sh
```

The merge step refuses to write unless it sees exactly `4083 x 12 = 48996` rows, so
a missing shard cannot leak into the summary. Re-run one shard with
`sbatch --array=<n> scripts/slurm/target_sar_array.sh`; `--resume` skips what it
already finished.

To run the GP simulation alone, with a budget suited to the grid size:

```bash
PYTHONPATH=src $PY scripts/analysis_bo_variants.py --method base \
    --n-initial 3 --n-iterations 60 --seeds 5 --out figures/bo_variants
```

**Pick the budget deliberately.** The default `--n-iterations 10` is 13 evaluations
total, which was half the search space back when there were 26 combinations and is
0.3% of it now. At that budget every method — BO and random alike — is mostly
sampling noise, and the §11 result that binary features win 100% of the time should
not be assumed to carry over.

## Cluster notes

- **`conda activate pf` fails.** The miniconda install has its pre-remount prefix
  (`/mnt/data/home/$USER`) baked into `conda.sh` and `bin/conda`'s shebang. The
  env's interpreter is a plain ELF binary needing no activation — call
  `/mnt/home/$USER/miniconda3/envs/pf/bin/python` directly. Paths moved in Aug
  2026: `/lustre/scratch` → `/mnt/scratch`, `/mnt/data/home` → `/mnt/home`.
- **Array size caps at ~48.** A wider `--array` is *rejected* at submit time with
  `AssocMaxSubmitJobLimit`. Separately, at most **15** array tasks run at once.
  So: ≤48 shards, each doing a slice, ~15 in flight.
- **`sacct` / `sacctmgr` do not work** — slurmdbd is unreachable
  (`_open_persist_conn: Connection refused`), so job state is gone once a task
  leaves the queue, and QOS limits cannot be queried. Check whether work
  *succeeded* by looking for its output file, which is what `combo_status.py` does.
- **Set `HF_HUB_OFFLINE=1` / `HF_DATASETS_OFFLINE=1`** in GPU jobs. All datasets
  are cached under `$HF_HOME`; without these, concurrent tasks re-resolve them
  against the Hub and some get a 429 several GPU-minutes in.
- **Jobs cannot read `/tmp` from the submitting node.** Stage anything a job needs
  at runtime on a shared filesystem.

## History

Two scaling changes are worth knowing about, because most of the older code and
`reports/findings.md` §1–§11 predate them:

1. **Sources grew 5 → 12, targets 4 → 12** (26 → 4083 combinations). Anything that
   hardcoded the old lists silently produced wrong output rather than failing:
   `combo_to_binary` encoded the 7 new sources as all-zero vectors, a 2×2 subplot
   grid plotted 4 of 12 targets, and a 4-entry color dict raised `KeyError`.
2. **SAR was ~72x slower than it needed to be.** `alignment_ratio` receives 1-D
   singular-value arrays, and `np.linalg.norm(v, ord=2)` on a 1-D input is the
   Euclidean norm — i.e. exactly the Frobenius norm of the matrix the spectrum came
   from. Both per-probe SVDs were therefore computed and discarded, 24 of the 25
   decompositions per (matrix, mixture). `utils._sar_modes` now takes those norms
   directly, never forms the projection at full size (`‖U_k U_kᵀ·tv‖_F ==
   ‖U_kᵀ·tv‖_F` for orthonormal `U_k`), and `utils.sar_both` shares the one
   remaining SVD between the raw and iso modes. Measured on real task vectors:
   1650 s → 23 s per combination, i.e. 78 days → 26 h for the full grid, and ~2 h
   sharded. Values agree with the original to float32 roundoff (1e-15 in float64).
