#!/bin/bash
# Submit an eval array in batches that respect the cluster limits:
#   MaxArraySize = 1001         -> array index can never exceed 1000
#   ~48 queued jobs per user    -> at most 48 array tasks in flight
#
# The eval scripts map the global combination index to
#   COMBO_OFFSET + SLURM_ARRAY_TASK_ID
# so each batch submits BATCH combinations with a per-batch COMBO_OFFSET, waits
# for that batch to finish, then submits the next one.
#
# After each batch drains it checks which combinations actually wrote their
# acc_coef CSV and stops if any didn't, printing the failed combo indices and a
# resume command. The check is on-disk output, NOT sacct: slurmdbd is unreachable
# on this cluster ("_open_persist_conn: Connection refused"), so sacct returns no
# records and silently green-lit 2062 crashed combinations on the first full run.
#
# Run this on the login node (it only submits + polls squeue; no compute):
#   scripts/slurm/run_eval_batched.sh eval_array.sh          [TOTAL] [BATCH]
#   scripts/slurm/run_eval_batched.sh eval_target_array.sh   [TOTAL] [BATCH]
# Resume from a given offset after a failure:
#   START=1488 scripts/slurm/run_eval_batched.sh eval_array.sh
# Re-run ONLY the combinations still without a CSV (gap-filling after failures);
# already-finished combinations are never resubmitted:
#   MISSING=1 scripts/slurm/run_eval_batched.sh eval_target_array.sh
# Also re-run combinations whose CSV predates a metric fix, identified by that
# metric's column being all zeros (see combo_status.py --require-nonzero). The
# job overwrites the CSV in place, so nothing needs deleting first:
#   MISSING=1 REQUIRE_NONZERO=stsb scripts/slurm/run_eval_batched.sh eval_target_array.sh
# Backgrounding is recommended:  nohup scripts/slurm/run_eval_batched.sh ... &

set -euo pipefail

JOB_NAME=${1:?"pass the job script name: eval_array.sh or eval_target_array.sh"}
JOB_SCRIPT="scripts/slurm/${JOB_NAME}"
TOTAL=${2:-4083}   # combinations of size >=2 for 12 tasks
BATCH=${3:-48}     # max jobs allowed in the queue at once
POLL=${POLL:-60}   # seconds between queue checks
START=${START:-0}  # first combination offset (set when resuming after a failure)
MISSING=${MISSING:-0}  # 1 = submit only combinations that have no CSV yet
REQUIRE_NONZERO=${REQUIRE_NONZERO:-}  # space-separated columns that must not be all-zero
STATUS="scripts/combo_status.py"
MAX_ARRAY_INDEX=1000

# Passed to every combo_status.py call, so the "what still needs running?" list
# and the post-chunk "did it work?" check always apply the same definition of
# done. A combination that comes back with the column still all-zero is then
# reported as a failure rather than silently accepted.
status_args=()
for col in $REQUIRE_NONZERO; do status_args+=(--require-nonzero "$col"); done

[[ -f "$JOB_SCRIPT" ]] || { echo "No such job script: $JOB_SCRIPT" >&2; exit 1; }
[[ -f "$STATUS"     ]] || { echo "No such helper: $STATUS" >&2; exit 1; }

# --- Which global combination indices are we submitting? -----------------------
# Either a contiguous run (the normal full sweep) or just the gaps (MISSING=1).
indices=()
if (( MISSING )); then
    mapfile -t indices < <(python3 "$STATUS" "$JOB_NAME" --start "$START" "${status_args[@]}" || true)
    if (( ${#indices[@]} == 0 )); then
        echo "Nothing to do: every combination from ${START} on already has a usable CSV."
        exit 0
    fi
    echo "Gap-fill mode: ${#indices[@]} combination(s) still need running."
    if [[ -n "$REQUIRE_NONZERO" ]]; then
        echo "  (an all-zero ${REQUIRE_NONZERO} column counts as not-yet-run)"
    fi
else
    for (( i=START; i<TOTAL; i++ )); do indices+=("$i"); done
fi

n_total=${#indices[@]}
echo "Batched submit: ${JOB_NAME}  combinations=${n_total}  BATCH=${BATCH}"
echo "Batches: $(( (n_total + BATCH - 1) / BATCH ))"
echo

# --- Submit in chunks of BATCH -------------------------------------------------
for (( c=0; c<n_total; c+=BATCH )); do
    chunk=("${indices[@]:c:BATCH}")
    off=${chunk[0]}
    last=${chunk[-1]}
    span=$(( last - off ))

    # Array indices are relative to COMBO_OFFSET and capped at MAX_ARRAY_INDEX.
    # Contiguous chunks are ~BATCH wide so this never trips; a sparse gap-fill
    # chunk could in principle straddle a wide hole, so shrink it if it does.
    while (( span > MAX_ARRAY_INDEX )); do
        chunk=("${chunk[@]:0:$(( ${#chunk[@]} / 2 ))}")
        last=${chunk[-1]}
        span=$(( last - off ))
    done
    n=${#chunk[@]}

    # sbatch takes the relative indices as a comma list, e.g. --array=0,3,7
    rel=""
    for idx in "${chunk[@]}"; do rel+="$(( idx - off )),"; done
    rel=${rel%,}

    echo "=== chunk at offset ${off}: ${n} combination(s), indices ${off}..${last} ==="
    jid=$(sbatch --parsable \
                 --export=ALL,COMBO_OFFSET="${off}" \
                 --array="${rel}"%"${BATCH}" \
                 "$JOB_SCRIPT")
    echo "submitted array job ${jid}; waiting for it to drain..."

    sleep 15  # give the scheduler a moment to register the array

    # Wait for the array to drain. The naive form of this loop --
    #   while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep; done
    # -- cannot tell "the array is gone" from "squeue did not answer", because
    # 2>/dev/null throws away the error and a failed squeue prints nothing on
    # stdout either. slurmctld on this cluster does go unreachable for a poll or
    # two (slurmdbd is down permanently, so this is not a well one), and when it
    # did, the loop exited ~5 min before the tasks actually finished, the CSV
    # check below then found nothing and the whole run halted on a chunk that
    # went on to succeed in full.
    #
    # So: only a *successful* squeue counts as evidence, and it has to come back
    # empty EMPTY_STREAK times in a row before we believe the array is done.
    EMPTY_STREAK=${EMPTY_STREAK:-3}
    empty=0
    while (( empty < EMPTY_STREAK )); do
        if queued=$(squeue -h -j "${jid}" 2>/dev/null); then
            if [[ -n "$queued" ]]; then empty=0; else empty=$(( empty + 1 )); fi
        else
            # squeue failed: says nothing about the array, so hold the streak.
            echo "   (squeue unavailable; not treating that as drained)" >&2
        fi
        if (( empty < EMPTY_STREAK )); then sleep "${POLL}"; fi
    done

    # squeue only tells us the tasks are gone, not whether they succeeded, and
    # sacct can't tell us anything at all here -- so ask the filesystem. Checking
    # the whole [off..last] span is right in both modes: indices inside the span
    # that weren't part of this chunk already have their CSV, so they stay quiet.
    failed=()
    mapfile -t failed < <(python3 "$STATUS" "$JOB_NAME" --start "$off" --n "$(( span + 1 ))" "${status_args[@]}")
    if (( ${#failed[@]} > 0 )); then
        echo "!! chunk at offset ${off}: ${#failed[@]} combination(s) produced no usable CSV:" >&2
        printf '     combo %s\n' "${failed[@]}" >&2
        echo "Logs: logs_bts_merged/${JOB_NAME%_array.sh}_array_${jid}_*.err" >&2
        echo "Pausing. Investigate the logs, then fill the gaps with:" >&2
        echo "   MISSING=1 ${REQUIRE_NONZERO:+REQUIRE_NONZERO=\"${REQUIRE_NONZERO}\" }$0 ${JOB_NAME} ${TOTAL} ${BATCH}" >&2
        exit 1
    fi

    echo "chunk at offset ${off} finished (all ${n} combinations wrote a CSV)."
    echo

    # Advance past any chunk we had to shrink above.
    (( c -= BATCH - n )) || true
done

echo "All ${n_total} combinations processed."
