#!/bin/bash
# One screen of "is the sweep still going, and how far along is it?".
#
# run_eval_batched.sh runs detached for days and only speaks when a chunk ends,
# so the useful state is spread across four places: the wrapper process, squeue,
# the CSVs on disk, and the wrapper log. This gathers all four.
#
#   scripts/sweep_status.sh                 # target sweep (stsb-aware, the default)
#   scripts/sweep_status.sh source          # the source sweep instead
#   watch -n 120 scripts/sweep_status.sh    # keep it on screen
#
# Progress is measured from the CSVs, not from the log: the log says which chunk
# was submitted, the filesystem says what actually landed. REQUIRE_NONZERO must
# match what the run was launched with or the "done" counts disagree with it.

set -uo pipefail

# cron runs with a near-empty environment and does not set USER, which the squeue
# calls below need; id(1) always knows.
USER=${USER:-$(id -un)}

KIND=${1:-target}
REQUIRE_NONZERO=${REQUIRE_NONZERO:-stsb}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

case "$KIND" in
    target) JOB_NAME=eval_target_array.sh ;;
    source) JOB_NAME=eval_array.sh ;;
    *) echo "usage: $0 [target|source]" >&2; exit 1 ;;
esac

status_args=()
# Only the target sweep has a fixed-metric column to re-check; see combo_status.py.
if [[ "$KIND" == "target" && -n "$REQUIRE_NONZERO" ]]; then
    for col in $REQUIRE_NONZERO; do status_args+=(--require-nonzero "$col"); done
fi

total=$(python3 -c "
import sys; sys.path.insert(0, 'scripts')
from combo_status import expected_csvs; print(len(expected_csvs('$KIND')))")
left=$(python3 scripts/combo_status.py "$KIND" "${status_args[@]}" --count)
done=$(( total - left ))
pct=$(( done * 100 / total ))

printf '%s sweep  —  %s\n' "$KIND" "$(date '+%F %H:%M')"
printf 'progress   %s/%s combinations (%s%%), %s to go\n' "$done" "$total" "$pct" "$left"

# --- the wrapper --------------------------------------------------------------
# Both sweeps write run_eval_batched_*.log, so pick the newest one that actually
# belongs to this sweep rather than whichever was started last.
log=""
for f in $(ls -t logs_bts_merged/run_eval_batched_*.log 2>/dev/null); do
    if grep -q "Batched submit: ${JOB_NAME}" "$f" 2>/dev/null; then log="$f"; break; fi
done
if pgrep -f "run_eval_batched.sh ${JOB_NAME}" >/dev/null 2>&1; then
    printf 'wrapper    alive (pid %s)\n' "$(pgrep -f "run_eval_batched.sh ${JOB_NAME}" | head -1)"
elif (( left == 0 )); then
    printf 'wrapper    not running — sweep complete\n'
elif [[ -z "$log" ]]; then
    printf 'wrapper    not running — this sweep has never been started\n'
else
    printf 'wrapper    NOT RUNNING — stopped with %s left, see log below\n' "$left"
fi
[[ -n "$log" ]] && printf 'log        %s\n' "$log"

# --- the queue ----------------------------------------------------------------
run=$(squeue -u "$USER" -h -t RUNNING 2>/dev/null | wc -l)
pend=$(squeue -u "$USER" -h -t PENDING -r 2>/dev/null | wc -l)
jid=$(squeue -u "$USER" -h -o '%A' 2>/dev/null | sort -u | head -1)
printf 'queue      %s running, %s pending' "$run" "$pend"
[[ -n "$jid" ]] && printf '  (array job %s)' "$jid"
printf '\n'

# --- throughput, measured rather than assumed ---------------------------------
# Counting CSVs written in the last 6h beats multiplying a nominal per-combo time
# by an assumed width: it already accounts for the real array throttle, queue
# waits and any chunk barrier the wrapper was sitting in.
# Throughput from the newest N CSVs rather than a fixed clock window. A fixed
# window is wrong in both directions here: right after a restart it has almost
# nothing in it ("284 days left"), and for a while afterwards it still spans the
# outage that caused the restart ("37 days"). Timing the last N completions
# instead measures the sweep while it was actually running, and rolls the stall
# out of view as soon as N fresh CSVs exist.
if (( left == 0 )); then
    printf 'rate       —\n'
else
    python3 - "$left" <<'PY'
import os, sys, time, glob
left = int(sys.argv[1])
N = 40
mt = sorted((os.path.getmtime(p) for p in glob.glob("saves_bts_merged/*/*/*/*_acc_coef.csv")),
            reverse=True)[:N]
if len(mt) < 10:
    print(f"rate       only {len(mt)} completed CSVs — too few to extrapolate yet")
else:
    span_h = (mt[0] - mt[-1]) / 3600.0
    if span_h <= 0:
        print("rate       —")
    else:
        per_h = (len(mt) - 1) / span_h
        # A wide span means the sample still straddles a stall, so the pace is
        # badly understated. Quoting it with a caveat still puts a wrong number
        # in an alert, and "185 days" is the part people remember — so withhold
        # it and say what it is waiting for instead.
        if span_h > 12:
            fresh = sum(1 for t in mt if time.time() - t < 6 * 3600)
            print(f"rate       measuring — {fresh} completion(s) since the last "
                  f"restart; estimate once the sample clears the outage")
        else:
            eta_h = left / per_h if per_h else 0
            print(f"rate       {per_h:.1f} combos/h over the last {len(mt)} "
                  f"→ ~{eta_h:.0f}h (~{eta_h/24:.1f} days) left")
PY
fi

# --- did anything actually break? ---------------------------------------------
if [[ -n "$jid" ]]; then
    errs=$(for f in logs_bts_merged/${JOB_NAME%.sh}_${jid}_*.err; do
               [[ -f "$f" ]] && tr '\r' '\n' < "$f" 2>/dev/null \
                   | grep -aE "Traceback|FileNotFoundError|CUDA out of memory|slurmstepd"
           done | wc -l)
    (( errs > 0 )) && printf 'errors     %s error line(s) in job %s logs — inspect them\n' "$errs" "$jid"
fi

# --- what the wrapper last said -----------------------------------------------
if [[ -n "$log" ]]; then
    printf '\n--- %s (tail) ---\n' "$log"
    tail -6 "$log"
fi
exit 0
