#!/bin/bash
# Cron watchdog for the batched sweep: mails you when it needs attention.
#
# run_eval_batched.sh halts on a bad chunk by design and then simply sits there
# — nothing announces it. That is how three days of idle cluster went unnoticed
# on 2026-08-07. This turns the stall into a push.
#
# Alerts on a *transition*, not on every run, so a healthy sweep is silent:
#   running  -> stalled    "sweep has stopped with N combinations left"
#   running  -> complete   "sweep finished"
# While stalled it re-sends at most once every REPEAT_HOURS so a stall you have
# not dealt with does not fall off the bottom of your inbox.
#
# Install (see the comment at the bottom for the crontab line):
#   scripts/sweep_alert.sh                 # one check, honours the state file
#   FORCE=1 scripts/sweep_alert.sh         # mail the current status right now
#
# MAILTO / KIND / REPEAT_HOURS can be overridden from the crontab line.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

KIND=${KIND:-target}
MAILTO=${MAILTO:-}          # opt-in second channel; see the delivery block below
REPEAT_HOURS=${REPEAT_HOURS:-24}
FORCE=${FORCE:-0}
STATE="logs_bts_merged/.sweep_alert_${KIND}.state"

case "$KIND" in
    target) JOB_NAME=eval_target_array.sh; status_args=(--require-nonzero stsb) ;;
    source) JOB_NAME=eval_array.sh;        status_args=() ;;
    *) echo "usage: KIND=[target|source] $0" >&2; exit 1 ;;
esac

left=$(python3 scripts/combo_status.py "$KIND" "${status_args[@]}" --count 2>/dev/null)
# A missing/garbled count means the check itself is broken, not that the sweep
# is done -- staying quiet beats mailing "0 left, all finished!".
[[ "$left" =~ ^[0-9]+$ ]] || exit 0

if pgrep -f "run_eval_batched.sh ${JOB_NAME}" >/dev/null 2>&1; then
    now=running
elif (( left == 0 )); then
    now=complete
else
    now=stalled
fi

prev=""; prev_at=0
if [[ -f "$STATE" ]]; then read -r prev prev_at < "$STATE" 2>/dev/null; fi
[[ "$prev_at" =~ ^[0-9]+$ ]] || prev_at=0

send=0
if (( FORCE )); then
    send=1
elif [[ "$now" != "$prev" ]] && [[ "$now" != "running" ]]; then
    send=1   # just entered a state worth knowing about
elif [[ "$now" == "stalled" ]] && (( $(date +%s) - prev_at >= REPEAT_HOURS * 3600 )); then
    send=1   # still stalled, nudge again
fi

if (( ! send )); then
    # Nothing to say, but still record where we are. Reaching a healthy state
    # clears the repeat clock so a later stall nudges immediately.
    stamp=$prev_at
    [[ "$now" == "running" ]] && stamp=0
    printf '%s %s\n' "$now" "$stamp" > "$STATE"
    exit 0
fi

case "$now" in
    stalled)  subject="bts: ${KIND} sweep STOPPED — ${left} combinations left"
              prio=high;    tags="rotating_light" ;;
    complete) subject="bts: ${KIND} sweep finished — all combinations done"
              prio=default; tags="white_check_mark" ;;
    *)        subject="bts: ${KIND} sweep status — ${left} left"
              prio=low;     tags="mag" ;;
esac

body=$(
    scripts/sweep_status.sh "$KIND" 2>&1
    if [[ "$now" == "stalled" ]]; then
        echo
        echo "Resume (finished combinations are skipped, nothing is redone):"
        echo "  cd $REPO"
        echo "  MISSING=1 REQUIRE_NONZERO=stsb nohup scripts/slurm/run_eval_batched.sh ${JOB_NAME} 4083 48 &"
        echo
        echo "Check the chunk's logs first -- the wrapper prints the job id it gave up on."
    fi
)

# --- delivery -----------------------------------------------------------------
# Not mail: the cluster relay accepts gmail.com and drops it with no bounce and
# nothing in the queue, so a mailed alert looks sent and simply never arrives.
# Outbound HTTPS does work from login01, so alerts go over ntfy instead. The
# topic name is the only thing protecting it, which is why it lives in a 0600
# file outside the repo rather than being written down here.
delivered=0
topic_file=${NTFY_TOPIC_FILE:-$HOME/.config/bts/ntfy_topic}
if [[ -r "$topic_file" ]]; then
    topic=$(tr -d '[:space:]' < "$topic_file")
    if [[ -n "$topic" ]] && curl -sSf --max-time 20 \
            -H "Title: ${subject}" -H "Priority: ${prio}" -H "Tags: ${tags}" \
            -d "$body" "https://ntfy.sh/${topic}" >/dev/null 2>&1; then
        delivered=1
    fi
fi
# MAILTO is opt-in and off by default; set it once you have an address the relay
# will actually deliver to.
if [[ -n "${MAILTO:-}" ]] && printf '%s\n' "$body" | mail -s "$subject" "$MAILTO" 2>/dev/null; then
    delivered=1
fi

# Only bank the alert timestamp if it actually went out. A failed push (ntfy down,
# network blip) leaves the clock alone so the next run tries again instead of
# silently swallowing the one alert that mattered.
if (( delivered )); then
    printf '%s %s\n' "$now" "$(date +%s)" > "$STATE"
else
    printf '%s %s\n' "$prev" "$prev_at" > "$STATE"
    echo "$(date '+%F %H:%M') delivery FAILED for: $subject" >&2
    exit 1
fi

# crontab line (crontab -e on login01 -- cron is per-node, and the wrapper runs
# on login01, so this has to live on the same node):
#   PATH=/usr/bin:/bin:/usr/sbin:/sbin
#   */30 * * * * /mnt/project/perun250162/bayesian-task-selection/scripts/sweep_alert.sh >> /mnt/project/perun250162/bayesian-task-selection/logs_bts_merged/sweep_alert.log 2>&1
