#!/bin/bash
# Arguments: USD limit, idle minutes, actual billing-start Unix epoch.
set -uo pipefail
RATE=${SAIG_HOURLY_RATE:-2.19}
BUDGET=${1:-6.40}
IDLE_MIN=${2:-8}
LOGDIR=${SAIG_LOG_DIR:-/root/logs}
CONTROL=${SAIG_CONTROL:-$LOGDIR/control}
mkdir -p "$CONTROL/jobs"
exec 8> "$CONTROL/guard.lock"
flock -n 8 || { echo 'Another budget guard is active'; exit 2; }
if [ -f "$CONTROL/billing_start" ]; then
  START=$(cat "$CONTROL/billing_start")
  if [ -n "${3:-}" ] && [ "$3" != "$START" ]; then
    echo 'Billing start differs from saved run state'; exit 2
  fi
else
  START=${3:-${SAIG_BILLING_START_EPOCH:-}}
  [[ "$START" =~ ^[0-9]+$ ]] && [ "$START" -le "$(date +%s)" ] || {
    echo 'Supply the actual pod billing-start epoch as argument 3'; exit 2;
  }
  printf '%s\n' "$START" > "$CONTROL/billing_start"
fi
awk -v b="$BUDGET" -v r="$RATE" -v i="$IDLE_MIN" 'BEGIN{exit !(b>0 && r>0 && i>0)}' || exit 2
LED=$LOGDIR/budget_guard.csv
[ -f "$LED" ] || echo 'ts,elapsed_s,estimated_usd,active_jobs,idle_s' > "$LED"
idle_since=$(date +%s)

verified_group() {
  local file=$1 pid stamp actual group
  read -r pid stamp < "$file" || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] && [ "$pid" -gt 1 ] && [ -r "/proc/$pid/stat" ] || return 1
  actual=$(awk '{print $22}' "/proc/$pid/stat")
  group=$(ps -o pgid= -p "$pid" | tr -d ' ')
  [ "$stamp" = "$actual" ] && [ "$pid" = "$group" ] || return 1
  printf '%s' "$pid"
}

stop_jobs() {
  local reason=$1 file pid
  printf '%s %s\n' "$(date -Is)" "$reason" > "$CONTROL/STOP.tmp"
  mv "$CONTROL/STOP.tmp" "$CONTROL/STOP"
  for file in "$CONTROL"/jobs/*.pid; do
    [ -f "$file" ] || continue
    pid=$(verified_group "$file") || continue
    echo "TERM experiment process group $pid"
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  echo 'COMPUTE_STOP_REQUESTED; pod Stop and local backup receipt are still required'
}

while true; do
  now=$(date +%s)
  elapsed=$((now - START))
  active=0
  for file in "$CONTROL"/jobs/*.pid; do
    [ -f "$file" ] || continue
    verified_group "$file" >/dev/null && active=$((active + 1))
  done
  if [ "$active" -gt 0 ]; then idle_since=$now; fi
  idle=$((now - idle_since))
  spent=$(awk -v t="$elapsed" -v r="$RATE" 'BEGIN{printf "%.6f", t*r/3600}')
  printf '%s,%s,%s,%s,%s\n' "$(date -Is)" "$elapsed" "$spent" "$active" "$idle" >> "$LED"
  echo "estimated_usd=$spent elapsed_s=$elapsed active=$active idle_s=$idle"
  if awk -v s="$spent" -v b="$BUDGET" 'BEGIN{exit !(s>=b)}'; then stop_jobs OVER_BUDGET; exit 3; fi
  if [ -f "$CONTROL/STOP" ]; then stop_jobs EXISTING_STOP; exit 3; fi
  if [ "$idle" -ge $((IDLE_MIN * 60)) ]; then stop_jobs IDLE_LIMIT; exit 4; fi
  sleep 30
done
