#!/bin/bash
# P6 validates P4 ledger records and saved tensors before running any edit.
set -euo pipefail
CONTROL=${SAIG_CONTROL:-/root/logs/control}
SCRIPT=${SAIG_PIPELINE:-/root/tonight_gpu.sh}
if [ "$#" -eq 0 ]; then seqs=(camel dog flamingo); else seqs=("$@"); fi
while pgrep -f '[t]onight_gpu.sh 4' >/dev/null; do
  [ ! -f "$CONTROL/STOP" ] || exit 3
  sleep 15
done
[ ! -f "$CONTROL/STOP" ] || exit 3
bash "$SCRIPT" 6 "${seqs[@]}"
bash "$SCRIPT" 5
bash "$SCRIPT" 7 "${seqs[@]}"
bash "$SCRIPT" 8
echo 'COMPUTE_COMPLETE BACKUP_READY; await local verified backup and pod Stop receipt'
