#!/bin/bash
# 补充实验 S3：中间帧门控的真实编辑（camel/dog/flamingo），并与 b0/gate/centre 对比
exec > /root/mid_edit.log 2>&1
set -x
V=/root/venv310/bin/python
FADE=/root/saig/FADE
M=/root/saig/CogVideoX-5b
SCR=/root/saig/saig_project/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M:%S)] MID $*"; }
sub() { case $1 in camel) echo camel;; dog) echo dog;; flamingo) echo flamingo;; esac; }
tgt() { case $1 in camel) echo horse;; dog) echo wolf;; flamingo) echo swan;; esac; }

cd "$FADE" || exit 1
say START

for s in camel dog flamingo; do
  OUT="output/b1m_$s.mp4"
  if [ -f "$OUT" ]; then say "SKIP $s"; continue; fi
  t0=$(date +%s)
  SAIG_GID="$FADE/masks/gid_mid_$s.npy" $V edit.py --model_path "$M" \
    --init_prompt "a $(sub $s)" --edit_prompt "a $(tgt $s)" --device cuda:0 --dtype bf16 \
    --input_video_path "input/$s.mp4" --latent_trajectory_path "inversion/$s.pt" \
    --mask_path "mask/$s.pt" --config_path configs/bear.yaml --fps 4 \
    --output_video_path "output/b1m_$s.partial.mp4" > "/root/logs/mid_${s}.log" 2>&1
  rc=$?
  grep -q SAIG_GATE_APPLIED "/root/logs/mid_${s}.log" || rc=65
  [ "$rc" -eq 0 ] && mv -f "output/b1m_$s.partial.mp4" "$OUT"
  say "$s rc=$rc wall=$(( $(date +%s) - t0 ))s frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $OUT 2>/dev/null)"
done

say "=== 指标（b0 / gate / centre / mid）==="
for s in camel dog flamingo; do
  $V "$SCR/metrics_compare_old.py" --seq "$s" --src "input/$s.mp4" \
    --ann "/root/saig/DAVIS/Annotations/480p/$s" \
    --init "a $(sub $s)" --target "a $(tgt $s)" \
    --videos "b0=output/b0_$s.mp4" "gate=output/b1_$s.mp4" "center=output/b1c_$s.mp4" "mid=output/b1m_$s.mp4" \
    --out "/root/logs/metrics_mid_$s.csv" 2>&1 | tail -6
  say "$s metrics rc=$?"
done
say MID_DONE
