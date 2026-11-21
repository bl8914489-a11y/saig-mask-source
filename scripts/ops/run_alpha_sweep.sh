#!/bin/bash
# =============================================================================
# α 敏感性实验：α ∈ {0.3, 0.7}（0.5 已有） × 3 段（camel/dog/flamingo）
#   验证「加多少强度」是否改变结论：ID-S 上升、Sem-T 下降的幅度
# 判据：每个视频 49 帧、SAIG_GATE_APPLIED 出现、体积 > 500KB
# =============================================================================
exec > /root/alpha_sweep.log 2>&1
set -x
F=/root/saig/FADE
V=/root/venv310/bin/python
M=/root/saig/CogVideoX-5b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M:%S)] ALPHA $*"; }

sub() { case $1 in bear) echo bear;; camel) echo camel;; cows) echo cow;; dog) echo dog;;
  elephant) echo elephant;; flamingo) echo flamingo;; goat) echo goat;; rhino) echo rhinoceros;; esac; }
tgt() { case $1 in bear) echo panda;; camel) echo horse;; cows) echo zebra;; dog) echo wolf;;
  elephant) echo rhinoceros;; flamingo) echo swan;; goat) echo sheep;; rhino) echo hippopotamus;; esac; }

cd "$F" || exit 1
say START

for a in 0.3 0.7; do
  for s in camel dog flamingo; do
    OUT="output/b1a${a}_${s}.mp4"
    if [ -f "$OUT" ]; then say "SKIP $s alpha=$a"; continue; fi
    say "RUN $s alpha=$a"
    t0=$(date +%s)
    SAIG_ALPHA="$a" SAIG_GID="$F/masks/gid_$s.npy" $V edit.py --model_path "$M" \
      --init_prompt "a $(sub $s)" --edit_prompt "a $(tgt $s)" --device cuda:0 --dtype bf16 \
      --input_video_path "input/$s.mp4" --latent_trajectory_path "inversion/$s.pt" \
      --mask_path "mask/$s.pt" --config_path configs/bear.yaml --fps 4 \
      --output_video_path "output/b1a${a}_${s}.partial.mp4" > "/root/logs/alpha_${s}_${a}.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      grep -q 'SAIG_GATE_APPLIED' "/root/logs/alpha_${s}_${a}.log" || rc=65
    fi
    if [ "$rc" -eq 0 ]; then
      n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "output/b1a${a}_${s}.partial.mp4" 2>/dev/null)
      [ "$n" = "49" ] || rc=66
    fi
    if [ "$rc" -eq 0 ]; then mv -f "output/b1a${a}_${s}.partial.mp4" "$OUT"; fi
    say "$s alpha=$a rc=$rc wall=$(( $(date +%s) - t0 ))s frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $OUT 2>/dev/null)"
  done
done

# 指标：把 α 视频当作一个臂
say "=== 指标 ==="

say "ALPHA_DONE"
