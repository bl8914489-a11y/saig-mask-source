#!/bin/bash
# P0.5：三个控制臂（M5 无掩码 / M6 GT 掩码 / M3 边界细化）× 3 段
exec > /root/p05.log 2>&1
set -x
V=/root/venv310/bin/python
FADE=/root/saig/FADE
M=/root/saig/CogVideoX-5b
SCR=/root/saig/saig_project/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M:%S)] P05 $*"; }
sub() { case $1 in camel) echo camel;; dog) echo dog;; flamingo) echo flamingo;; esac; }
tgt() { case $1 in camel) echo horse;; dog) echo wolf;; flamingo) echo swan;; esac; }

cd "$FADE" || exit 1
say START

# 0 造掩码
$V /root/build_control_masks.py 2>&1 | tail -6
say "masks rc=$?"

# 1 九个视频
for s in camel dog flamingo; do
  for arm in m5_nomask:none m6_gtmask:gt m3_refine:refine; do
    NAME=${arm%%:*}; MK=${arm##*:}
    OUT="output/${NAME}_$s.mp4"
    if [ -f "$OUT" ]; then say "SKIP $NAME $s"; continue; fi
    t0=$(date +%s)
    $V edit.py --model_path "$M" \
      --init_prompt "a $(sub $s)" --edit_prompt "a $(tgt $s)" --device cuda:0 --dtype bf16 \
      --input_video_path "input/$s.mp4" --latent_trajectory_path "inversion/$s.pt" \
      --mask_path "mask/${MK}_$s.pt" --config_path configs/bear.yaml --fps 4 \
      --output_video_path "output/${NAME}_$s.partial.mp4" > "/root/logs/${NAME}_${s}.log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] && mv -f "output/${NAME}_$s.partial.mp4" "$OUT"
    say "$NAME $s rc=$rc wall=$(( $(date +%s) - t0 ))s frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $OUT 2>/dev/null)"
  done
done

# 2 指标（旧版脚本，含新臂）
say "=== 指标 ==="
for s in camel dog flamingo; do
  $V "$SCR/metrics_compare_old.py" --seq "$s" --src "input/$s.mp4" \
    --ann "/root/saig/DAVIS/Annotations/480p/$s" \
    --init "a $(sub $s)" --target "a $(tgt $s)" \
    --videos "b0=output/b0_$s.mp4" "gate=output/b1_$s.mp4" "centre=output/b1c_$s.mp4" \
             "nomask=output/m5_nomask_$s.mp4" "gtmask=output/m6_gtmask_$s.mp4" \
             "refine=output/m3_refine_$s.mp4" \
    --out "/root/logs/metrics_p05_$s.csv" 2>&1 | tail -8
  say "$s metrics rc=$?"
done

say "P05_DONE"
