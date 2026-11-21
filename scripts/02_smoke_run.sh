#!/usr/bin/env bash
# G1 smoke test / 基线(B0)首跑 —— 采集"四数"，且四数必须是实测而不是猜的。
#
#   export FADE_DIR=~/FADE
#   export COGVIDEO_PATH=/path/to/CogVideoX-5b        # 或 zai-org/CogVideoX-5b（在线）
#   # 模式 A（默认，最忠实基线）：直接跑 FADE 自带的 bear 演示，用 FADE 自己的长 prompt
#   bash scripts/02_smoke_run.sh
#   # 模式 B：换成我们自己的输入与短 prompt
#   export VIDEO=$FADE_DIR/input/bear.mp4
#   export INIT_PROMPT="a bear walking in the forest"
#   export EDIT_PROMPT="a panda walking in the forest"
#   SMOKE_MODE=generated bash scripts/02_smoke_run.sh
#
# 产出 logs/G1_<timestamp>/：cmd.txt stdout.log time.txt gpu.csv summary.txt latents_shape.txt
#
# 修订 R-01（2026-11-17，见 logs/RUN_LOG.md）——
#   旧版的两个测量缺陷已修：
#     (a) 旧版 export 了 VIDEO/INIT_PROMPT/EDIT_PROMPT 但内层只 `bash edit_pipeline.sh`，
#         **三个变量从未生效**，实际永远跑 FADE 硬编码的 bear 示例；
#     (b) 旧版在 `python -c` 包装进程里读 torch.cuda.max_memory_allocated()，
#         而真正的计算在 subprocess 子进程里（独立 CUDA 上下文），
#         **峰值必然打印 0.00 GB**——那是假数字，违反"预估不得当实测"。
#   现在：峰值只认 nvidia-smi 采样最大值（1 Hz），并显式标注采样口径与局限。
set -u

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="logs/G1_${STAMP}"
mkdir -p "$OUT"

: "${FADE_DIR:?set FADE_DIR}"
SMOKE_MODE=${SMOKE_MODE:-stock}          # stock | generated
MODEL_PATH=${MODEL_PATH:-${COGVIDEO_PATH:-zai-org/CogVideoX-5b}}
CONFIG_PATH=${CONFIG_PATH:-configs/bear.yaml}
FPS=${FPS:-4}
REF_TEXT=${REF_TEXT:-a bear}
DEVICE=${DEVICE:-cuda:0}
NUM_STEPS=${NUM_STEPS:-50}
LBE_STEP=${LBE_STEP:-35}                  # 掩码消费点，SSOT i<35

echo "== G1 smoke test ==" | tee "$OUT/head.txt"
{
  echo "FADE_DIR   $FADE_DIR"
  echo "MODEL_PATH $MODEL_PATH"
  echo "SMOKE_MODE $SMOKE_MODE"
  echo "CONFIG     $CONFIG_PATH"
  echo "FPS        $FPS"
  echo "LBE_STEP   $LBE_STEP"
  echo "date       $(date -Is)"
  echo "host       $(hostname)"
} | tee -a "$OUT/head.txt"

# 1. 环境快照（不修改任何东西）
python scripts/env_check.py 2>&1 | tee "$OUT/env.txt"

# 2. 运行前基线显存 + 1 Hz 采样
IDLE_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "idle_memory_mib=$IDLE_MIB" | tee "$OUT/gpu_idle.txt"

( while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
               --format=csv,noheader >> "$OUT/gpu.csv" 2>/dev/null
    sleep 1
  done ) &
GPU_PID=$!
trap 'kill $GPU_PID 2>/dev/null' EXIT

# 3. 组装要跑的命令
cd "$FADE_DIR" || exit 1
if [ "$SMOKE_MODE" = "stock" ]; then
  RUNNER=(bash edit_pipeline.sh)
  echo "mode=stock -> bash edit_pipeline.sh（FADE 自带 bear 演示 + 自带长 prompt）" | tee "$OUT/mode.txt"
else
  : "${VIDEO:?set VIDEO (mp4, 不是 DAVIS 的 jpg)}"
  : "${INIT_PROMPT:?set INIT_PROMPT}"
  : "${EDIT_PROMPT:?set EDIT_PROMPT}"
  PIP="$OUT/pipeline.sh"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -e'
    echo 'python ddim_inversion.py --model_path "$MODEL_PATH" --prompt "$INIT_PROMPT" \'
    echo '       --video_path "$VIDEO" --output_path "$LATENT_PT" --fps "$FPS" --device "$DEVICE"'
    echo 'python generate_mask.py --input_video "$VIDEO" --output_path "$MASK_PT" \'
    echo '       --ref_text "$REF_TEXT" --device "$DEVICE"'
    echo 'python edit.py --model_path "$MODEL_PATH" --init_prompt "$INIT_PROMPT" \'
    echo '       --edit_prompt "$EDIT_PROMPT" --device "$DEVICE" --dtype bf16 \'
    echo '       --input_video_path "$VIDEO" --latent_trajectory_path "$LATENT_PT" \'
    echo '       --mask_path "$MASK_PT" --config_path "$CONFIG_PATH" --fps "$FPS" \'
    echo '       --num_inference_steps "$NUM_STEPS" --local_blend_end_step "$LBE_STEP" \'
    echo '       --output_video_path "$OUT_VIDEO"'
  } > "$PIP"
  export VIDEO INIT_PROMPT EDIT_PROMPT MODEL_PATH CONFIG_PATH FPS REF_TEXT DEVICE NUM_STEPS LBE_STEP
  export LATENT_PT="$PWD/inversion/smoke_${STAMP}.pt"
  export MASK_PT="$PWD/mask/smoke_${STAMP}.pt"
  export OUT_VIDEO="$PWD/output/smoke_${STAMP}.mp4"
  mkdir -p inversion mask output
  RUNNER=(bash "$OLDPWD/$PIP")
  echo "mode=generated -> $OUT/pipeline.sh" | tee "$OUT/mode.txt"
fi
echo "${RUNNER[*]}" > "$OUT/cmd.txt"

# 4. 跑，并取真实墙钟
T0=$(date +%s)
{ "${RUNNER[@]}" ; } > "$OUT/stdout.log" 2>&1
RC=$?
T1=$(date +%s)
WALL=$((T1 - T0))
sleep 1
kill $GPU_PID 2>/dev/null

# 5. 四数（全部实测；不可得就写 NOT_MEASURED，绝不填预估值）
PEAK_MIB=$(awk -F',' 'NR>1{gsub(/ /,"",$2); if($2+0>m) m=$2+0} END{print (m?m:"")}' "$OUT/gpu.csv")
if [ -n "$PEAK_MIB" ]; then
  PEAK_GB=$(awk -v m="$PEAK_MIB" 'BEGIN{printf "%.2f", m/1024}')
else
  PEAK_GB="NOT_MEASURED"
fi
if grep -qiE 'out of memory|CUDA error|RuntimeError' "$OUT/stdout.log"; then OOM="YES"; else OOM="no"; fi
LATENTS=$(grep -m1 'latents.shape' "$OUT/stdout.log" || echo "NOT_FOUND (步4 需先给 edit.py 加一行 print)")
echo "exit_code=$RC"                >  "$OUT/summary.txt"
echo "wall_seconds=$WALL"           >> "$OUT/summary.txt"
echo "peak_gpu_gb=$PEAK_GB"         >> "$OUT/summary.txt"
echo "peak_gpu_mib=$PEAK_MIB"       >> "$OUT/summary.txt"
echo "idle_memory_mib=$IDLE_MIB"    >> "$OUT/summary.txt"
echo "oom=$OOM"                     >> "$OUT/summary.txt"
echo "latents_shape=$LATENTS"       >> "$OUT/summary.txt"
echo "measurement_scope=nvidia-smi 1Hz max(memory.used)，含同卡其他进程；独占机才等价于本作业峰值" >> "$OUT/summary.txt"
cat "$OUT/summary.txt" | tee "$OUT/summary.print.txt"

cat <<EOF

【四数 · 实测】目录 $OUT
  1. 单条耗时   ${WALL} s        （档位：实测）
  2. 峰值显存   ${PEAK_GB} GB    （档位：实测，1 Hz 采样上限，可能低估 1 帧尖峰）
  3. 是否 OOM   ${OOM}
  4. 肉眼质量   需要你自己看 $FADE_DIR/output/
  latents.shape $LATENTS

下一步（都要手记进 logs/RUN_LOG.md）：
  · 若 wall > 6 min/条 → 按 SSOT：11/22 拆两夜（唯一允许的日程改动）
  · 把 peak 与 wall 回填 SSOT 的"~3 min / ~23.6 GB"两行（现在是论文里的预估值，不是你的机器）
  · 步4：edit.py 里加一行 print("latents.shape", latents.shape)，第二维必须是 13
EOF
exit $RC
