#!/bin/bash
# =============================================================================
# 显存探针：在任意卡上跑一次真实编辑，测「开 offload 后的真实峰值显存」
#
# 目的：判定便宜的 48 GB 卡（RTX 4090 48G / L20 / A6000）能不能顶 13 潜帧 FADE。
#       历史数字 79 GiB 是「不开 offload」的峰值，开 offload 后无人量过。
#
# 用法（pod 上，FADE 目录已有 inversion/mask）：
#   bash /root/gpu_probe.sh camel
# 判据：
#   PEAK_TORCH_GB < 42  → 48 GB 卡可用（留 6 GB 余量）
#   42–46               → 临界，需降分辨率/步数再试
#   > 46                → 必须 80 GB 级
# =============================================================================
set -uo pipefail
SEQ=${1:-camel}
F=/root/saig/FADE
V=/root/venv310/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /root/logs
OUT=/root/logs/probe_${SEQ}_$(date +%H%M%S).log

cd "$F" || { echo "PROBE_FAIL 无 $F"; exit 1; }
[ -f "inversion/$SEQ.pt" ] || { echo "PROBE_FAIL 缺 inversion/$SEQ.pt"; exit 1; }
[ -f "mask/$SEQ.pt" ] || { echo "PROBE_FAIL 缺 mask/$SEQ.pt"; exit 1; }

case "$SEQ" in
  bear) SUB=bear; TGT=panda;; camel) SUB=camel; TGT=horse;; cows) SUB=cow; TGT=zebra;;
  dog) SUB=dog; TGT=wolf;; elephant) SUB=elephant; TGT=rhinoceros;;
  flamingo) SUB=flamingo; TGT=swan;; goat) SUB=goat; TGT=sheep;;
  rhino) SUB=rhinoceros; TGT=hippopotamus;; *) SUB=$SEQ; TGT=panda;;
esac
export SUB TGT SEQ

echo "PROBE_START $(date -Is) seq=$SEQ sub=$SUB tgt=$TGT" | tee "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$OUT"

SMI=/root/logs/probe_smi.txt
: > "$SMI"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SMI" 2>/dev/null; sleep 1; done ) &
SAMPLER=$!

SAIG_GID="$F/masks/gid_$SEQ.npy" "$V" - <<'PY' 2>&1 | tee -a "$OUT"
import os, sys, runpy, torch

seq = os.environ["SEQ"]; sub = os.environ["SUB"]; tgt = os.environ["TGT"]
sys.argv = ["edit.py", "--model_path", "/root/saig/CogVideoX-5b",
            "--init_prompt", "a " + sub, "--edit_prompt", "a " + tgt,
            "--device", "cuda:0", "--dtype", "bf16",
            "--input_video_path", "input/" + seq + ".mp4",
            "--latent_trajectory_path", "inversion/" + seq + ".pt",
            "--mask_path", "mask/" + seq + ".pt",
            "--config_path", "configs/bear.yaml", "--fps", "4",
            "--num_inference_steps", "5",
            "--output_video_path", "output/_probe_mem.mp4"]
sys.path.insert(0, os.getcwd())
torch.cuda.reset_peak_memory_stats()
try:
    runpy.run_path("edit.py", run_name="__main__")
except SystemExit:
    pass
except Exception as e:
    print("PROBE_EXC", type(e).__name__, str(e)[:200])
print("PEAK_TORCH_GB %.2f" % (torch.cuda.max_memory_allocated() / 1024**3))
print("PEAK_RESERVED_GB %.2f" % (torch.cuda.max_memory_reserved() / 1024**3))
PY

kill $SAMPLER 2>/dev/null
SMI_PEAK=$(sort -n "$SMI" 2>/dev/null | tail -1)
echo "PEAK_NVIDIA_SMI_MiB=${SMI_PEAK:-NA}" | tee -a "$OUT"
echo "PROBE_END $(date -Is)" | tee -a "$OUT"
echo "→ 判据：PEAK_TORCH_GB < 42 → 48 GB 卡可用；42–46 → 临界；> 46 → 必须 80 GB 级"
