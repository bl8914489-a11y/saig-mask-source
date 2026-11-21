#!/bin/bash
# =============================================================================
# 预检 v2（针对优云智算 / compshare）：机型兼容性 + 平台模型库 + 网络出口
#
# 与 v1 的差别：
#   1. 新增 GPU 架构（compute capability）检查，直接判 torch cu124 能不能用
#   2. 新增 /model 平台模型库扫描（优云把 HF/ModelScope 权重挂在 /model/... 下）
#   3. 新增显存是否够 13 潜帧 FADE 的判断（峰值 ≈79 GiB）
#
# 用法：bash precheck.sh v2
# 判据：末尾三行 —— GPU_VERDICT / MODELLIB / MIRROR_PLAN
# =============================================================================
echo "==================== 1. 机型与驱动 ===================="
date -Is
echo "HOST=$(hostname)"
[ -f /etc/os-release ] && . /etc/os-release && echo "OS=$PRETTY_NAME"
echo "CPU=$(nproc) cores  MEM=$(free -g | awk '/Mem:/{print $2"G"}')"
df -h / /root 2>/dev/null | sed -n '1,4p'

GPU_NAME=""; GPU_MEM=""; GPU_CC=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  echo "GPU=$GPU_NAME  MEM=${GPU_MEM}MiB  DRIVER=$DRV"
else
  echo "NO_NVIDIA_SMI"
fi

echo
echo "==================== 2. GPU 架构兼容性（决定性） ===================="
CC=""
if command -v python3 >/dev/null 2>&1; then
  CC=$(python3 - <<'PY' 2>/dev/null
try:
    import torch
    if torch.cuda.is_available():
        maj, mi = torch.cuda.get_device_capability(0)
        print("%d%d" % (maj, mi))
except Exception:
    pass
PY
)
fi
if [ -z "$CC" ]; then
  # 没有 torch 时用型号推断
  case "$GPU_NAME" in
    *"PRO 6000"*|*"5090"*|*"5080"*|*"5070"*|*"5060"*|*"B200"*|*"GB200"*) CC="120" ;;
    *H100*|*H800*|*H20*|*A100*|*A800*) CC="90" ;;
    *4090*|*4080*|*L40*|*L20*|*A6000*|*A40*) CC="89" ;;
    *3090*|*A10*|*A30*) CC="86" ;;
    *V100*) CC="70" ;;
  esac
fi
echo "COMPUTE_CAPABILITY=$CC  (空表示无法判断)"

VERDICT="UNKNOWN"
case "$CC" in
  120) VERDICT="FAIL_SM120  该卡是 Blackwell(sm_120)，torch 2.5.1+cu124 无内核，会报 no kernel image。需 torch>=2.7+cu128 重编，或换卡" ;;
  100) VERDICT="WARN_SM100  Blackwell 数据中心卡，同样需 cu128" ;;
  90|89|86|80|75|70) VERDICT="OK  该架构被 torch 2.5.1+cu124 支持" ;;
  *) VERDICT="UNKNOWN  请把 GPU 型号发回确认" ;;
esac
echo "GPU_VERDICT=$VERDICT"

echo
echo "==================== 3. 显存够不够（13 潜帧 FADE） ===================="
if [ -n "$GPU_MEM" ]; then
  if [ "$GPU_MEM" -ge 78000 ]; then
    echo "VRAM_OK ${GPU_MEM}MiB ≥ 80G 级：不开 offload 也能跑（峰值 ≈79 GiB）"
  elif [ "$GPU_MEM" -ge 45000 ]; then
    echo "VRAM_TIGHT ${GPU_MEM}MiB：必须开 cpu offload + expandable_segments，且掩码阶段风险高"
  else
    echo "VRAM_FAIL ${GPU_MEM}MiB：不足以跑 13 潜帧编辑"
  fi
fi

echo
echo "==================== 4. 平台模型库（优云 /model/...） ===================="
FOUND_CV=""; FOUND_SAM=""; FOUND_CLIP=""; FOUND_DINO=""
if [ -d /model ]; then
  echo "MODEL_ROOT=/model"
  ls -1 /model 2>/dev/null | head -12 | sed 's/^/  /'
  FOUND_CV=$(ls -d /model/*/*CogVideoX* /model/*CogVideoX* 2>/dev/null | head -1)
  FOUND_SAM=$(ls -d /model/*/*segment*anything* /model/*sam* 2>/dev/null | head -1)
  FOUND_CLIP=$(ls -d /model/*/*clip* /model/*CLIP* 2>/dev/null | head -1)
  FOUND_DINO=$(ls -d /model/*/*dino* /model/*DINO* 2>/dev/null | head -1)
else
  echo "MODEL_ROOT=无 /model 目录"
fi
echo "MODEL_COGVIDEOX=${FOUND_CV:-NOT_FOUND}"
echo "MODEL_SAM=${FOUND_SAM:-NOT_FOUND}"
echo "MODEL_CLIP=${FOUND_CLIP:-NOT_FOUND}"
echo "MODEL_DINO=${FOUND_DINO:-NOT_FOUND}"
[ -n "$FOUND_CV" ] && echo "→ 命中模型库，可软链到 /root/saig/CogVideoX-5b，省掉 21 GB 下载"

echo
echo "==================== 5. 网络出口 ===================="
probe() {
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 8 -L "$2" 2>/dev/null)
  secs=$(curl -s -o /dev/null -w "%{time_total}" -m 8 -L "$2" 2>/dev/null)
  printf "%-22s %s  %ss\n" "$1" "${code:-timeout}" "${secs:-0}"
}
probe "pypi.org" "https://pypi.org/simple/"
probe "pypi-tuna" "https://pypi.tuna.tsinghua.edu.cn/simple/"
probe "pytorch-official" "https://download.pytorch.org/whl/cu124/"
probe "huggingface" "https://huggingface.co"
probe "hf-mirror" "https://hf-mirror.com"
probe "modelscope" "https://www.modelscope.cn"
probe "github" "https://github.com"
probe "ghproxy" "https://ghproxy.net"
probe "davis(eth)" "https://data.vision.ee.ethz.ch/csergi/share/davis/"

echo
echo "==================== 6. 结论三行 ===================="
hf_ok=$(curl -s -o /dev/null -w "%{http_code}" -m 8 https://huggingface.co 2>/dev/null)
hfm_ok=$(curl -s -o /dev/null -w "%{http_code}" -m 8 https://hf-mirror.com 2>/dev/null)
ms_ok=$(curl -s -o /dev/null -w "%{http_code}" -m 8 https://www.modelscope.cn 2>/dev/null)
gh_ok=$(curl -s -o /dev/null -w "%{http_code}" -m 8 https://github.com 2>/dev/null)
pt_ok=$(curl -s -o /dev/null -w "%{http_code}" -m 8 https://download.pytorch.org/whl/cu124/ 2>/dev/null)
PLAN="pip=$([ "$pt_ok" = 200 ] && echo official || echo tsinghua)"
if [ "$hf_ok" = 200 ]; then PLAN="$PLAN hf=official"
elif [ "$hfm_ok" = 200 ]; then PLAN="$PLAN hf=hf-mirror"
elif [ "$ms_ok" = 200 ]; then PLAN="$PLAN hf=modelscope"
else PLAN="$PLAN hf=BLOCKED"; fi
PLAN="$PLAN git=$([ "$gh_ok" = 200 ] && echo official || echo ghproxy)"
[ -n "$FOUND_CV" ] && PLAN="$PLAN cgv=modellib"

echo "GPU_VERDICT=$VERDICT"
echo "MODELLIB cogvideox=${FOUND_CV:-none}"
echo "MIRROR_PLAN=$PLAN"
echo "→ 把这三行发回，即可确定重建与选型方案。"
