#!/bin/bash
# =============================================================================
# A800 一键环境：把 H20 上踩过的所有坑都提前修掉
#   1 apt 装 git/unzip/ffmpeg/python3-venv
#   2 venv310 + torch 2.5.1+cu124
#   3 FADE + HybridGL
#   4 requirements + hf_hub 0.25.2 + CLIP .pth
#   5 spacy 模型（ghfast.top 六线程，失败再退回直连）
#   6 segment_anything + SAM 权重（HF LFS）
#   7 CogVideoX-5b（ModelScope）
#   8 DAVIS JPEGImages（HF 数据集 zip）
#   9 四类补丁（offload / rope / tag / sink / gate）
#  10 验收
# 用法：bash setup_env_part1.sh     日志：/root/setup_a800.log
# =============================================================================
exec > /root/setup_a800.log 2>&1
set -x
step() { echo "STEP $*"; }
die()  { echo "SETUP_FAIL $*"; exit 1; }

V=/root/venv310/bin/python
ROOT=/root/saig
FADE=$ROOT/FADE
M=$ROOT/CogVideoX-5b
PIPM="-i https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_DEFAULT_TIMEOUT=180

echo "SETUP_A800_START $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ---------- 1 apt ----------
step apt
export DEBIAN_FRONTEND=noninteractive
if ! command -v git >/dev/null || ! command -v ffmpeg >/dev/null || ! command -v unzip >/dev/null; then
  apt-get update -y -qq || echo WARN_apt_update
  apt-get install -y -qq git unzip ffmpeg python3-venv python3.10-venv || echo WARN_apt_partial
fi
for t in git unzip ffmpeg; do command -v $t >/dev/null && echo "OK $t" || echo "MISS $t"; done

# ---------- 2 venv + torch ----------
step venv
[ -x "$V" ] || python3 -m venv /root/venv310 || die "venv 创建失败"
$V -c "import pip" 2>/dev/null || $V -m ensurepip --upgrade || die "pip 自举失败"
$V -m pip install -q -U pip wheel $PIPM || die "pip 升级失败"

step torch
if ! $V -c "import torch;assert torch.cuda.is_available()" 2>/dev/null; then
  $V -m pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 300 || die "torch 安装失败"
fi
$V -c "import torch;print('TORCH',torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_capability(0))" || die "torch 导入失败"

# ---------- 3 代码 ----------
step clone
mkdir -p "$ROOT" && cd "$ROOT" || die "mkdir 失败"
[ -d FADE/.git ] || git clone --depth 1 https://github.com/EternalEvan/FADE.git FADE || die "FADE clone 失败"
[ -d FADE/HybridGL/.git ] || git clone --depth 1 https://github.com/fhgyuanshen/HybridGL.git FADE/HybridGL || die "HybridGL clone 失败"

# ---------- 4 依赖 ----------
step requirements
cd "$FADE" || die "cd FADE 失败"
$V -m pip install -r requirements.txt -q $PIPM --timeout 300 || die "requirements 失败"
$V -m pip install -q "setuptools<81" "huggingface_hub==0.25.2" $PIPM || die "基础包失败"
SP=/root/venv310/lib/python3.10/site-packages
echo "$FADE/HybridGL/third_party/modified_CLIP" > "$SP/zz_clip_fork.pth"
$V -c "import clip;print('CLIP_OK')" || die "clip 不可用"

# ---------- 5 spacy ----------
step spacy
if ! $V -c "import en_core_web_lg" 2>/dev/null; then
  WHL=https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
  # 六线程分段（H20 上验证最快）
  SIZE=$(curl -sIL -m 20 "https://ghfast.top/$WHL" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{gsub(/\r/,"",v);print v}')
  if [ -n "$SIZE" ] && [ "$SIZE" -gt 1000000 ]; then
    N=6; CHUNK=$(( (SIZE + N - 1) / N )); mkdir -p /root/spacy_parts; rm -f /root/spacy_parts/part_*
    for i in $(seq 0 $((N-1))); do
      S0=$(( i * CHUNK )); E0=$(( S0 + CHUNK - 1 )); [ $E0 -ge $SIZE ] && E0=$((SIZE-1))
      ( curl -sS -L -m 3600 -r "${S0}-${E0}" -o "/root/spacy_parts/part_$i" "https://ghfast.top/$WHL" || echo "PART_$i FAIL" ) &
    done
    wait
    cat /root/spacy_parts/part_* > /root/spacy.whl 2>/dev/null
    SZ=$(stat -c %s /root/spacy.whl 2>/dev/null || echo 0)
    echo "SPACY_ASM=$SZ EXPECT=$SIZE"
    [ "$SZ" = "$SIZE" ] && $V -m pip install -q --no-deps /root/spacy.whl
  fi
  $V -c "import en_core_web_lg" 2>/dev/null || {
    $V -m pip install -q "en-core-web-lg @ $WHL" || $V -m spacy download en_core_web_lg || die "spacy 模型失败"
  }
fi
$V -c "import en_core_web_lg;print('SPACY_OK')" || die "spacy 导入失败"

# ---------- 6 SAM ----------
step sam
$V -c "import segment_anything" 2>/dev/null || \
  $V -m pip install -q segment-anything-py $PIPM || \
  $V -m pip install -q "git+https://github.com/facebookresearch/segment-anything.git" || die "SAM 包失败"
$V -c "import segment_anything;print('SAM_PKG_OK')" || die "SAM 导入失败"
mkdir -p "$FADE"/{output,input,inversion,mask,saig_out,masks} "$FADE/HybridGL/checkpoints"
CK=$FADE/HybridGL/checkpoints/sam_vit_h_4b8939.pth
if [ ! -s "$CK" ]; then
  HF_HUB_ENABLE_HF_TRANSFER=0 $V - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_h_4b8939.pth", local_dir="/root/sam_dl")
print("HF_SAM", p)
PY
  cp -f /root/sam_dl/checkpoints/sam_vit_h_4b8939.pth "$CK" 2>/dev/null
  [ -s "$CK" ] || curl -L --retry 2 -o "$CK" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi
ls -lh "$CK" || die "SAM 权重缺失"

# ---------- 7 CogVideoX ----------
step cogvideox
if [ ! -f "$M/model_index.json" ]; then
  $V -c "import modelscope" 2>/dev/null || $V -m pip install -q modelscope $PIPM || die "modelscope 安装失败"
  $V - <<'PY' || die "CogVideoX-5b 下载失败"
from modelscope import snapshot_download
print(snapshot_download("ZhipuAI/CogVideoX-5b", local_dir="/root/saig/CogVideoX-5b"))
PY
fi
du -sh "$M" || die "模型目录缺失"

# ---------- 8 DAVIS ----------
step davis
mkdir -p /root/saig/DAVIS
if [ ! -d /root/saig/DAVISraw/DAVIS/JPEGImages ]; then
  HF_HUB_ENABLE_HF_TRANSFER=0 $V - <<'PY' || echo WARN_DAVIS
from huggingface_hub import hf_hub_download
p = hf_hub_download("rg4q33/DAVIS-2017-Unsupervised-trainval-480p",
                    "DAVIS-2017-Unsupervised-trainval-480p.zip",
                    repo_type="dataset", local_dir="/root/saig/DAVISraw")
print("DAVIS_ZIP", p)
PY
  cd /root/saig/DAVISraw 2>/dev/null && unzip -q -o DAVIS-2017-Unsupervised-trainval-480p.zip
fi
if [ -d /root/saig/DAVISraw/DAVIS/JPEGImages ]; then
  ln -sfn /root/saig/DAVISraw/DAVIS/JPEGImages /root/saig/DAVIS/JPEGImages
fi
ls /root/saig/DAVIS/JPEGImages/480p 2>/dev/null | wc -l

# ---------- 9 补丁 ----------
step patches
cd "$FADE" || die "cd FADE 失败"
[ -f edit_pipeline.sh ] && sed -i 's|^model_path=.*|model_path="/root/saig/CogVideoX-5b"|' edit_pipeline.sh
$V /root/apply_saig_patches.py 2>/dev/null || echo "WARN 补丁脚本未上传，跳过（稍后单独跑）"

# ---------- 10 验收 ----------
step verify
$V -c "import torch,clip,spacy,segment_anything,en_core_web_lg;print('ALL_IMPORTS_OK')" || die "依赖导入失败"
[ -f "$M/model_index.json" ] || die "缺 model_index.json"
grep -q "enable_model_cpu_offload" edit.py || echo "WARN offload 补丁未生效（等 apply_saig_patches 上传后补）"
df -h / | tail -1
echo "SETUP_A800_OK $(date -Is)"
