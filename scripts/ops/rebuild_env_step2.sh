#!/bin/bash
# =============================================================================
# FADE/SAIG 环境重建 v2（适配「裸镜像 + python3.12 无 torch」的 RunPod 容器）
#
# 与旧 rebuild.sh 的差别：
#   1. 去掉开头的「系统 python 必须已有 CUDA torch」硬检查（裸镜像必失败）
#   2. 所有 pip 安装显式绑定 venv 解释器（uv venv 默认不带 pip）
#   3. hf 下载改用 venv 内的 huggingface_hub CLI
#   4. 每一步打印 REBUILD_STEP，便于尾部追踪进度
#   5. 末尾用 venv 解释器做 CUDA + 依赖校验，通过才打印 REBUILD_OK
# =============================================================================
exec > "${SAIG_REBUILD_LOG:-/root/rebuild.log}" 2>&1
set -uo pipefail
set -x

step() { echo "REBUILD_STEP $*"; }
die()  { echo "REBUILD_FAIL $*"; exit 1; }

echo "REBUILD_START $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h / | tail -1

# ---------- 1. uv ----------
step "install_uv"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
uv --version || die "uv 安装失败"

# ---------- 2. venv 3.10 ----------
step "venv310"
[ -x /root/venv310/bin/python ] || {
  uv python install 3.10 || die "python3.10 下载失败"
  uv venv --python 3.10 /root/venv310 || die "venv 创建失败"
}
V=/root/venv310/bin/python
$V -V || die "venv python 不可用"

# ---------- 3. torch ----------
step "torch_cu124"
uv pip install --python "$V" --no-cache torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124 || die "torch 安装失败"
$V -c "import torch;print('TORCH',torch.__version__,torch.cuda.is_available())" || die "torch 导入失败"

# ---------- 4. 代码 ----------
step "clone_FADE"
mkdir -p /root/saig && cd /root/saig || die "mkdir /root/saig 失败"
[ -d FADE/.git ] || git clone --depth 1 https://github.com/EternalEvan/FADE.git FADE || die "FADE clone 失败"
[ -d FADE/HybridGL/.git ] || git clone --depth 1 https://github.com/fhgyuanshen/HybridGL.git FADE/HybridGL || die "HybridGL clone 失败"

# ---------- 5. 依赖 ----------
step "pip_requirements"
cd /root/saig/FADE || die "进入 FADE 失败"
uv pip install --python "$V" -r requirements.txt || die "requirements 安装失败"
uv pip install --python "$V" -q "setuptools<81" || die "setuptools 失败"
uv pip install --python "$V" 'huggingface_hub==0.25.2' \
  'git+https://github.com/facebookresearch/segment-anything.git' || die "锁定 hf_hub 或安装 SAM 失败"

step "spacy_model"
uv pip install --python "$V" \
  'en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl' \
  || die "spacy 模型失败"

SP=/root/venv310/lib/python3.10/site-packages
echo "/root/saig/FADE/HybridGL/third_party/modified_CLIP" > "$SP/zz_clip_fork.pth"

# ---------- 6. 权重与数据 ----------
step "dirs"
mkdir -p HybridGL/checkpoints inversion mask output input saig_out masks

step "sam_weights"
[ -s HybridGL/checkpoints/sam_vit_h_4b8939.pth ] || \
  curl -L -o HybridGL/checkpoints/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth || die "SAM 权重下载失败"

step "cogvideox"
if [ ! -f /root/saig/CogVideoX-5b/model_index.json ]; then
  uv pip install --python "$V" 'huggingface_hub==0.25.2' || die "hf_hub 安装失败"
  "$V" -m huggingface_hub.commands.huggingface_cli download zai-org/CogVideoX-5b \
    --local-dir /root/saig/CogVideoX-5b || die "CogVideoX 下载失败"
fi

step "davis"
if [ ! -d /root/saig/DAVIS/JPEGImages ]; then
  cd /root/saig || die "cd /root/saig 失败"
  curl -L -o davis.zip https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip || die "DAVIS 下载失败"
  unzip -q davis.zip -d DAVIS || die "DAVIS 解压失败"
  rm -f davis.zip
  [ -d DAVIS/DAVIS ] && { mv DAVIS/DAVIS T && rm -rf DAVIS && mv T DAVIS; }
fi

# ---------- 7. FADE 补丁（offload + rope 帧数） ----------
step "fade_patches"
cd /root/saig/FADE || die "进入 FADE 失败"
[ -f edit_pipeline.sh ] && sed -i 's|^model_path=.*|model_path="/root/saig/CogVideoX-5b"|' edit_pipeline.sh
grep -q "enable_model_cpu_offload" edit.py || sed -i 's#\.to(args\.device)#; pipe.enable_model_cpu_offload()#' edit.py
"$V" - <<'PY' || die "rope 补丁失败"
p = "diffusion_core/guiders/guidance_editing_cogvideo.py"
s = open(p, encoding="utf-8").read()
if "self.inv_latents[0].shape[1]" in s:
    print("ROPE_ALREADY")
else:
    s = s.replace("_prepare_rotary_positional_embeddings(480, 720, 3,",
                  "_prepare_rotary_positional_embeddings(480, 720, self.inv_latents[0].shape[1],")
    open(p, "w", encoding="utf-8").write(s)
    print("ROPE_PATCHED")
PY

# ---------- 8. 验收 ----------
step "verify"
"$V" -c "import torch,clip,spacy,segment_anything;print('IMPORTS_OK',torch.__version__)" || die "依赖导入失败"
"$V" -c "import torch;assert torch.cuda.is_available();print('CUDA_OK')" || die "CUDA 不可用"
[ -f /root/saig/CogVideoX-5b/model_index.json ] || die "缺 model_index.json"
[ -d /root/saig/DAVIS/JPEGImages/480p/bear ] || die "缺 DAVIS bear"
grep -q "enable_model_cpu_offload" edit.py || die "offload 补丁未生效"
df -h / | tail -1
echo "REBUILD_OK $(date -Is)"
