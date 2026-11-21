#!/bin/bash
# =============================================================================
# 重建续跑 v3：从 spacy_model 失败处继续（venv 已建、torch/requirements 已装、两个 repo 已 clone）
# 关键修正：
#   1. venv 没有 pip → 一律用 `uv pip install --python $V`
#   2. spacy 模型不用 `spacy download`，直接装与 spacy 3.7.6 匹配的 3.7.1 wheel
#   3. CogVideoX 用 snapshot_download（比 CLI 稳），并关掉 hf_transfer
# =============================================================================
exec > /root/rebuild_resume.log 2>&1
set -x

step() { echo "RESUME_STEP $*"; }
die()  { echo "RESUME_FAIL $*"; exit 1; }

V=/root/venv310/bin/python
export PATH=/root/.local/bin:$PATH
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "RESUME_START $(date -Is)"
[ -x "$V" ] || die "venv 不存在"
cd /root/saig/FADE || die "无 FADE 目录"

# ---------- spacy 模型（3.7.1 对应 spacy 3.7.x） ----------
step "spacy_wheel"
if "$V" -c "import spacy.util,sys;spacy.util.get_package_path('en_core_web_lg');sys.exit(0)" 2>/dev/null; then
  echo "SPACY_MODEL_ALREADY"
else
  uv pip install --python "$V" \
    "en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl" \
    || die "spacy 模型 wheel 安装失败"
fi

# ---------- clip fork 路径 ----------
step "clip_fork"
SP=/root/venv310/lib/python3.10/site-packages
echo "/root/saig/FADE/HybridGL/third_party/modified_CLIP" > "$SP/zz_clip_fork.pth"

# ---------- 目录 ----------
step "dirs"
mkdir -p HybridGL/checkpoints inversion mask output input saig_out masks

# ---------- SAM 权重 ----------
step "sam_weights"
if [ ! -s HybridGL/checkpoints/sam_vit_h_4b8939.pth ]; then
  curl -L --retry 3 -o HybridGL/checkpoints/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth || die "SAM 下载失败"
fi

# ---------- CogVideoX-5b ----------
step "cogvideox"
if [ ! -f /root/saig/CogVideoX-5b/model_index.json ]; then
  uv pip install --python "$V" -q -U huggingface_hub || die "hf_hub 安装失败"
  HF_HUB_ENABLE_HF_TRANSFER=0 "$V" - <<'PY' || die "CogVideoX 下载失败"
from huggingface_hub import snapshot_download
p = snapshot_download("zai-org/CogVideoX-5b", local_dir="/root/saig/CogVideoX-5b",
                      local_dir_use_symlinks=False, resume_download=True,
                      max_workers=8)
print("SNAPSHOT", p)
PY
fi

# ---------- DAVIS ----------
step "davis"
if [ ! -d /root/saig/DAVIS/JPEGImages ]; then
  cd /root/saig || die "cd /root/saig 失败"
  curl -L --retry 3 -o davis.zip https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip || die "DAVIS 下载失败"
  unzip -q davis.zip -d DAVIS || die "DAVIS 解压失败"
  rm -f davis.zip
  [ -d DAVIS/DAVIS ] && { mv DAVIS/DAVIS T && rm -rf DAVIS && mv T DAVIS; }
fi

# ---------- FADE 补丁 ----------
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

# ---------- 验收 ----------
step "verify"
"$V" -c "import torch,clip,spacy,segment_anything;print('IMPORTS_OK',torch.__version__)" || die "依赖导入失败"
"$V" -c "import torch;assert torch.cuda.is_available();print('CUDA_OK')" || die "CUDA 不可用"
"$V" -c "import en_core_web_lg;print('SPACY_MODEL_OK')" || die "spacy 模型导入失败"
[ -f /root/saig/CogVideoX-5b/model_index.json ] || die "缺 model_index.json"
[ -d /root/saig/DAVIS/JPEGImages/480p/bear ] || die "缺 DAVIS bear"
grep -q "enable_model_cpu_offload" edit.py || die "offload 补丁未生效"
grep -q "self.inv_latents\[0\].shape\[1\]" diffusion_core/guiders/guidance_editing_cogvideo.py || die "rope 补丁未生效"
df -h / | tail -1
echo "REBUILD_OK $(date -Is)"
