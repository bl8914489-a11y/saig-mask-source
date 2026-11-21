#!/bin/bash
# =============================================================================
# 国内云（优云智算等）环境重建：带镜像回退，适配 GitHub/HF 不通或很慢的情况
#
# 用法：
#   bash rebuild_env_step1.sh              # 自动探测并选择镜像
#   PIP=tsinghua HF=hf-mirror bash rebuild_env_step1.sh   # 手动指定
#
# 与 rebuild_env_step3_resume.sh 的区别：所有外部下载都有国内替代源，失败会明确报出是哪一段
# =============================================================================
exec > /root/rebuild_cn.log 2>&1
set -x

step() { echo "STEP $*"; }
die()  { echo "REBUILD_FAIL $*"; exit 1; }
have() { curl -s -o /dev/null -w "%{http_code}" -m 8 -L "$1" 2>/dev/null; }

echo "REBUILD_CN_START $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "无 GPU 驱动"
df -h / | tail -1

# ---------------- 源选择 ----------------
step "choose_mirrors"
PIP_HOME=${PIP:-auto}
HF_HOME_KIND=${HF:-auto}

if [ "$PIP_HOME" = "auto" ]; then
  [ "$(have https://download.pytorch.org/whl/cu124/)" = "200" ] && PIP_HOME=official || PIP_HOME=tsinghua
fi
if [ "$HF_HOME_KIND" = "auto" ]; then
  if [ "$(have https://huggingface.co)" = "200" ]; then HF_HOME_KIND=official
  elif [ "$(have https://hf-mirror.com)" = "200" ]; then HF_HOME_KIND=hfmirror
  elif [ "$(have https://www.modelscope.cn)" = "200" ]; then HF_HOME_KIND=modelscope
  else HF_HOME_KIND=blocked; fi
fi
echo "PIP_HOME=$PIP_HOME HF_HOME_KIND=$HF_HOME_KIND"

case "$PIP_HOME" in
  official) PIPIDX=https://download.pytorch.org/whl/cu124 ;;
  *)        PIPIDX=https://pypi.tuna.tsinghua.edu.cn/simple ;;
esac
case "$HF_HOME_KIND" in
  hfmirror) export HF_ENDPOINT=https://hf-mirror.com ;;
esac
export HF_HUB_ENABLE_HF_TRANSFER=0
export PIP_INDEX_URL="$PIPIDX"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn download.pytorch.org pypi.org files.pythonhosted.org" || true

# ---------------- 1 uv ----------------
step "install_uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv 安装失败（astral.sh 不通时：pip3 install uv）"
fi
export PATH=/root/.local/bin:$PATH
uv --version || die "uv 不可用"
command -v uv >/dev/null 2>&1 || pip3 install -q uv

# ---------------- 2 venv ----------------
step "venv310"
if [ ! -x /root/venv310/bin/python ]; then
  uv python install 3.10 || die "python3.10 下载失败"
  uv venv --python 3.10 /root/venv310 || die "venv 创建失败"
fi
V=/root/venv310/bin/python
$V -V || die "venv python 不可用"

# ---------------- 3 torch ----------------
step "torch"
if ! $V -c "import torch" 2>/dev/null; then
  if [ "$PIP_HOME" = "official" ]; then
    uv pip install --python "$V" --no-cache torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124 || die "torch(official) 失败"
  else
    # 清华镜像装 torch（CUDA 版，体积大但国内快）
    uv pip install --python "$V" --no-cache torch==2.5.1 torchvision==0.20.1 \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple || die "torch(tsinghua) 失败"
  fi
fi
$V -c "import torch;print('TORCH',torch.__version__,torch.cuda.is_available())" || die "torch 导入失败"
$V -c "import torch;assert torch.cuda.is_available()" || echo "WARN torch 无 CUDA（可能装了 CPU 版，需换源重装）"

# ---------------- 4 代码 ----------------
step "clone_code"
GH=${GH:-auto}
if [ "$GH" = "auto" ]; then
  [ "$(have https://github.com)" = "200" ] && GH=official || GH=ghproxy
fi
clone() {  # $1=repo 路径 $2=目标
  if [ "$GH" = "official" ]; then
    git clone --depth 1 "https://github.com/$1.git" "$2"
  else
    git clone --depth 1 "https://ghproxy.net/https://github.com/$1.git" "$2" ||
    git clone --depth 1 "https://gitclone.com/github.com/$1.git" "$2"
  fi
}
mkdir -p /root/saig && cd /root/saig || die "mkdir 失败"
[ -d FADE/.git ] || clone "EternalEvan/FADE" FADE || die "FADE clone 失败"
[ -d FADE/HybridGL/.git ] || clone "fhgyuanshen/HybridGL" FADE/HybridGL || die "HybridGL clone 失败"

# ---------------- 5 依赖 ----------------
step "requirements"
cd /root/saig/FADE || die "进入 FADE 失败"
uv pip install --python "$V" -r requirements.txt || die "requirements 失败"
uv pip install --python "$V" -q "setuptools<81"
uv pip install --python "$V" -q -U "huggingface_hub==0.25.2" || die "hf_hub 锁定失败"
uv pip install --python "$V" -q "git+https://github.com/facebookresearch/segment-anything.git" 2>/dev/null ||
  uv pip install --python "$V" -q "segment-anything-py" || echo "WARN segment_anything 未装（掩码生成会失败）"

step "spacy_model"
if ! $V -c "import en_core_web_lg" 2>/dev/null; then
  WHL=https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
  uv pip install --python "$V" "en-core-web-lg @ $WHL" 2>/dev/null ||
  uv pip install --python "$V" "en-core-web-lg @ https://ghproxy.net/$WHL" 2>/dev/null ||
  $V -m spacy download en_core_web_lg || die "spacy 模型失败（三个源都不通）"
fi
SP=/root/venv310/lib/python3.10/site-packages
echo "/root/saig/FADE/HybridGL/third_party/modified_CLIP" > "$SP/zz_clip_fork.pth"

# ---------------- 6 权重 ----------------
step "sam_weights"
mkdir -p HybridGL/checkpoints inversion mask output input saig_out masks
if [ ! -s HybridGL/checkpoints/sam_vit_h_4b8939.pth ]; then
  URL=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  curl -L --retry 3 -o HybridGL/checkpoints/sam_vit_h_4b8939.pth "$URL" ||
  curl -L --retry 3 -o HybridGL/checkpoints/sam_vit_h_4b8939.pth \
    "https://hf-mirror.com/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth" ||
    die "SAM 权重失败"
fi

step "cogvideox"
if [ ! -f /root/saig/CogVideoX-5b/model_index.json ]; then
  case "$HF_HOME_KIND" in
    official|hfmirror)
      $V - <<'PY' || die "CogVideoX 下载失败（HF 通道）"
from huggingface_hub import snapshot_download
print(snapshot_download("zai-org/CogVideoX-5b", local_dir="/root/saig/CogVideoX-5b",
                        max_workers=8, resume_download=True))
PY
      ;;
    modelscope)
      uv pip install --python "$V" -q modelscope || die "modelscope 安装失败"
      $V - <<'PY' || die "CogVideoX 下载失败（ModelScope 通道）"
from modelscope import snapshot_download
print(snapshot_download("ZhipuAI/CogVideoX-5b", local_dir="/root/saig/CogVideoX-5b"))
PY
      ;;
    *) die "HF 与 ModelScope 都不通：无法取得 CogVideoX-5b" ;;
  esac
fi

step "davis"
if [ ! -d /root/saig/DAVIS/JPEGImages ]; then
  cd /root/saig || die "cd 失败"
  curl -L --retry 3 -o davis.zip https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip ||
    die "DAVIS 下载失败（ETH 不通时：从本机迁移包上传 Annotations，或只做已有段）"
  unzip -q davis.zip -d DAVIS && rm -f davis.zip
  [ -d DAVIS/DAVIS ] && { mv DAVIS/DAVIS T && rm -rf DAVIS && mv T DAVIS; }
fi

# ---------------- 7 FADE 补丁 ----------------
step "fade_patches"
cd /root/saig/FADE || die "进入 FADE 失败"
[ -f edit_pipeline.sh ] && sed -i 's|^model_path=.*|model_path="/root/saig/CogVideoX-5b"|' edit_pipeline.sh
grep -q "enable_model_cpu_offload" edit.py || sed -i 's#\.to(args\.device)#; pipe.enable_model_cpu_offload()#' edit.py
$V - <<'PY' || die "rope 补丁失败"
p = "diffusion_core/guiders/guidance_editing_cogvideo.py"
s = open(p, encoding="utf-8").read()
if "self.inv_latents[0].shape[1]" not in s:
    s = s.replace("_prepare_rotary_positional_embeddings(480, 720, 3,",
                  "_prepare_rotary_positional_embeddings(480, 720, self.inv_latents[0].shape[1],")
    open(p, "w", encoding="utf-8").write(s)
    print("ROPE_PATCHED")
PY

# ---------------- 8 验收 ----------------
step "verify"
$V -c "import torch,clip,spacy,segment_anything;print('IMPORTS_OK',torch.__version__)" || die "依赖导入失败"
$V -c "import torch;assert torch.cuda.is_available();print('CUDA_OK')" || die "CUDA 不可用"
$V -c "import en_core_web_lg;print('SPACY_OK')" || die "spacy 模型导入失败"
[ -f /root/saig/CogVideoX-5b/model_index.json ] || die "缺 model_index.json"
grep -q "enable_model_cpu_offload" edit.py || die "offload 补丁未生效"
df -h / | tail -1
echo "REBUILD_CN_OK $(date -Is)"
