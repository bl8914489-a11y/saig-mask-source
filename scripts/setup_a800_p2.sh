#!/bin/bash
# A800 第二阶段：SAM / CogVideoX / DAVIS / 补丁 / 验收（spacy 走独立后台，不阻塞）
exec > /root/setup_a800_p2.log 2>&1
set -x
V=/root/venv310/bin/python
ROOT=/root/saig
FADE=$ROOT/FADE
M=$ROOT/CogVideoX-5b
PIPM="-i https://pypi.tuna.tsinghua.edu.cn/simple"
step() { echo "STEP $*"; }
die()  { echo "P2_FAIL $*"; exit 1; }

echo "P2_START $(date -Is)"

# ---------- SAM 包 + 权重 ----------
step sam_pkg
$V -c "import segment_anything" 2>/dev/null || \
  $V -m pip install -q segment-anything-py $PIPM || \
  $V -m pip install -q "git+https://github.com/facebookresearch/segment-anything.git" || die "SAM 包失败"
$V -c "import segment_anything;print('SAM_PKG_OK')" || die "SAM 导入失败"

step sam_weights
mkdir -p "$FADE/HybridGL/checkpoints"
CK=$FADE/HybridGL/checkpoints/sam_vit_h_4b8939.pth
if [ ! -s "$CK" ]; then
  HF_HUB_ENABLE_HF_TRANSFER=0 $V - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_h_4b8939.pth", local_dir="/root/sam_dl")
print("HF_SAM", p)
PY
  cp -f /root/sam_dl/checkpoints/sam_vit_h_4b8939.pth "$CK" 2>/dev/null
  [ -s "$CK" ] || curl -L --retry 2 -o "$CK" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi
ls -lh "$CK" || die "SAM 权重缺失"

# ---------- CogVideoX ----------
step cogvideox
if [ ! -f "$M/model_index.json" ]; then
  $V -c "import modelscope" 2>/dev/null || $V -m pip install -q modelscope $PIPM || die "modelscope 失败"
  $V - <<'PY' || die "CogVideoX 下载失败"
from modelscope import snapshot_download
print(snapshot_download("ZhipuAI/CogVideoX-5b", local_dir="/root/saig/CogVideoX-5b"))
PY
fi
du -sh "$M" || die "模型目录缺失"

# ---------- DAVIS ----------
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
  ( cd /root/saig/DAVISraw && unzip -q -o DAVIS-2017-Unsupervised-trainval-480p.zip )
fi
[ -d /root/saig/DAVISraw/DAVIS/JPEGImages ] && ln -sfn /root/saig/DAVISraw/DAVIS/JPEGImages /root/saig/DAVIS/JPEGImages
echo "DAVIS_SEQ=$(ls /root/saig/DAVIS/JPEGImages/480p 2>/dev/null | wc -l)"

# ---------- 补丁 ----------
step patches
cd "$FADE" || die "cd FADE 失败"
[ -f edit_pipeline.sh ] && sed -i 's|^model_path=.*|model_path="/root/saig/CogVideoX-5b"|' edit_pipeline.sh
$V /root/apply_saig_patches.py || die "补丁失败"
grep -q "enable_model_cpu_offload" edit.py || die "offload 未生效"
grep -q "SAIG_SINK.append" diffusion_core/guiders/opt_guiders_cogvideo.py || die "插桩缺失"
grep -q "SAIG_GATE_APPLIED" diffusion_core/guiders/guidance_editing_cogvideo.py || die "门控缺失"

# ---------- 验收 ----------
step verify
$V -c "import torch,clip,segment_anything;print('CORE_IMPORTS_OK', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))" || die "核心依赖失败"
$V -c "import en_core_web_lg;print('SPACY_OK')" 2>/dev/null || echo "SPACY_PENDING（后台下载中，不影响 AUROC/指标/编辑）"
[ -f "$M/model_index.json" ] || die "缺 model_index.json"
$V -m py_compile edit.py diffusion_core/guiders/opt_guiders_cogvideo.py diffusion_core/guiders/guidance_editing_cogvideo.py || die "编译失败"
df -h / | tail -1
echo "P2_OK $(date -Is)"
