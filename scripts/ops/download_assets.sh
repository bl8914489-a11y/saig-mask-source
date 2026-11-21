#!/bin/bash
# =============================================================================
# 重建完成后：资产就位 + 双补丁 + 编译校验 + 数量核对
# 用法：bash /root/download_assets.sh
# 判据：末尾出现 DEPLOY_OK，且 output=21 / gid=8 / masks=9 / 补丁各 1 处
# =============================================================================
exec > /root/deploy.log 2>&1
set -x
die() { echo "DEPLOY_FAIL $*"; exit 1; }

V=/root/venv310/bin/python
S=/root/stage/scripts
F=/root/saig/FADE
P=/root/saig/saig_project/scripts

echo "DEPLOY_START $(date -Is)"
[ -x "$V" ] || die "venv 不存在"
[ -d "$F/diffusion_core" ] || die "FADE 不存在"
[ -d "$S" ] || die "stage/scripts 不存在"

mkdir -p "$F"/{output,saig_out,masks,input,inversion,mask} "$P" /root/logs

# ---------- 脚本 ----------
cp -f "$S"/*.py "$S"/*.sh "$P"/ 2>/dev/null
cp -f "$S/export_gid_rev1.py" "$P/export_gid_rev1.py"        # 回退兼容名
cp -f "$S/run_all_night.sh" /root/run_all_night.sh
cp -f "$S/budget_guard.sh" /root/budget_guard.sh
chmod +x /root/run_all_night.sh /root/budget_guard.sh

# ---------- 已有产物 ----------
cp -f /root/stage/output/*.mp4   "$F/output/"  2>/dev/null
cp -f /root/stage/saig_out/*.npy "$F/saig_out/" 2>/dev/null
cp -f /root/stage/masks/*.npy    "$F/masks/"    2>/dev/null

# ---------- 补丁 ----------
cd "$F" || die "cd FADE 失败"
echo "=== PATCH1 插桩 ==="
"$V" "$P/patch_fade_saig_rev1.py" 2>&1 | tail -n 6 || die "插桩补丁执行失败"
grep -q "SAIG_SINK.append" diffusion_core/guiders/opt_guiders_cogvideo.py || die "插桩点缺失"
echo "=== PATCH2 门控 ==="
"$V" "$P/patch_guider_saig_gate.py" 2>&1 | tail -n 8 || die "门控补丁执行失败"
grep -q "SAIG_GATE_APPLIED" diffusion_core/guiders/guidance_editing_cogvideo.py || die "门控注入点缺失"

# ---------- 编译校验 ----------
"$V" -m py_compile edit.py \
  diffusion_core/guiders/opt_guiders_cogvideo.py \
  diffusion_core/guiders/guidance_editing_cogvideo.py || die "补丁后源码编译失败"

# ---------- 数量核对 ----------
NOUT=$(ls -1 "$F"/output/*.mp4 2>/dev/null | wc -l)
NGID=$(ls -1 "$F"/saig_out/*_gid.npy 2>/dev/null | wc -l)
NMSK=$(ls -1 "$F"/masks/gid_*.npy 2>/dev/null | wc -l)
echo "COUNTS output=$NOUT gid=$NGID masks=$NMSK"
[ "$NOUT" = "21" ] || echo "WARN output 不是 21（继续）"
[ "$NGID" = "8" ] || die "g_id 应为 8"
[ "$NMSK" = "9" ] || die "masks 应为 9（8 段 + center）"
echo "DEPLOY_OK $(date -Is)"
