#!/bin/bash
# =============================================================================
# SAIG 今晚 GPU 分阶段推进（幂等 / 带闸门 / 可断点续跑）
#
# 上机第一件事（语法闸门）：
#   bash -n /root/run_all_night.sh && echo SYNTAX_OK
#
# 用法：
#   bash /root/run_all_night.sh 0        # 预检：GPU / 磁盘 / venv / 依赖
#   bash /root/run_all_night.sh 1        # 重建环境（已存在则跳过，约 45–60 min）
#   bash /root/run_all_night.sh 2        # 打两个补丁（插桩 + 门控）
#   bash /root/run_all_night.sh 3        # 校验上传资产（8 g_id / 8 闸门图 / center / b0）
#   bash /root/run_all_night.sh 4 bear   # 补 inversion+mask（可传段名，默认 8 段全做）
#   bash /root/run_all_night.sh 5        # AUROC + 三零模型（闸门）
#   bash /root/run_all_night.sh 6        # B1 两臂：默认 bear goat
#   bash /root/run_all_night.sh 7        # ID-S / Sem-T（CPU）
#   bash /root/run_all_night.sh 8        # 生成带 SHA256 的备份清单，尚未下载或停机
#
# 说明：
#   - 台账写 /root/logs/runlog_b1.csv（inversion/mask/编辑），不改动 runlog_b0.csv
#   - 任一闸门不过：打印 GATE_FAIL 并以非零退出，绝不带错继续烧卡
#   - 日志双写到 /root/logs/tonight_<HHMMSS>.log
# =============================================================================
set -uo pipefail

ROOT=${SAIG_ROOT:-/root/saig}
FADE=$ROOT/FADE
SCR=${SAIG_SCRIPTS:-$ROOT/saig_project/scripts}
VENV=${SAIG_VENV:-/root/venv310/bin/python}
M=$ROOT/CogVideoX-5b
DAVIS=$ROOT/DAVIS/JPEGImages/480p
LOGDIR=${SAIG_LOG_DIR:-/root/logs}
CONTROL=${SAIG_CONTROL:-$LOGDIR/control}
LED=$LOGDIR/runlog_b1.csv
LOG=$LOGDIR/tonight_$(date +%Y%m%d_%H%M%S)_$$.log
mkdir -p "$LOGDIR" "$CONTROL/jobs"
[ -f "$LED" ] || echo "seq,stage,wall_s,rc" > "$LED"

say()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() { echo "[$(date +%H:%M:%S)] GATE_FAIL: $*" | tee -a "$LOG"; exit 1; }

ALLSEQ="bear camel cows dog elephant flamingo goat rhino"
SUBof() { case "$1" in bear) echo bear;; camel) echo camel;; cows) echo cow;;
  dog) echo dog;; elephant) echo elephant;; flamingo) echo flamingo;;
  goat) echo goat;; rhino) echo rhinoceros;; *) fail "未知段名 $1";; esac; }


check_stop() { [ ! -f "$CONTROL/STOP" ] || fail "预算或人工停止信号已设置"; }

run_logged() {
  local log=$1 pid start rc
  shift
  check_stop
  setsid "$@" >> "$log" 2>&1 &
  pid=$!
  if [ -r "/proc/$pid/stat" ]; then
    start=$(awk '{print $22}' "/proc/$pid/stat")
    printf '%s %s\n' "$pid" "$start" > "$CONTROL/jobs/$pid.pid"
  fi
  if [ -f "$CONTROL/STOP" ]; then
    kill -TERM -- "-$pid" 2>/dev/null || true
  fi
  wait "$pid"
  rc=$?
  rm -f "$CONTROL/jobs/$pid.pid"
  if [ -f "$CONTROL/STOP" ]; then return 130; fi
  return "$rc"
}

TGTOf() { case "$1" in
  bear) echo panda;; camel) echo horse;; cows) echo zebra;; dog) echo wolf;;
  elephant) echo rhinoceros;; flamingo) echo swan;; goat) echo sheep;;
  rhino) echo hippopotamus;; *) fail "未知段名 $1";; esac; }

verify_stage() {
  "$VENV" "$SCR/verify_downloads.py" stage --ledger "$LED" --seq "$1" \
    --stage "$2" --path "$3" --kind "$4" >> "$LOG" 2>&1
}

# ------------------------------------------------------------------ 0 预检
p0() {
  say "P0 预检"
  command -v nvidia-smi >/dev/null 2>&1 || fail "无 nvidia-smi"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | tee -a "$LOG"
  df -h / | tail -1 | tee -a "$LOG"
  if [ -x "$VENV" ]; then
    "$VENV" -c "import torch;print('TORCH',torch.__version__,torch.cuda.is_available())" 2>&1 | tee -a "$LOG"
    "$VENV" -c "import torch;assert torch.cuda.is_available()" || fail "venv 里 CUDA 不可用"
  else
    say "P0 提示：$VENV 不存在（P1 会重建）"
  fi
  say "PHASE0_OK"
}

# ------------------------------------------------------------------ 1 重建
p1() {
  if [ -d "$FADE/diffusion_core" ] && [ -x "$VENV" ] && [ -f "$M/model_index.json" ] && \
      "$VENV" -c "import torch,clip,spacy,segment_anything,diffusers,huggingface_hub,en_core_web_lg; assert torch.cuda.is_available(); assert huggingface_hub.__version__=='0.25.2'" >> "$LOG" 2>&1; then
    say "PHASE1_SKIP 环境基础验收通过"; return 0
  fi
  say "P1 重建环境（约 45–60 min）"
  [ -f /root/rebuild_env_step2.sh ] || fail "缺 /root/rebuild_env_step2.sh"
  local rebuildlog="$LOGDIR/rebuild_$(date +%Y%m%d_%H%M%S)_$$.log"
  run_logged "$rebuildlog" env SAIG_REBUILD_LOG="$rebuildlog" bash /root/rebuild_env_step2.sh || fail "环境重建失败，见 $rebuildlog"
  grep -q REBUILD_OK "$rebuildlog" || fail "rebuild 未打印 REBUILD_OK，看 $rebuildlog"
  [ -d "$M" ] || fail "缺模型目录 $M"
  [ -x "$VENV" ] || fail "缺 venv $VENV"
  say "PHASE1_OK"
}

# ------------------------------------------------------------------ 2 补丁
p2() {
  cd "$FADE" 2>/dev/null || fail "无 $FADE"
  say "P2 打补丁"
  out=$("$VENV" "$SCR/patch_fade_saig_rev1.py" 2>&1); echo "$out" | tee -a "$LOG"
  case "$out" in *"RESULT: PATCH_V2_OK"*|*"PATCH1_SKIP_ALREADY"*) ;; *) fail "插桩补丁失败";; esac
  out=$("$VENV" "$SCR/patch_guider_saig_gate.py" 2>&1); echo "$out" | tee -a "$LOG"
  case "$out" in *"RESULT: PATCH_OK"*|*"ALREADY_PATCHED"*) ;; *) fail "门控补丁失败";; esac
  grep -q "SAIG_GATE_APPLIED" diffusion_core/guiders/guidance_editing_cogvideo.py \
    || fail "门控注入点缺失"
  grep -q "SAIG_SINK.append" diffusion_core/guiders/opt_guiders_cogvideo.py \
    || fail "插桩点缺失"
  "$VENV" -m py_compile edit.py diffusion_core/guiders/opt_guiders_cogvideo.py \
    diffusion_core/guiders/guidance_editing_cogvideo.py \
    || fail "补丁后源码编译失败（用 *.bak_saig 回滚）"
  say "PHASE2_OK"
}

# ------------------------------------------------------------------ 3 资产
p3() {
  say "P3 校验上传资产"
  local n m p i k
  n=$(ls "$FADE"/saig_out/*_gid.npy 2>/dev/null | wc -l); say "P3 g_id=$n"
  [ "$n" = "8" ] || fail "saig_out 应为 8 个 .npy"
  m=$(ls "$FADE"/masks/gid_*.npy 2>/dev/null | grep -v gid_center | wc -l); say "P3 闸门图=$m"
  [ "$m" = "8" ] || fail "masks/ 应为 8 个 gid_<seq>.npy"
  [ -f "$FADE/masks/gid_center.npy" ] || fail "缺 masks/gid_center.npy"
  p=$(ls "$FADE"/output/b0_*.mp4 2>/dev/null | wc -l); say "P3 b0 成片=$p"
  i=$(ls "$FADE"/inversion/*.pt 2>/dev/null | wc -l); say "P3 inversion=$i"
  k=$(ls "$FADE"/mask/*.pt 2>/dev/null | wc -l); say "P3 mask=$k"
  say "PHASE3_OK"
}

# ------------------------------------------------------------------ 4 inversion+mask
p4() {
  cd "$FADE" 2>/dev/null || fail "无 $FADE"
  local seqs="${*:-$ALLSEQ}"
  [ -f "$LED" ] || echo "seq,stage,wall_s,rc" > "$LED"
  say "P4 目标段：$seqs"
  for SEQ in $seqs; do
    check_stop
    SUB=$(SUBof "$SEQ") || fail "未知段名 $SEQ"
    if [ ! -f "input/$SEQ.mp4" ]; then
      [ -d "$DAVIS/$SEQ" ] || fail "既无 input/$SEQ.mp4 也无 DAVIS/$SEQ"
      say "P4 生成 input/$SEQ.mp4"
      ffmpeg -y -loglevel error -framerate 4 -start_number 0 -i "$DAVIS/$SEQ/%05d.jpg" \
        -frames:v 49 -c:v libx264 -pix_fmt yuv420p -crf 18 "input/$SEQ.mp4" || fail "ffmpeg 失败 $SEQ"
    fi
    if [ ! -f "inversion/$SEQ.pt" ]; then
      say "P4 inversion $SEQ（约 4 min）"
      t0=$(date +%s)
      run_logged "$LOG" "$VENV" ddim_inversion.py --model_path "$M" --prompt "a $SUB" \
        --video_path "input/$SEQ.mp4" --output_path "inversion/$SEQ.partial.pt" --fps 4 --device cuda:0
      rc=$?
      if [ "$rc" -eq 0 ]; then
        "$VENV" "$SCR/verify_downloads.py" tensor --kind inversion --path "inversion/$SEQ.partial.pt" >> "$LOG" 2>&1
        rc=$?
      fi
      if [ "$rc" -eq 0 ]; then mv "inversion/$SEQ.partial.pt" "inversion/$SEQ.pt" || rc=$?; fi
      echo "$SEQ,inversion,$(( $(date +%s) - t0 )),$rc" >> "$LED"
      [ $rc -eq 0 ] || fail "inversion 失败 $SEQ（见 $LOG）"
    else
      verify_stage "$SEQ" inversion "inversion/$SEQ.pt" inversion || fail "已有 inversion 无有效成功记录或文件损坏 $SEQ"
      say "P4 SKIP inversion $SEQ"
    fi
    if [ ! -f "mask/$SEQ.pt" ]; then
      say "P4 mask $SEQ（约 25–65 min）"
      t0=$(date +%s)
      run_logged "$LOG" "$VENV" generate_mask.py --input_video "input/$SEQ.mp4" --output_path "mask/$SEQ.partial.pt" \
        --ref_text "a $SUB" --device cuda:0
      rc=$?
      if [ "$rc" -eq 0 ]; then
        "$VENV" "$SCR/verify_downloads.py" tensor --kind mask --path "mask/$SEQ.partial.pt" >> "$LOG" 2>&1
        rc=$?
      fi
      if [ "$rc" -eq 0 ]; then mv "mask/$SEQ.partial.pt" "mask/$SEQ.pt" || rc=$?; fi
      echo "$SEQ,mask,$(( $(date +%s) - t0 )),$rc" >> "$LED"
      [ $rc -eq 0 ] || fail "mask 失败 $SEQ（见 $LOG）"
    else
      verify_stage "$SEQ" mask "mask/$SEQ.pt" mask || fail "已有 mask 无有效成功记录或文件损坏 $SEQ"
      say "P4 SKIP mask $SEQ"
    fi
  done
  check_stop
  say "PHASE4_OK"
}

# ------------------------------------------------------------------ 5 AUROC
p5() {
  cd "$FADE" || fail "无 $FADE"
  local result="$LOGDIR/auroc_tonight_$(date +%Y%m%d_%H%M%S).csv"
  run_logged "$LOG" "$VENV" "$SCR/auroc_gid.py" --out "$result" || fail "AUROC 计算失败"
  grep -q "RESULT: AUROC_DONE rows=8" "$LOG" || fail "AUROC 需要 8 段"
  grep -q "GATE_AUROC" "$LOG" || fail "AUROC 缺判定行"
  say "PHASE5_OK 计算已完成，效果判定见 $LOG"
}

# ------------------------------------------------------------------ 6 B1 两臂
p6() {
  cd "$FADE" || fail "无 $FADE"
  local seqs="${*:-bear goat}" runlog tmp
  for SEQ in $seqs; do
    SUB=$(SUBof "$SEQ") || fail "未知段名"
    TGT=$(TGTOf "$SEQ") || fail "未知段名"
    verify_stage "$SEQ" inversion "inversion/$SEQ.pt" inversion || fail "反演未验收 $SEQ"
    verify_stage "$SEQ" mask "mask/$SEQ.pt" mask || fail "掩码未验收 $SEQ"
    for ARM in gate center; do
      check_stop
      if [ "$ARM" = gate ]; then G="$FADE/masks/gid_$SEQ.npy"; OUT="output/b1_$SEQ.mp4"
      else G="$FADE/masks/gid_center.npy"; OUT="output/b1c_$SEQ.mp4"; fi
      [ -f "$G" ] || fail "缺闸门图 $G"
      if [ -f "$OUT" ]; then
        verify_stage "$SEQ" "b1_$ARM" "$OUT" video || fail "已有视频未通过验收 $OUT"
        say "P6 SKIP $ARM $SEQ"
        continue
      fi
      runlog="$LOGDIR/edit_${SEQ}_${ARM}_$(date +%Y%m%d_%H%M%S)_$$.log"
      tmp="${OUT%.mp4}.partial.mp4"
      t0=$(date +%s)
      run_logged "$runlog" env SAIG_GID="$G" "$VENV" edit.py --model_path "$M" \
        --init_prompt "a $SUB" --edit_prompt "a $TGT" --device cuda:0 --dtype bf16 \
        --input_video_path "input/$SEQ.mp4" --latent_trajectory_path "inversion/$SEQ.pt" \
        --mask_path "mask/$SEQ.pt" --config_path configs/bear.yaml --fps 4 --output_video_path "$tmp"
      rc=$?
      if [ "$rc" -eq 0 ]; then grep -q 'SAIG_GATE_APPLIED' "$runlog" || rc=65; fi
      if [ "$rc" -eq 0 ]; then "$VENV" "$SCR/verify_downloads.py" video --path "$tmp" >> "$runlog" 2>&1; rc=$?; fi
      if [ "$rc" -eq 0 ]; then mv "$tmp" "$OUT" || rc=$?; fi
      echo "$SEQ,b1_$ARM,$(( $(date +%s) - t0 )),$rc" >> "$LED"
      [ "$rc" -eq 0 ] || fail "$SEQ $ARM 失败，见 $runlog"
      say "P6 VERIFIED $SEQ $ARM $OUT"
    done
  done
  say "PHASE6_OK"
}

# ------------------------------------------------------------------ 7 指标
p7() {
  cd "$FADE" || fail "无 $FADE"
  local seqs="${*:-bear goat}" out
  for SEQ in $seqs; do
    check_stop
    SUB=$(SUBof "$SEQ") || fail "未知段名"
    TGT=$(TGTOf "$SEQ") || fail "未知段名"
    for file in "output/b0_$SEQ.mp4" "output/b1_$SEQ.mp4" "output/b1c_$SEQ.mp4"; do
      "$VENV" "$SCR/verify_downloads.py" video --path "$file" >> "$LOG" 2>&1 || fail "视频未通过验收 $file"
    done
    out="$LOGDIR/metrics_$SEQ.csv"
    run_logged "$LOG" env HF_HUB_ENABLE_HF_TRANSFER=0 "$VENV" "$SCR/metrics_compare.py" \
      --seq "$SEQ" --src "input/$SEQ.mp4" --ann "$ROOT/DAVIS/Annotations/480p/$SEQ" \
      --init "a $SUB" --target "a $TGT" --out "$out" \
      --videos "b0=output/b0_$SEQ.mp4" "gate=output/b1_$SEQ.mp4" "center=output/b1c_$SEQ.mp4" \
      || fail "指标计算失败 $SEQ"
  done
  "$VENV" "$SCR/verify_downloads.py" metrics --directory "$LOGDIR" --seqs $seqs \
    --out "$LOGDIR/metrics_all.csv" >> "$LOG" 2>&1 || fail "指标汇总未通过验收"
  say "PHASE7_OK $LOGDIR/metrics_all.csv"
}

p8() {
  "$VENV" "$SCR/make_output_manifest.py" prepare --root "$ROOT" --logs "$LOGDIR" \
    --destination "$ROOT/backup_snapshot" || fail "备份清单生成失败"
  say "BACKUP_READY 本机拉取并校验后方可确认备份完成"
}

cmd=${1:-}
[ -z "$cmd" ] && { sed -n '3,25p' "$0"; exit 0; }
shift || true
if [ "$cmd" != 8 ]; then check_stop; fi
exec 9> "$CONTROL/phase.lock"
flock -n 9 || fail "已有阶段正在执行"
case "$cmd" in
  0) p0 ;;
  1) p1 ;;
  2) p2 ;;
  3) p3 ;;
  4) p4 "$@" ;;
  5) p5 ;;
  6) p6 "$@" ;;
  7) p7 "$@" ;;
  8) p8 ;;
  *) echo "未知阶段：$cmd"; sed -n '10,23p' "$0"; exit 2 ;;
esac
