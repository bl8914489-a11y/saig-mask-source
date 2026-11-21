# -*- coding: utf-8 -*-
"""A/B 测试：同一段编辑，开 / 不开 cpu offload，测真实 s/it 与峰值显存。

用法（pod 上，FADE 目录内）：
    /root/venv310/bin/python /root/ab_test_offload.py camel 12

判据：
    per_step(offload_off) < 0.8 × per_step(offload_on)  → 大显存卡值得加钱
    offload_off 报 OOM                                  → 这台卡必须走 offload
"""
import atexit
import os
import re
import subprocess
import sys
import time

SEQ = sys.argv[1] if len(sys.argv) > 1 else "camel"
STEPS = sys.argv[2] if len(sys.argv) > 2 else "12"
F = "/root/saig/FADE"
V = "/root/venv310/bin/python"
OG = "/root/logs/ab_offload_%s_%s.log" % (SEQ, time.strftime("%H%M%S"))

SUB = {"bear": "bear", "camel": "camel", "cows": "cow", "dog": "dog", "elephant": "elephant",
       "flamingo": "flamingo", "goat": "goat", "rhino": "rhinoceros"}.get(SEQ, SEQ)
TGT = {"bear": "panda", "camel": "horse", "cows": "zebra", "dog": "wolf",
       "elephant": "rhinoceros", "flamingo": "swan", "goat": "sheep",
       "rhino": "hippopotamus"}.get(SEQ, "panda")

RUNNER = "_ab_runner.py"
RUNNER_SRC = '''
import atexit, os, runpy, sys, torch

def _report():
    try:
        print("PEAK_GB %.2f" % (torch.cuda.max_memory_allocated() / 1024**3), flush=True)
    except Exception:
        pass

atexit.register(_report)
target = os.environ["AB_TARGET"]
sys.argv = ["edit.py"] + os.environ["AB_ARGS"].split("\\x1f")
sys.path.insert(0, os.getcwd())
torch.cuda.reset_peak_memory_stats()
try:
    runpy.run_path(target, run_name="__main__")
except SystemExit:
    pass
except Exception as e:
    print("AB_EXC", type(e).__name__, str(e)[:200], flush=True)
'''


def log(msg):
    print(msg, flush=True)
    with open(OG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def prepare_edit(offload_off):
    src = open(os.path.join(F, "edit.py"), encoding="utf-8").read()
    if not offload_off:
        return "edit.py"
    out = os.path.join(F, "_edit_nooffload.py")
    s = src.replace("pipe.enable_model_cpu_offload()", "pass  # AB: offload disabled")
    s = s.replace(".to(args.device)", ".to(args.device)  # AB: keep on device")
    open(out, "w", encoding="utf-8").write(s)
    return "_edit_nooffload.py"


def run(mode):
    off = (mode == "off")
    target = prepare_edit(off)
    args = [
        "--model_path", "/root/saig/CogVideoX-5b",
        "--init_prompt", "a " + SUB, "--edit_prompt", "a " + TGT,
        "--device", "cuda:0", "--dtype", "bf16",
        "--input_video_path", "input/%s.mp4" % SEQ,
        "--latent_trajectory_path", "inversion/%s.pt" % SEQ,
        "--mask_path", "mask/%s.pt" % SEQ,
        "--config_path", "configs/bear.yaml", "--fps", "4",
        "--num_inference_steps", STEPS,
        "--output_video_path", "output/_ab_%s.mp4" % mode,
    ]
    env = dict(os.environ)
    env["AB_TARGET"] = target
    env["AB_ARGS"] = "\x1f".join(args)
    env["SAIG_GID"] = os.path.join(F, "masks", "gid_%s.npy" % SEQ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    open(RUNNER, "w", encoding="utf-8").write(RUNNER_SRC)
    log("=== AB_RUN mode=%s target=%s steps=%s ===" % (mode, target, STEPS))
    t0 = time.time()
    p = subprocess.run([V, RUNNER], cwd=F, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    wall = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    with open(OG, "a", encoding="utf-8") as f:
        f.write(out[-4000:] + "\n")

    peak = None
    m = re.search(r"PEAK_GB\s+([\d.]+)", out)
    if m:
        peak = float(m.group(1))
    oom = "OutOfMemoryError" in out or "out of memory" in out.lower()
    per = wall / max(int(STEPS), 1)
    log("AB_RESULT mode=%s wall=%.1fs per_step=%.1fs peak_gb=%s oom=%s"
        % (mode, wall, per, ("%.2f" % peak) if peak else "NA", oom))
    return dict(mode=mode, wall=wall, per=per, peak=peak, oom=oom)


def main():
    os.chdir(F)
    for need in ("inversion/%s.pt" % SEQ, "mask/%s.pt" % SEQ):
        if not os.path.exists(need):
            print("AB_FAIL 缺 %s（先跑 run_all_night.sh 4 %s）" % (need, SEQ))
            return 2
    log("AB_START seq=%s steps=%s" % (SEQ, STEPS))
    r_on = run("on")
    r_off = run("off")

    print("\n================ 结果 ================")
    for r in (r_on, r_off):
        print("mode=%-4s wall=%7.1fs  per_step=%5.1fs  peak=%s GB  oom=%s"
              % (r["mode"], r["wall"], r["per"],
                 ("%.2f" % r["peak"]) if r["peak"] else "NA", r["oom"]))
    if r_off["oom"]:
        print("结论：这台卡**必须**走 offload（不开就 OOM）")
    elif r_off["peak"] and r_on["peak"]:
        ratio = r_off["per"] / max(r_on["per"], 1e-9)
        speed = r_on["per"] / max(r_off["per"], 1e-9)
        print("结论：关掉 offload 后 per_step 变化 = %.2f×（>1.25× 才算值得为大显存加钱）" % speed)
    print("日志：%s" % OG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
