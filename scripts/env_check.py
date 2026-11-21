"""Pre-flight check for the SAIG run.  Nothing is modified; it only reports.

Usage:  python scripts/env_check.py
Exit code 0 = every hard requirement met, 1 = something must be fixed.
"""

import os
import platform
import shutil
import subprocess
import sys

FRAME_TOKENS = 30 * 45
LATENT_FRAMES = 13
SEQ_LEN = FRAME_TOKENS * LATENT_FRAMES

fails = []


def line(k, v, ok=None):
    mark = "" if ok is None else ("  OK " if ok else "  !! ")
    print(f"{mark}{k:<26} {v}")
    if ok is False:
        fails.append(k)


print("== SAIG pre-flight ==")
line("platform", platform.platform())
line("python", sys.version.split()[0], sys.version_info >= (3, 10))
line("cwd", os.getcwd())

# --- constants sanity
line("frame tokens", FRAME_TOKENS, FRAME_TOKENS == 1350)
line("latent frames", LATENT_FRAMES, LATENT_FRAMES == 13)
line("seq len", SEQ_LEN, SEQ_LEN == 17550)
print(f"       head-matrix fp32 per head: {SEQ_LEN**2 * 4 / 1e9:.2f} GB "
      f"(x48 heads = {SEQ_LEN**2 * 4 * 48 / 1e9:.0f} GB -> never allocate all)")

# --- python deps
for mod, need in (("numpy", True), ("torch", True), ("diffusers", True),
                  ("transformers", True), ("PIL", False), ("scipy", False)):
    try:
        m = __import__(mod)
        line(mod, getattr(m, "__version__", "installed"), True)
    except Exception as e:
        line(mod, f"MISSING ({type(e).__name__})", not need)

# --- versions declared in the FADE requirements
try:
    import torch
    tv = torch.__version__
    line("torch==2.5.1?", tv, tv.startswith("2.5"))
    line("cuda available", torch.cuda.is_available(), torch.cuda.is_available())
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total = props.total_memory / 1024 ** 3
        line("gpu", f"{props.name} {total:.1f} GB", total >= 23.0)
        line("gpu >=40GB (native)", total >= 40, None)
        line("gpu >=23.6GB (offload ok)", total >= 23.6, total >= 23.6)
except Exception:
    pass

# --- disk
try:
    total, used, free = shutil.disk_usage(os.getcwd())
    line("disk free", f"{free / 1024**3:.1f} GB", free > 60 * 1024 ** 3)
except Exception:
    pass

# --- external commands
for cmd in ("nvidia-smi", "git", "huggingface-cli"):
    p = shutil.which(cmd)
    line(cmd, p or "NOT FOUND", p is not None)

# --- expected assets (edit these paths)
EXPECT = {
    "CogVideoX-5B weights": os.environ.get("COGVIDEO_PATH", ""),
    "FADE repo": os.environ.get("FADE_DIR", ""),
    "DAVIS root": os.environ.get("DAVIS_ROOT", ""),
}
for k, v in EXPECT.items():
    if not v:
        line(k, "not set (export an env var)", None)
    else:
        line(k, v, os.path.exists(v))

print("\nRESULT:", "READY" if not fails else f"BLOCKED on {len(fails)} item(s): {fails}")
sys.exit(0 if not fails else 1)
