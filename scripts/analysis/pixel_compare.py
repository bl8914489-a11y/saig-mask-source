# -*- coding: utf-8 -*-
"""三臂像素对照 v2：支持任意段（B0 / B1-gate / B1-center），本机零 GPU。

用法：
  python pixel_compare_rev1.py --seqs camel dog flamingo
输出：
  <outdir>/pixel_metrics.csv        每个段 × 每个时刻 × 三个对照对
  <outdir>/compare_<seq>_t6.png        该段 t=6 s 的三臂 + 两张差值热图
"""
import argparse
import csv
import os
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

FF = imageio_ffmpeg.get_ffmpeg_exe()
B0DIR = r"F:\deepseek\saig_backup\pod_20261119\outputs"
NEWDIR = r"F:\deepseek\saig_backup\run_20261119_night\output"
OUTDIR = r"F:\deepseek\saig_backup\run_20261119_night\metrics_pixel"
WORK = r"F:\deepseek\work\pixframes2"
TS = [2, 4, 6, 8, 10]


def paths(seq):
    return {
        "B0": os.path.join(B0DIR, "b0_%s.mp4" % seq),
        "gate": os.path.join(NEWDIR, "b1_%s.mp4" % seq),
        "center": os.path.join(NEWDIR, "b1c_%s.mp4" % seq),
    }


def grab(video, t, dst):
    p = subprocess.run([FF, "-y", "-loglevel", "error", "-ss", str(t), "-i", video,
                        "-frames:v", "1", dst], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise RuntimeError("grab failed %s t=%s %s" % (video, t, (p.stderr or "")[-160:]))
    return np.asarray(Image.open(dst).convert("RGB")).astype(np.float32)


def box(img, r=5):
    H, W = img.shape
    ii = np.pad(img, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    y0 = np.clip(np.arange(H) - r, 0, H)
    y1 = np.clip(np.arange(H) + r + 1, 0, H)
    x0 = np.clip(np.arange(W) - r, 0, W)
    x1 = np.clip(np.arange(W) + r + 1, 0, W)
    S = ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)] - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)]
    n = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return S / n


def ssim(a, b):
    if a.ndim == 3:
        a = a.mean(axis=2)
    if b.ndim == 3:
        b = b.mean(axis=2)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = box(a), box(b)
    va, vb, vab = box(a * a) - ma ** 2, box(b * b) - mb ** 2, box(a * b) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * vab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


def metrics(a, b):
    d = np.abs(a - b)
    mse = float(((a - b) ** 2).mean())
    g = lambda x: x.mean(axis=2)
    return dict(MAE=float(d.mean()), PSNR=10 * np.log10(255 ** 2 / max(mse, 1e-12)),
                SSIM=ssim(a, b), dfrac15=float((g(d) > 15).mean()),
                dfrac30=float((g(d) > 30).mean()))


def colorize(d, amp):
    v = np.clip(d * amp, 0, 255) / 255.0
    cps = np.array([[0, 0, 0], [20, 20, 120], [0, 140, 200], [40, 200, 80],
                    [240, 220, 60], [240, 60, 40], [255, 255, 255]], dtype=np.float32)
    pos = np.linspace(0, 1, len(cps))
    out = np.zeros(v.shape + (3,), dtype=np.float32)
    for k in range(3):
        out[..., k] = np.interp(v, pos, cps[:, k])
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["camel", "dog", "flamingo"])
    ap.add_argument("--out", default=OUTDIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    pairs = [("gate-vs-center", "gate", "center"), ("B0-vs-gate", "B0", "gate"),
             ("B0-vs-center", "B0", "center")]
    rows = []
    for seq in a.seqs:
        ps = paths(seq)
        if not all(os.path.exists(v) for v in ps.values()):
            print("MISS", seq, {k: os.path.exists(v) for k, v in ps.items()})
            continue
        store = {}
        for t in TS:
            imgs = {k: grab(v, t, os.path.join(WORK, "%s_%s_%d.png" % (k, seq, t)))
                    for k, v in ps.items()}
            store[t] = imgs
            for name, ka, kb in pairs:
                m = metrics(imgs[ka], imgs[kb])
                rows.append(dict(seq=seq, pair=name, t=t, **m))
        imgs = store[6]
        h, w = imgs["B0"].shape[:2]
        panels = [imgs["B0"], imgs["gate"], imgs["center"],
                  colorize(np.abs(imgs["gate"] - imgs["center"]).mean(axis=2), 8.0),
                  colorize(np.abs(imgs["B0"] - imgs["gate"]).mean(axis=2), 0.5)]
        labels = ["B0 (no gate)", "B1 SAIG gate", "B1 center prior",
                  "|gate-center| x8", "|B0-gate| x0.5"]
        canvas = Image.new("RGB", (w * 5, h + 26), "white")
        dr = ImageDraw.Draw(canvas)
        for i, (p, lab) in enumerate(zip(panels, labels)):
            canvas.paste(Image.fromarray(p.astype(np.uint8)), (i * w, 26))
            dr.rectangle([i * w, 0, (i + 1) * w - 1, 25], fill=(20, 20, 20))
            dr.text((i * w + 6, 8), lab, fill=(255, 255, 255))
        canvas.save(os.path.join(a.out, "compare_%s_t6.png" % seq))
        print("SEQ_DONE", seq, flush=True)

    if rows:
        csvp = os.path.join(a.out, "pixel_metrics.csv")
        with open(csvp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["seq", "pair", "t", "MAE", "PSNR", "SSIM", "dfrac15", "dfrac30"])
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
        print("== 均值（按段 × 对照）==")
        seqs = list(dict.fromkeys(r["seq"] for r in rows))
        for seq in seqs:
            for name, _, _ in pairs:
                sel = [r for r in rows if r["seq"] == seq and r["pair"] == name]
                print("%-9s %-15s MAE=%6.2f SSIM=%.4f Δ>15=%.3f"
                      % (seq, name, np.mean([s["MAE"] for s in sel]),
                         np.mean([s["SSIM"] for s in sel]), np.mean([s["dfrac15"] for s in sel])))
        print("SAVED", csvp)
    print("RESULT: PIXEL_V2_DONE n=%d" % len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
