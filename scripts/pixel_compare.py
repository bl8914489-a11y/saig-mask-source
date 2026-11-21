# -*- coding: utf-8 -*-
"""三臂影片的像素级两两对照（本机、零 GPU、零网络）。

背景：pod 磁盘被清空后，720×480 的源视频已不可得（本机 FADE/input/bear.mp4 是 512×512 的
demo，规格不符）。因此本脚本只做**三条已产出影片之间**的两两比较——它们来自同一条输入、
同一组参数，差异只可能来自掩码来源，正是我们要的证据。

指标：
  MAE      逐通道平均绝对差（0–255）
  PSNR     = 10·log10(255²/MSE)
  SSIM     全局 SSIM（11×11 均值窗，C1/C2 标准常数）
  Δ>15/Δ>30  差值超过阈值的像素占比（编辑足迹）
时刻：t = 2, 4, 6, 8, 10 s（4 fps，时长 12.25 s）
"""
import csv
import os
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

FF = imageio_ffmpeg.get_ffmpeg_exe()
BASE = r"F:\deepseek\saig_backup\pod_20261119"
ARMS = {
    "B0": os.path.join(BASE, "outputs", "b0_bear.mp4"),
    "gate": os.path.join(BASE, "outputs", "b1p_gate_bear.mp4"),
    "center": os.path.join(BASE, "outputs", "b1p_center_bear.mp4"),
}
PAIRS = [("gate-vs-center", "gate", "center"),
         ("B0-vs-gate", "B0", "gate"),
         ("B0-vs-center", "B0", "center")]
TS = [2, 4, 6, 8, 10]
OUTDIR = os.path.join(BASE, "metrics_pixel")
WORK = r"F:\deepseek\work\pixframes"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


def grab(video, t, dst):
    p = subprocess.run([FF, "-y", "-loglevel", "error", "-ss", str(t), "-i", video,
                        "-frames:v", "1", dst], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise RuntimeError("grab failed %s t=%s rc=%s err=%s"
                           % (video, t, p.returncode, (p.stderr or "")[-200:]))
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
    va = box(a * a) - ma ** 2
    vb = box(b * b) - mb ** 2
    vab = box(a * b) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * vab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


def metrics(a, b):
    d = np.abs(a - b)
    mse = float(((a - b) ** 2).mean())
    g = lambda x: x.mean(axis=2)
    return dict(MAE=float(d.mean()), PSNR=10 * np.log10(255 ** 2 / max(mse, 1e-12)),
                SSIM=ssim(g(a), g(b)), dfrac15=float((g(d) > 15).mean()),
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
    rows, store = [], {}
    for t in TS:
        imgs = {k: grab(p, t, os.path.join(WORK, "%s_%d.png" % (k, t))) for k, p in ARMS.items()}
        store[t] = imgs
        for name, ka, kb in PAIRS:
            m = metrics(imgs[ka], imgs[kb])
            rows.append(dict(pair=name, t=t, **m))
            print("t=%2ds %-15s MAE=%6.2f PSNR=%5.2f SSIM=%.4f  Δ>15=%.3f Δ>30=%.3f"
                  % (t, name, m["MAE"], m["PSNR"], m["SSIM"], m["dfrac15"], m["dfrac30"]),
                  flush=True)

    print("\n== 平均（%d 个时刻） ==" % len(TS))
    agg = {}
    for name, _, _ in PAIRS:
        sel = [r for r in rows if r["pair"] == name]
        agg[name] = {k: float(np.mean([s[k] for s in sel]))
                     for k in ("MAE", "PSNR", "SSIM", "dfrac15", "dfrac30")}
        a = agg[name]
        print("%-15s MAE=%6.2f PSNR=%5.2f SSIM=%.4f  Δ>15=%.3f Δ>30=%.3f"
              % (name, a["MAE"], a["PSNR"], a["SSIM"], a["dfrac15"], a["dfrac30"]))

    csvp = os.path.join(OUTDIR, "pixel_metrics.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "t", "MAE", "PSNR", "SSIM", "dfrac15", "dfrac30"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    imgs = store[6]
    h, w_ = imgs["B0"].shape[:2]
    panels = [imgs["B0"], imgs["gate"], imgs["center"],
              colorize(np.abs(imgs["gate"] - imgs["center"]).mean(axis=2), 8.0),
              colorize(np.abs(imgs["B0"] - imgs["gate"]).mean(axis=2), 0.5)]
    labels = ["B0 (no gate)", "B1 SAIG gate", "B1 center prior",
              "|gate - center| x8", "|B0 - gate| x0.5"]
    canvas = Image.new("RGB", (w_ * 5, h + 26), "white")
    dr = ImageDraw.Draw(canvas)
    for i, (p, lab) in enumerate(zip(panels, labels)):
        canvas.paste(Image.fromarray(p.astype(np.uint8)), (i * w_, 26))
        dr.rectangle([i * w_, 0, (i + 1) * w_ - 1, 25], fill=(20, 20, 20))
        dr.text((i * w_ + 6, 8), lab, fill=(255, 255, 255))
    png = os.path.join(OUTDIR, "pixel_compare_t6.png")
    canvas.save(png)

    print("\nSAVED", png, canvas.size)
    print("SAVED", csvp)
    print("RESULT: PIXEL_METRICS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
