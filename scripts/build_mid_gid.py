# -*- coding: utf-8 -*-
"""从已捕获的 q/k 生成「中间帧」门控图 g_id_mid（层 2，参考帧 6，head 平均）。

输出：/root/saig/FADE/masks/gid_mid_<seq>.npy  shape (60,90), float32, [0,1]
"""
import glob
import os
import re

import numpy as np
import torch

TPF = 1350
GH, GW = 30, 45
MH, MW = 60, 90
FRAMES = 13
OUT = "/root/saig/FADE/masks"


def as_shd(t):
    if t.dim() == 4:
        t = t[0]
    h = None
    for ax in range(3):
        if t.shape[ax] == 48:
            h = ax
            break
    if h is None:
        h = 1
    others = [ax for ax in range(3) if ax != h]
    seq_ax = max(others, key=lambda ax: t.shape[ax])
    d_ax = [ax for ax in others if ax != seq_ax][0]
    if (seq_ax, h, d_ax) != (0, 1, 2):
        t = t.permute(seq_ax, h, d_ax)
    return t.contiguous()


def mass_mean_heads(q, k, ref):
    S, H, d = q.shape
    scale = 1.0 / (d ** 0.5)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    acc = torch.zeros(S, device=dev)
    lo, hi = ref * TPF, (ref + 1) * TPF
    for h in range(H):
        qh = q[:, h, :].to(dev).float()
        kh = k[:, h, :].to(dev).float()
        num = torch.zeros(S, device=dev)
        den = torch.zeros(S, device=dev)
        for i in range(0, S, 512):
            j = min(i + 512, S)
            s = (qh[i:j] @ kh.t()) * scale
            m = s.max(dim=1, keepdim=True).values
            e = torch.exp(s - m)
            num[i:j] = e[:, lo:hi].sum(dim=1)
            den[i:j] = e.sum(dim=1)
        acc += num / den.clamp_min(1e-20)
        del qh, kh
        torch.cuda.empty_cache()
    return (acc / H).cpu()


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob("/root/qk/*_L02.pt"))
    for f in files:
        seq = re.search(r"/([a-z]+)_L\d+\.pt$", f).group(1)
        blob = torch.load(f, map_location="cpu")
        q, k = as_shd(blob["q"]), as_shd(blob["k"])
        S0 = q.shape[0]
        pfx = S0 - FRAMES * TPF
        q, k = q[pfx:].contiguous(), k[pfx:].contiguous()
        m = mass_mean_heads(q, k, ref=6)
        v = m[: FRAMES * TPF].reshape(FRAMES, TPF).mean(0).numpy().astype(np.float32)
        g = v.reshape(GH, GW)
        g = (g - g.min()) / (g.max() - g.min() + 1e-9)
        up = np.repeat(np.repeat(g, MH // GH, axis=0), MW // GW, axis=1).astype(np.float32)
        p = os.path.join(OUT, "gid_mid_%s.npy" % seq)
        np.save(p, up)
        print("SAVED %s shape=%s min=%.3f mean=%.3f max=%.3f"
              % (p, up.shape, up.min(), up.mean(), up.max()), flush=True)
        # 同时存一份与旧口径同名的对照（层 2 首帧）
        m0 = mass_mean_heads(q, k, ref=0)
        v0 = m0[: FRAMES * TPF].reshape(FRAMES, TPF).mean(0).numpy().astype(np.float32)
        g0 = v0.reshape(GH, GW)
        g0 = (g0 - g0.min()) / (g0.max() - g0.min() + 1e-9)
        up0 = np.repeat(np.repeat(g0, MH // GH, axis=0), MW // GW, axis=1).astype(np.float32)
        np.save(os.path.join(OUT, "gid_first_%s.npy" % seq), up0)
        print("SAVED gid_first_%s.npy (对照)" % seq, flush=True)
    print("RESULT: MID_GID_DONE n=%d" % len(files))


if __name__ == "__main__":
    main()
