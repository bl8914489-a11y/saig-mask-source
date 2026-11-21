# -*- coding: utf-8 -*-
"""聚合方式扫描：参考帧 × head 聚合 的 AUROC（从已存的 q/k 离线算）。

用法：
  /root/venv310/bin/python /root/sweep_agg.py
依赖：/root/qk/<seq>_L<NN>.pt、DAVIS 标注
输出：/root/logs/sweep_agg.csv + 控制台表格
"""
import csv
import glob
import os
import re

import numpy as np
import torch
from PIL import Image

TPF = 1350
GH, GW = 30, 45
MH, MW = 60, 90
FRAMES = 13
ANN = "/root/saig/DAVIS/Annotations/480p"
OUT = "/root/logs/sweep_agg.csv"


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


def auc_score(score, pos):
    score = np.asarray(score, dtype=np.float64).ravel()
    pos = np.asarray(pos, dtype=bool).ravel()
    n = score.size
    n_pos = int(pos.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s_sorted = score[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def gt_mask(seq):
    p = os.path.join(ANN, seq, "00000.png")
    im = Image.open(p)
    a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., 0]
    m = (a > 0).astype(np.float32)
    im2 = Image.fromarray((m * 255).astype(np.uint8)).resize((MW, MH), Image.NEAREST)
    return (np.asarray(im2) > 127)


def head_agg(mh, how):
    """mh: (H,S) 每头质量 -> (S,)"""
    if how == "mean":
        return mh.mean(0)
    if how == "max":
        return mh.max(0).values
    if how == "top5":
        k = min(5, mh.shape[0])
        return mh.topk(k, dim=0).values.mean(0)
    raise ValueError(how)


def window_mass_multi(q, k, refs):
    """q,k: (S,H,d)。一次遍历同时算多个参考帧窗口，返回 {ref: (H,S)}。GPU 流式。"""
    S, H, d = q.shape
    scale = 1.0 / (d ** 0.5)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {r: torch.zeros(H, S) for r in refs}
    for h in range(H):
        qh = q[:, h, :].to(dev).float()
        kh = k[:, h, :].to(dev).float()
        den = torch.zeros(S, device=dev)
        nums = {r: torch.zeros(S, device=dev) for r in refs}
        for i in range(0, S, 512):
            j = min(i + 512, S)
            s = (qh[i:j] @ kh.t()) * scale
            m = s.max(dim=1, keepdim=True).values
            e = torch.exp(s - m)
            den[i:j] = e.sum(dim=1)
            for r in refs:
                lo, hi = r * TPF, (r + 1) * TPF
                nums[r][i:j] = e[:, lo:hi].sum(dim=1)
            del s, e
        for r in refs:
            out[r][h] = (nums[r] / den.clamp_min(1e-20)).cpu()
        del qh, kh
        torch.cuda.empty_cache()
    return out


def to_g(m):
    """(F*TPF,) 或 (S,) -> (60,90)"""
    v = m[: FRAMES * TPF].reshape(FRAMES, TPF).mean(0)
    g = v.reshape(GH, GW).numpy()
    lo, hi = np.nanmin(g), np.nanmax(g)
    g = (g - lo) / (hi - lo + 1e-9)
    up = np.repeat(np.repeat(g, MH // GH, axis=0), MW // GW, axis=1)
    return up


def main():
    files = sorted(glob.glob("/root/qk/*_L*.pt"))
    if not files:
        print("RESULT: NO_QK")
        return
    rows = []
    for f in files:
        m = re.search(r"/([a-z]+)_L(\d+)\.pt$", f)
        seq, layer = m.group(1), int(m.group(2))
        try:
            gt = gt_mask(seq)
        except Exception as e:
            print("SKIP", seq, "no ann", type(e).__name__)
            continue
        blob = torch.load(f, map_location="cpu")
        q, k = as_shd(blob["q"]), as_shd(blob["k"])
        S0 = q.shape[0]
        pfx = S0 - FRAMES * TPF
        if pfx < 0:
            print("SKIP", seq, layer, "S", S0)
            continue
        q, k = q[pfx:].contiguous(), k[pfx:].contiguous()
        refs = (0, 6, 12)
        mass = window_mass_multi(q, k, refs)
        for ref in refs:
            mh = mass[ref]
            for how in ("mean", "max", "top5"):
                mv = head_agg(mh, how)
                auc = auc_score(to_g(mv).ravel(), gt.ravel())
                rows.append(dict(seq=seq, layer=layer, ref_frame=ref, head_agg=how, auroc=round(auc, 4)))
                print("%-9s L%02d ref=%-2d head=%-4s AUROC=%.4f" % (seq, layer, ref, how, auc), flush=True)
        del q, k, blob

    if not rows:
        print("RESULT: NO_ROWS")
        return
    os.makedirs("/root/logs", exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["seq", "layer", "ref_frame", "head_agg", "auroc"])
        w.writeheader()
        w.writerows(rows)
    import collections
    best = max(rows, key=lambda r: r["auroc"])
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["layer"], r["ref_frame"], r["head_agg"])].append(r["auroc"])
    print()
    print("=== 按配置平均（跨段）===")
    for key in sorted(agg, key=lambda k: -np.mean(agg[k]))[:8]:
        print("layer=%d ref=%d head=%-4s mean_auroc=%.4f" % (key[0], key[1], key[2], float(np.mean(agg[key]))))
    print()
    print("BEST_ROW", best)
    print("RESULT: SWEEP_AGG_DONE rows=%d" % len(rows))


if __name__ == "__main__":
    main()
