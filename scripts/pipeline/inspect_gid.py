"""AUROC 失败后的敏感性诊断（只解释，不改判据、不改阈值）。

判据已在 auroc_gid.py 中锁定：时间平均 g_id、正类=DAVIS GT 主体、阈值 0.75 / 裕度 +0.05。
本脚本只是回答「换哪个无关紧要的实现选择，结论会不会翻」：
  A  time_mean   13 帧平均（正式口径，等价 auroc_gid.py）
  B  frame0      只用第 1 个潜帧的 token 当分数（更接近"首帧身份"的字面含义）
  C  smooth3     时间平均后做 3×3 均值平滑（抑制 30×45 分块上采样的块状伪影）
  D  fade_mask   用 FADE 的 mask/<seq>.pt 走同一 max_pool+分组流程得到 (60,90) 当 GT
也输出每个变体相对 center_gauss 的裕度。
"""
import os

import numpy as np
import torch
from PIL import Image

TPF = 1350
GH, GW = 30, 45
MH, MW = 60, 90
SEQS = ["bear", "camel", "cows", "dog", "elephant", "flamingo", "goat", "rhino"]
GID = "/root/saig/FADE/saig_out"
ANN = "/root/saig/DAVIS/Annotations/480p"
MASK = "/root/saig/FADE/mask"


def auc(s, p):
    s = np.asarray(s, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=bool).ravel()
    n = s.size
    npos = int(p.sum())
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    ss = s[o]
    r = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[p].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def gt_davis(seq):
    a = np.asarray(Image.open(os.path.join(ANN, seq, "00000.png")))
    if a.ndim == 3:
        a = a[..., 0]
    mx = int(a.max()) if a.size else 0
    b = np.zeros(a.shape, dtype=np.uint8) if mx == 0 else \
        (a >= (1 if mx <= 1 else 128)).astype(np.uint8)
    return np.asarray(Image.fromarray(b).resize((MW, MH), Image.NEAREST)) >= 1


def gt_fade(seq):
    m = torch.load(os.path.join(MASK, "%s.pt" % seq)).float().detach().cpu()
    m = torch.nn.functional.max_pool2d(m.unsqueeze(1), kernel_size=(8, 8), stride=(8, 8))
    m = m.squeeze(1)
    N, H, W = m.shape
    k = (N - 1) // 4
    first = m[:-1].reshape(k, 4, H, W).max(dim=1)[0]
    last = m[-4:].max(dim=0)[0].unsqueeze(0)
    full = torch.cat([first, last], dim=0)          # (13,60,90)
    return (full.mean(dim=0).numpy() > 0.5)


def upsample_t(plane):
    return np.repeat(np.repeat(plane, MH // GH, axis=0), MW // GW, axis=1)


def smooth3(x):
    k = np.ones(3) / 3.0
    y = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, x)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, y)


def cg():
    y = np.linspace(-1, 1, MH)[:, None]
    x = np.linspace(-1, 1, MW)[None, :]
    return np.exp(-(x ** 2 + y ** 2) / (2 * 0.25 ** 2))


def main():
    c = cg()
    res = {k: [] for k in ("A", "B", "C", "D")}
    for seq in SEQS:
        g = np.load(os.path.join(GID, "%s_gid.npy" % seq))
        t_mean = upsample_t(g.mean(axis=0).reshape(GH, GW))
        frame0 = upsample_t(g[0].reshape(GH, GW))
        sm = smooth3(t_mean)
        lab = gt_davis(seq)
        lab_f = gt_fade(seq)
        res["A"].append(auc(t_mean, lab))
        res["B"].append(auc(frame0, lab))
        res["C"].append(auc(sm, lab))
        res["D"].append(auc(t_mean, lab_f))
        print("%-9s A=%.3f B=%.3f C=%.3f D=%.3f | cg(davis)=%.3f cg(fade)=%.3f obj=%.3f"
              % (seq, res["A"][-1], res["B"][-1], res["C"][-1], res["D"][-1],
                 auc(c, lab), auc(c, lab_f), lab.mean()), flush=True)

    print("MEAN_A %.4f" % np.mean(res["A"]))
    print("MEAN_B %.4f" % np.mean(res["B"]))
    print("MEAN_C %.4f" % np.mean(res["C"]))
    print("MEAN_D %.4f" % np.mean(res["D"]))
    print("MIN_BEST %.4f" % min(min(res[k]) for k in res))
    print("RESULT: DIAG_DONE")


if __name__ == "__main__":
    main()
