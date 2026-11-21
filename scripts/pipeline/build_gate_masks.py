"""第 4 步前段：把 g_id (13,1350) 变成身份闸门图 masks/gid_<seq>.npy，形状 (60,90)。

口径（写进论文 4.1）：
  - (13,1350) 取时间平均 -> (30,45)（1350 = 30×45，2×2 patch）-> ×2 上采样 -> (60,90)
  - 最小-最大归一化到 [0,1]，得到 g↑
  - 真正的合成在 FADE guider 内完成：mask_saig = mask_edit ⊙ (1 − 0.5·g↑)
    （见 patch_guider_saig_gate.py；在 60×90 的 mask 上施加，避免 8×8 max-pool 把软权重抹平）
"""
import argparse
import os

import numpy as np

TPF = 1350
GH, GW = 30, 45
MH, MW = 60, 90


def gid_to_map(path):
    g = np.load(path)
    if g.ndim != 2 or g.shape[1] != TPF:
        raise ValueError("unexpected g_id shape %s" % (g.shape,))
    m = g.mean(axis=0).reshape(GH, GW)
    m = np.repeat(np.repeat(m, MH // GH, axis=0), MW // GW, axis=1)
    lo, hi = float(m.min()), float(m.max())
    m = (m - lo) / max(hi - lo, 1e-8)
    return m.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid_dir", default="/root/saig/FADE/saig_out")
    ap.add_argument("--mask_dir", default="/root/saig/FADE/mask")
    ap.add_argument("--out_dir", default="/root/saig/FADE/masks")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seqs = ["bear", "camel", "cows", "dog", "elephant", "flamingo", "goat", "rhino"]
    n = 0
    for seq in seqs:
        gp = os.path.join(args.gid_dir, "%s_gid.npy" % seq)
        mp = os.path.join(args.mask_dir, "%s.pt" % seq)
        if not os.path.exists(gp):
            print("MISS_GID", seq)
            continue
        if not os.path.exists(mp):
            print("MISS_MASK", seq)
            continue
        g = gid_to_map(gp)
        out = os.path.join(args.out_dir, "gid_%s.npy" % seq)
        np.save(out, g)
        n += 1
        print("GIDMAP %-9s shape=%s min=%.4f mean=%.4f max=%.4f -> %s"
              % (seq, g.shape, g.min(), g.mean(), g.max(), out), flush=True)

    print("RESULT: GIDMAP_DONE n=%d" % n)
    return 0 if n == 8 else 2


if __name__ == "__main__":
    raise SystemExit(main())
