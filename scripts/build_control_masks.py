# -*- coding: utf-8 -*-
"""构造 P0.5 三个控制臂的掩码：
  none_<seq>.pt   —— 全 1（无保留），用于 M5 无掩码臂
  gt_<seq>.pt     —— DAVIS 标注（subject=1），用于 M6 oracle 上界臂
  refine_<seq>.pt —— 仅在掩码边界 ±16 px 内用 g_id 抑制，用于 M3 边界细化臂
约定：mask=1 表示可编辑（主体），mask=0 表示保留（背景）。
"""
import os
import re

import cv2
import numpy as np
import torch

FADE = "/root/saig/FADE"
ANN = "/root/saig/DAVIS/Annotations/480p"
H, W = 480, 720
NFR = 49
ALPHA = 0.5
BAND = 16


def gmap(seq):
    p = os.path.join(FADE, "masks", "gid_%s.npy" % seq)
    g = np.load(p).astype(np.float32)          # (60,90)
    up = np.repeat(np.repeat(g, H // g.shape[0], axis=0), W // g.shape[1], axis=1)
    return np.clip(up, 0.0, 1.0)


def ann_frames(seq):
    d = os.path.join(ANN, seq)
    files = sorted(os.listdir(d))[:NFR]
    out = []
    for f in files:
        a = np.asarray(cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE))
        if a is None:
            a = np.asarray(cv2.imread(os.path.join(d, f), cv2.IMREAD_UNCHANGED))
        if a.ndim == 3:
            a = a[..., 0]
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
        out.append((a > 0).astype(np.uint8))
    return np.stack(out)                        # (49,480,720) uint8


def main():
    seqs = ["camel", "dog", "flamingo"]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND + 1, 2 * BAND + 1))
    for s in seqs:
        base = torch.load(os.path.join(FADE, "mask", "%s.pt" % s), map_location="cpu")
        base = base.bool().numpy() if torch.is_tensor(base) else np.asarray(base).astype(bool)
        assert base.shape == (NFR, H, W), base.shape

        # M5：全 1（全部可编辑）
        none = np.ones_like(base, dtype=bool)
        torch.save(torch.from_numpy(none), os.path.join(FADE, "mask", "none_%s.pt" % s))

        # M6：GT 标注（subject=1）
        gt = ann_frames(s).astype(bool)
        assert gt.shape == base.shape, (gt.shape, base.shape)
        torch.save(torch.from_numpy(gt), os.path.join(FADE, "mask", "gt_%s.pt" % s))

        # M3：边界带内用 g 抑制，带外保持原掩码
        g = gmap(s)
        ref = base.copy()
        for i in range(NFR):
            m = base[i].astype(np.uint8)
            dil = cv2.dilate(m, k, iterations=1)
            ero = cv2.erode(m, k, iterations=1)
            band = (dil == 1) & (ero == 0)
            # 迟滞式细化：只有注意力"高置信"的地方才改写，否则保留原掩码
            hi = band & (g >= 0.7)
            lo = band & (g <= 0.30)
            ref[i][hi] = True
            ref[i][lo] = False
        torch.save(torch.from_numpy(ref.astype(bool)), os.path.join(FADE, "mask", "refine_%s.pt" % s))

        print("BUILT %s  none_mean=%.3f gt_mean=%.3f base_mean=%.3f refine_mean=%.3f band=%.3f"
              % (s, none.mean(), gt.mean(), base.mean(), ref.mean(),
                 float((base ^ ref).mean())), flush=True)
    print("RESULT: CONTROL_MASKS_DONE")


if __name__ == "__main__":
    main()
