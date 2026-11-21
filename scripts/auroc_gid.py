"""第 3 步闸门：g_id 的 AUROC + 三个零模型（CPU，无需 GPU）。

判据（第 3.3 表原文）：AUROC≥0.75 且领先零模型 ≥+0.05。

口径（必须写进论文 4.1，避免事后解释；方向不可事后择向，这里固定为先验约定）：
  - 正类 = DAVIS 首帧主体（GT=1）。依据：档案第 8 章第 3 步原文是
    "把 DAVIS 首帧 GT 降到 60×90 → 与 g_id 算 AUROC"，GT 为正类。
    同时输出反向口径（正类=背景/保区）仅作方向复核，不用于判据。
  - g_id 形状 (T=13, 1350)，1350 = 30×45（CogVideoX 潜空间 60×90 经 2×2 patch）。
  - 取 13 帧平均 -> (30,45) -> ×2 上采样到 (60,90)，与 FADE 掩码同尺寸。
  - GT：DAVIS Annotations/480p/<seq>/00000.png 缩到 (60,90) 后 >127 为真。
  - 零模型（档案第 8 章第 3 步列出的三个）：
      center_gauss  居中二维高斯（σ=0.25·min(H,W)），与 g_id 同尺寸
      full_frame    全 1 常数（等价于"对整帧均匀关注"）
      random_layer  把 g_id 空间排布随机置换（固定种子），保留直方图、破坏结构
  - AUC 用 Mann–Whitney 秩和公式，含并列秩平均（numpy 自实现，pod 无 sklearn）。
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image

TPF = 1350
GH, GW = 30, 45      # 潜空间 token 网格
MH, MW = 60, 90      # 掩码尺寸


def auc_score(score, pos):
    """score: (N,) 连续值；pos: (N,) bool 正类。返回 AUC（并列取平均秩）。"""
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
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def gid_map(path):
    g = np.load(path)
    if g.ndim != 2 or g.shape[1] != TPF:
        raise ValueError("unexpected g_id shape %s (want (T,%d))" % (g.shape, TPF))
    m = g.mean(axis=0).reshape(GH, GW)
    up = np.repeat(np.repeat(m, MH // GH, axis=0), MW // GW, axis=1)
    return up  # (60,90)


def gt_mask(png):
    """DAVIS 标注实测是 palette(P) 模式、像素值是 0/1 索引。

    坑：convert('L') 会走 palette 查表，把索引 1 映射成灰度 38（实测 L_MAX=38），
    任何 `>127` 阈值都会判成全背景。所以这里不转换色彩模式，直接读原始索引，
    先把二值化做完，再上采样到 (60,90)。0/1 与 0/255 两种编码都吃。
    """
    im = Image.open(png)
    a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., 0]
    mx = int(a.max()) if a.size else 0
    if mx == 0:
        b = np.zeros(a.shape, dtype=np.uint8)
    else:
        b = (a >= (1 if mx <= 1 else 128)).astype(np.uint8)
    small = np.asarray(Image.fromarray(b).resize((MW, MH), Image.NEAREST))
    return small >= 1


def center_gauss():
    y = np.linspace(-1, 1, MH)[:, None]
    x = np.linspace(-1, 1, MW)[None, :]
    r2 = x ** 2 + y ** 2
    return np.exp(-r2 / (2 * 0.25 ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid_dir", default="/root/saig/FADE/saig_out")
    ap.add_argument("--ann", default="/root/saig/DAVIS/Annotations/480p")
    ap.add_argument("--out", default="/root/logs/auroc_gid.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seqs = ["bear", "camel", "cows", "dog", "elephant", "flamingo", "goat", "rhino"]
    rng = np.random.default_rng(args.seed)
    cg = center_gauss()
    ff = np.ones((MH, MW))
    rows = []

    for seq in seqs:
        gp = os.path.join(args.gid_dir, "%s_gid.npy" % seq)
        mp = os.path.join(args.ann, seq, "00000.png")
        if not (os.path.exists(gp) and os.path.exists(mp)):
            print("MISS %s gid=%s gt=%s" % (seq, os.path.exists(gp), os.path.exists(mp)))
            continue
        s = gid_map(gp)
        obj = gt_mask(mp)                 # 正类 = 主体
        bg = ~obj
        rl = s.ravel()[rng.permutation(s.size)].reshape(MH, MW)

        a_gid = auc_score(s, obj)
        a_bg = auc_score(s, bg)           # 反向口径，仅复核方向
        a_cg = auc_score(cg, obj)
        a_ff = auc_score(ff, obj)
        a_rl = auc_score(rl, obj)
        zero_max = max(a_cg, a_ff, a_rl)
        rows.append(dict(seq=seq, auc_gid=a_gid, auc_bg_rev=a_bg,
                         center_gauss=a_cg, full_frame=a_ff, random_layer=a_rl,
                         zero_max=zero_max, margin=a_gid - zero_max,
                         obj_ratio=float(obj.mean())))
        print("%-9s AUC_gid=%.4f  bg_rev=%.4f  cg=%.4f ff=%.4f rl=%.4f  margin=%+.4f  obj=%.3f"
              % (seq, a_gid, a_bg, a_cg, a_ff, a_rl, a_gid - zero_max, obj.mean()), flush=True)

    if not rows:
        print("NO_DATA")
        return 2

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    a = np.array([r["auc_gid"] for r in rows])
    m = np.array([r["margin"] for r in rows])
    print("MEAN_AUC_GID %.4f" % a.mean())
    print("MEAN_MARGIN %+.4f" % m.mean())
    print("MIN_AUC_GID %.4f" % a.min())
    print("MIN_MARGIN %+.4f" % m.min())
    gate = (a.mean() >= 0.75) and (m.mean() >= 0.05)
    print("GATE_AUROC", "PASS" if gate else "FAIL")
    print("RESULT: AUROC_DONE rows=%d" % len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
