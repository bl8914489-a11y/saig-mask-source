"""本机版 ID-S / Sem-T 三方对照（不依赖 pod；源视频用 F:\\deepseek\\FADE\\input\\bear.mp4）。

与 pod 版的差别（必须写进论文的 limitation）：
  - pod 上 DAVIS 标注 + SAM 掩码已随 pod 停机，无法取回；本机用「时域方差」自动定位主体框：
    对源视频逐像素算时间标准差，取 90 分位的连通高方差区域的外接框（外扩 15%）。
  - 因此本机的 ID-S 框是「运动主体框」代理，不是 DAVIS GT 框；两套口径不可混用。
  - ID-S = DINOv2 CLS 余弦（源帧 vs 编辑帧，同一框内），逐帧均值。
  - Sem-T = CLIP 图文余弦（编辑帧整帧 vs 目标/源文本），逐帧均值。
"""
import argparse
import csv
import os

import numpy as np
import torch

K = 13


def read_frames(path, k=K):
    import imageio.v3 as iio
    frames = list(iio.imiter(path))
    n = len(frames)
    idx = np.linspace(0, n - 1, min(k, n)).astype(int)
    return [frames[i] for i in idx], n, idx


def motion_box(frames):
    g = np.stack([f.mean(axis=2) for f in frames]).astype(np.float32)
    sd = g.std(axis=0)
    thr = np.percentile(sd, 90.0)
    ys, xs = np.nonzero(sd >= thr)
    if ys.size == 0:
        h, w = sd.shape
        return (0, 0, w - 1, h - 1)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = sd.shape
    py = int(0.15 * (y1 - y0 + 1))
    px = int(0.15 * (x1 - x0 + 1))
    return (max(0, x0 - px), max(0, y0 - py), min(w - 1, x1 + px), min(h - 1, y1 + py))


def crop(f, box):
    x0, y0, x1, y1 = box
    return f[y0:y1 + 1, x0:x1 + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"F:\deepseek\FADE\input\bear.mp4")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--target", default="a panda")
    ap.add_argument("--init", default="a bear")
    ap.add_argument("--out", default=r"F:\deepseek\saig_backup\pod_20261119\logs\metrics_local.csv")
    args = ap.parse_args()

    dev = "cpu"
    import timm
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    dino = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True).eval()
    dtf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=dino))
    clipm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    clipp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    src_frames, n_src, idx = read_frames(args.src)
    box = motion_box(src_frames)
    print("SRC", args.src, "n=", n_src, "sampled=", len(src_frames), "motion_box=", box, flush=True)

    def pil(a):
        return Image.fromarray(a.astype(np.uint8))

    def dino_feat(a):
        with torch.no_grad():
            t = dtf(pil(a)).unsqueeze(0)
            return dino(t).float().numpy().ravel()

    with torch.no_grad():
        tt = clipp(text=[args.target, args.init], return_tensors="pt", padding=True)
        tf = clipm.get_text_features(**tt).float().numpy()

    def clip_feat(a):
        with torch.no_grad():
            inp = clipp(images=pil(a), return_tensors="pt")
            return clipm.get_image_features(**inp).float().numpy().ravel()

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    rows = []
    for spec in args.videos:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print("MISS", name, path)
            continue
        fr, n, _ = read_frames(path)
        ids_box, ids_full, st, si = [], [], [], []
        for i, a in enumerate(fr):
            if i >= len(src_frames):
                break
            s = src_frames[i]
            ids_box.append(cos(dino_feat(crop(s, box)), dino_feat(crop(a, box))))
            ids_full.append(cos(dino_feat(s), dino_feat(a)))
            e = clip_feat(a)
            en = e / (np.linalg.norm(e) + 1e-12)
            st.append(float(en @ (tf[0] / (np.linalg.norm(tf[0]) + 1e-12))))
            si.append(float(en @ (tf[1] / (np.linalg.norm(tf[1]) + 1e-12))))
        r = dict(video=name, n=len(ids_box), ID_S_box=float(np.mean(ids_box)),
                 ID_S_full=float(np.mean(ids_full)), Sem_T_target=float(np.mean(st)),
                 Sem_T_init=float(np.mean(si)))
        rows.append(r)
        print("%-16s ID_S(box)=%.4f ID_S(full)=%.4f SemT(target)=%.4f SemT(init)=%.4f n=%d"
              % (name, r["ID_S_box"], r["ID_S_full"], r["Sem_T_target"], r["Sem_T_init"], r["n"]),
              flush=True)

    if rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print("RESULT: LOCAL_METRICS_DONE n=%d" % len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
