"""第 5 步：ID-S / Sem-T 三方对照（B0 vs SAIG 门控 vs 中心先验门控）。

口径（写进论文 4.4，避免事后解释）：
  ID-S  身份保持度 = 编辑后帧与原始帧在「主体区域」上的 DINOv2 CLS 余弦相似度。
        主体区域用 DAVIS 标注该帧的 bbox（外扩 10%）裁剪；逐帧算再取均值。
        取值越高 = 主体身份越接近原始。
  Sem-T 语义贴合度 = 编辑后帧整帧与目标文本的 CLIP 图文余弦；同时报与源文本的分数。
  - 采样：等间隔取 K=13 帧（与潜帧数一致）。
  - 模型：timm vit_base_patch14_dinov2.lvd142m（DINOv2）；openai/clip-vit-base-patch32。
  - 显式跑在 CPU 上，不占 GPU（B 轨实验期间不得抢卡）。
"""
import argparse
import csv
import os
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image

K = 13


def frames_of(video, tmpdir, tag):
    out = os.path.join(tmpdir, tag)
    os.makedirs(out, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                    os.path.join(out, "%04d.png")], check=True)
    fs = sorted(os.listdir(out))
    return [os.path.join(out, f) for f in fs]


def load_gt_boxes(ann_dir, n_frames):
    """返回每帧主体 bbox（DAVIS 0/1 palette 标注，取非零像素外接框）。"""
    boxes = []
    files = sorted(os.listdir(ann_dir))[:n_frames]
    for f in files:
        a = np.asarray(Image.open(os.path.join(ann_dir, f)))
        if a.ndim == 3:
            a = a[..., 0]
        ys, xs = np.nonzero(a)
        if ys.size == 0:
            boxes.append(None)
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        pad_y = int(0.1 * (y1 - y0 + 1))
        pad_x = int(0.1 * (x1 - x0 + 1))
        boxes.append((max(0, x0 - pad_x), max(0, y0 - pad_y),
                      min(a.shape[1] - 1, x1 + pad_x), min(a.shape[0] - 1, y1 + pad_y)))
    return boxes


def crop(img, box):
    x0, y0, x1, y1 = box
    return img.crop((x0, y0, x1 + 1, y1 + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="bear")
    ap.add_argument("--src", default="/root/saig/FADE/input/bear.mp4")
    ap.add_argument("--ann", default="/root/saig/DAVIS/Annotations/480p/bear")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="name=path 形式，例如 b0=/root/saig/FADE/output/b0_bear.mp4")
    ap.add_argument("--target", default="a panda")
    ap.add_argument("--init", default="a bear")
    ap.add_argument("--out", default="/root/logs/metrics_pilot.csv")
    args = ap.parse_args()

    dev = "cpu"
    import timm
    from transformers import CLIPModel, CLIPProcessor

    dino = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True).eval().to(dev)
    dcfg = timm.data.resolve_data_config({}, model=dino)
    dtf = timm.data.create_transform(**dcfg)
    clipm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(dev)
    clipp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    tmp = tempfile.mkdtemp(prefix="mcmp_")
    src_files = frames_of(args.src, tmp, "src")
    n = len(src_files)
    boxes = load_gt_boxes(args.ann, n)
    idx = np.linspace(0, n - 1, min(K, n)).astype(int)

    def dino_feat(im):
        with torch.no_grad():
            t = dtf(im.convert("RGB")).unsqueeze(0).to(dev)
            return dino(t).float().cpu().numpy().ravel()

    with torch.no_grad():
        txt_t = clipp(text=[args.target, args.init], return_tensors="pt", padding=True).to(dev)
        tf = clipm.get_text_features(**txt_t).float().cpu().numpy()

    def clip_img(im):
        with torch.no_grad():
            inp = clipp(images=im.convert("RGB"), return_tensors="pt").to(dev)
            return clipm.get_image_features(**inp).float().cpu().numpy().ravel()

    rows = []
    for spec in args.videos:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print("MISS", name, path)
            continue
        vfiles = frames_of(path, tmp, name)
        m = min(len(vfiles), n)
        ids, semt, seminit = [], [], []
        for i in idx:
            if i >= m:
                continue
            src = Image.open(src_files[i])
            ed = Image.open(vfiles[i])
            box = boxes[i] if i < len(boxes) else None
            if box is None:
                continue
            a = dino_feat(crop(src, box))
            b = dino_feat(crop(ed, box))
            ids.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
            ei = clip_img(ed)
            ei = ei / (np.linalg.norm(ei) + 1e-12)
            semt.append(float(ei @ (tf[0] / (np.linalg.norm(tf[0]) + 1e-12))))
            seminit.append(float(ei @ (tf[1] / (np.linalg.norm(tf[1]) + 1e-12))))
        rows.append(dict(video=name, n=len(ids), ID_S=float(np.mean(ids)) if ids else float("nan"),
                         Sem_T_target=float(np.mean(semt)) if semt else float("nan"),
                         Sem_T_init=float(np.mean(seminit)) if seminit else float("nan")))
        r = rows[-1]
        print("%-14s ID_S=%.4f  SemT(target)=%.4f  SemT(init)=%.4f  n=%d"
              % (name, r["ID_S"], r["Sem_T_target"], r["Sem_T_init"], r["n"]), flush=True)

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print("RESULT: METRICS_DONE n=%d" % len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
