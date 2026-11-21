# -*- coding: utf-8 -*-
"""生成 TVC 版四张图（PIL，无色内标题，300 dpi 等效）。

fig1_pipeline.png      管线与闸门（无内嵌标题）
fig2_ids.png           8 段三臂 ID-S 均值 + 95% CI 误差棒
fig3_qualitative.png   4 段 × 4 臂定性对照
fig4_sweep.png         108 格扫描热力图（4 面板 = 头聚合 × 4×3）
"""
import csv
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

FIG = r"F:\deepseek\saig_project\figures"
BUNDLE = r"F:\deepseek\saig_backup\saig_results_20261121\output"
SWEEP = r"F:\deepseek\work\sweep_agg.csv"
NAVY = (31, 58, 95)
ACC = (198, 84, 40)
GREY = (110, 110, 110)
BLUE = (70, 110, 165)
LIGHT = (232, 238, 247)


def font(sz, bold=False):
    names = ["msyhbd.ttc", "msyh.ttc"] if bold else ["msyh.ttc", "simhei.ttf"]
    for n in names:
        for root in (r"C:\Windows\Fonts", r"C:\windows\fonts"):
            try:
                return ImageFont.truetype(os.path.join(root, n), sz)
            except Exception:
                continue
    return ImageFont.load_default()


def fig1():
    W, H = 1800, 760
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_b, f_s = font(19, True), font(16)

    def box(x, y, w, h, title, sub, edge=NAVY, fill=LIGHT, tcol=NAVY):
        d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=fill, outline=edge, width=2)
        d.text((x + 12, y + 9), title, font=f_b, fill=tcol)
        yy = y + 36
        for line in sub:
            d.text((x + 12, yy), line, font=f_s, fill=(30, 30, 30))
            yy += 22

    def arrow(pts, col=NAVY, w=3):
        for i in range(len(pts) - 1):
            d.line([pts[i], pts[i + 1]], fill=col, width=w)
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        L = 12
        d.polygon([(x2, y2), (x2 - L * math.cos(a - 0.4), y2 - L * math.sin(a - 0.4)),
                   (x2 - L * math.cos(a + 0.4), y2 - L * math.sin(a + 0.4))], fill=col)

    y1 = 40
    box(30, y1, 230, 110, "Source clip", ["49 frames @ 4 fps", "720 x 480"])
    box(300, y1, 250, 110, "DDIM inversion", ["latent trajectory", "[51, 1, 13, 16, 60, 90]"])
    box(590, y1, 380, 110, "FADE denoising (50 steps)", ["frequency factorisation", "mask consumed only at i < 35"])
    box(1010, y1, 230, 110, "Output video", ["49 frames, 720x480"])
    arrow([(262, y1 + 55), (296, y1 + 55)])
    arrow([(552, y1 + 55), (586, y1 + 55)])
    arrow([(972, y1 + 55), (1006, y1 + 55)])

    mx0, mx1, my0, my1 = 590, 970, y1 + 126, y1 + 172
    d.rounded_rectangle([mx0, my0, mx1, my1], radius=8, fill=(255, 244, 235), outline=ACC, width=2)
    d.text((mx0 + 12, my0 + 11), "mask  M = 1 editable, M = 0 keep   [1,T,1,60,90]", font=f_s, fill=ACC)
    arrow([(mx0 + 190, my0), (mx0 + 190, y1 + 112)], ACC, 4)

    y2 = 360
    d.text((30, y2 - 34), "Identity-gate branch (the experimental variable)", font=font(20, True), fill=ACC)
    box(30, y2, 300, 120, "Attention capture", ["both patched_forward sites", "layers {2,3} of 4 visited"],
        edge=ACC, fill=(255, 247, 240), tcol=ACC)
    box(370, y2, 320, 120, "Drop text prefix", ["(1,48,17776,64) -> 17550 image", "226 text tokens removed"],
        edge=ACC, fill=(255, 247, 240), tcol=ACC)
    box(730, y2, 320, 120, "Mass to reference frame", ["sum over 1,350 keys", "mean over 48 heads, 13 frames"],
        edge=ACC, fill=(255, 247, 240), tcol=ACC)
    box(1090, y2, 300, 120, "g_id up to 60x90", ["min-max, x2 upsample", "continuous in [0,1]"],
        edge=ACC, fill=(255, 247, 240), tcol=ACC)
    arrow([(332, y2 + 60), (366, y2 + 60)], ACC)
    arrow([(692, y2 + 60), (726, y2 + 60)], ACC)
    arrow([(1052, y2 + 60), (1086, y2 + 60)], ACC)
    arrow([(1240, y2), (1240, my1 + 22), (mx1 + 6, my1 + 22), (mx1 + 6, my1)], ACC, 4)
    d.text((900, y2 - 34), "mask_gate = M (*) (1 - alpha * g_id),  alpha = 0.5", font=font(18, True), fill=ACC)

    y3 = 580
    box(560, y3, 420, 104, "Zero models (same formula, no model)",
        ["centred Gaussian (sigma = 0.25 min(H,W))", "full frame | permuted g_id"],
        edge=GREY, fill=(242, 242, 242), tcol=(60, 60, 60))
    arrow([(770, y3), (770, my1 + 4)], GREY)

    d.text((30, H - 34), "Only the mask source differs across arms; prompts, input clip, inversion trajectory, "
                          "step count and resolution are identical.", font=f_s, fill=GREY)
    out = os.path.join(FIG, "fig1_pipeline.png")
    img.save(out)
    print("SAVED", out, img.size)


def fig2():
    # 三臂 ID-S 均值 + 95% CI（来自 ci_table）
    arms = [("B0 (external mask)", 0.462, 0.326, 0.603, (140, 140, 140)),
            ("B1-gate (SAIG)", 0.859, 0.806, 0.906, ACC),
            ("B1-centre (prior)", 0.786, 0.723, 0.848, NAVY)]
    W, H = 1200, 700
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_b, f_a = font(17, True), font(15)
    x0, y0, w, h = 150, 80, 900, 480
    d.line([(x0, y0 + h), (x0 + w, y0 + h)], fill=GREY, width=2)
    d.line([(x0, y0), (x0, y0 + h)], fill=GREY, width=2)
    for t in range(5):
        v = t / 4.0
        y = y0 + h - h * v
        d.line([(x0, y), (x0 + w, y)], fill=(230, 230, 230), width=1)
        d.text((x0 - 52, y - 8), "%.2f" % v, font=f_a, fill=GREY)
    bw = w / (len(arms) * 2.2)
    for i, (lab, m, lo, hi, col) in enumerate(arms):
        cx = x0 + w * (i + 0.5) / len(arms)
        bx = cx - bw / 2
        top = y0 + h - h * m
        d.rectangle([bx, top, bx + bw, y0 + h], fill=col)
        # error bar
        yl = y0 + h - h * lo
        yh = y0 + h - h * hi
        d.line([(cx, yl), (cx, yh)], fill=(40, 40, 40), width=3)
        for yy in (yl, yh):
            d.line([(cx - 14, yy), (cx + 14, yy)], fill=(40, 40, 40), width=3)
        d.text((cx - 26, yh - 26), "%.3f" % m, font=f_b, fill=(30, 30, 30))
        d.text((cx - 84, y0 + h + 12), lab, font=f_a, fill=(40, 40, 40))
    d.text((x0 - 40, y0 - 52), "ID-S (DINOv2 CLS cosine), mean and 95% bootstrap CI over 8 sequences",
           font=f_b, fill=NAVY)
    out = os.path.join(FIG, "fig2_ids.png")
    img.save(out)
    print("SAVED", out, img.size)


def fig3():
    seqs = ["camel", "dog", "goat", "elephant"]
    arms = [("b0", "B0 external mask"), ("b1", "B1-gate"), ("b1c", "B1-centre"), ("b1m", "B1-gate, middle frame")]
    try:
        import imageio_ffmpeg
        FF = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        print("SKIP fig3 (imageio_ffmpeg missing)")
        return
    W = 400
    canvas = Image.new("RGB", (W * len(arms), (260 + 26) * len(seqs) + 40), "white")
    d = ImageDraw.Draw(canvas)
    f = font(15, True)
    got = 0
    for r, s in enumerate(seqs):
        for c, (pfx, lab) in enumerate(arms):
            p = os.path.join(BUNDLE, "%s_%s.mp4" % (pfx, s))
            if not os.path.exists(p):
                continue
            tmp = os.path.join(FIG, "_tmp_%s_%s.png" % (pfx, s))
            subprocess.run([FF, "-y", "-loglevel", "error", "-ss", "6", "-i", p, "-frames:v", "1", tmp], check=False)
            if not os.path.exists(tmp):
                continue
            im = Image.open(tmp).convert("RGB").resize((W, 260), Image.LANCZOS)
            x, y = c * W, 40 + r * 286
            canvas.paste(im, (x, y))
            d.text((x + 6, y - 20), "%s — %s" % (lab, s), font=f, fill=NAVY)
            os.remove(tmp)
            got += 1
    out = os.path.join(FIG, "fig3_qualitative.png")
    canvas.save(out)
    print("SAVED", out, canvas.size, "tiles", got)


def fig4():
    rows = [r for r in csv.DictReader(open(SWEEP, encoding="utf-8")) if r.get("auroc")]
    agg = {}
    for r in rows:
        k = (int(r["layer"]), int(r["ref_frame"]), r["head_agg"])
        agg.setdefault(k, []).append(float(r["auroc"]))
    mean = {k: sum(v) / len(v) for k, v in agg.items()}
    heads = ["mean", "max", "top5"]
    layers = [1, 2, 3, 0]
    refs = [0, 6, 12]
    cw, ch = 120, 90
    panel_w = cw * len(refs) + 150
    panel_h = ch * len(layers) + 90
    W, H = panel_w * len(heads), panel_h + 70
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_t, f_a, f_s = font(17, True), font(14), font(12)

    def col(v, lo=0.45, hi=0.90):
        t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
        return (int(250 - 180 * t), int(235 - 120 * t), int(215 - 90 * t))

    for pi, hd in enumerate(heads):
        ox = pi * panel_w
        d.text((ox + 10, 12), "head aggregation: %s" % hd, font=f_t, fill=NAVY)
        for li, L in enumerate(layers):
            for ri, R in enumerate(refs):
                x = ox + 110 + ri * cw
                y = 60 + li * ch
                v = mean.get((L, R, hd), float("nan"))
                d.rectangle([x, y, x + cw - 4, y + ch - 4], fill=col(v), outline=(200, 200, 200))
                d.text((x + 8, y + 8), "L%d" % L, font=f_s, fill=(30, 30, 30))
                d.text((x + 8, y + 26), "ref %d" % R, font=f_s, fill=(30, 30, 30))
                d.text((x + 8, y + 50), "%.3f" % v, font=f_a, fill=(20, 20, 20))
        for ri, R in enumerate(refs):
            d.text((ox + 110 + ri * cw + 16, 44), "frame %d" % R, font=f_s, fill=GREY)
        # 先验线
        d.line([(ox + 110, 60 + len(layers) * ch + 6), (ox + 110 + len(refs) * cw, 60 + len(layers) * ch + 6)],
               fill=ACC, width=3)
        d.text((ox + 110, 60 + len(layers) * ch + 12), "centred-Gaussian prior: 0.914 (all cells below)",
               font=f_s, fill=ACC)
    out = os.path.join(FIG, "fig4_sweep.png")
    img.save(out)
    print("SAVED", out, img.size)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
