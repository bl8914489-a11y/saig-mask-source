"""导出 SAIG 的 g_id（不跑完整 50 步管线，只在 1 次前向后从 FADE 内部取 q/k）。

用法（FADE 目录、venv 下）：
  python export_gid_rev1.py --video input/bear.mp4 --out saig_out/bear_gid.npy

v4 变更（2026-11-19 01:30，全部有实测出处）：
  - 【v3 继承】插桩抓到的 q/k 是 4-D (B,H,S,d)，as_shd() 统一成 (S,H,d)。
    实测证据：/root/exp2.log 旧版崩在 `S, H, d = q.shape`；
    v3 首跑打印 QK_SHAPE layer=2 raw_q=(1, 48, 17776, 64)。
  - 【v4 新增】S 不是 17,550，而是 17,776 = 226 文本 token + 13×1,350 图像 token。
    CogVideoX-5b 的 T5 文本固定补到 226（即历史日志里的 "226 警告"）。
    因此 reshape(13,1350) 会报 `cannot reshape array of size 17776 into shape (13,1350)`。
    v4 先切掉前 prefix 个文本 token（prefix = S − frames×1,350），只用图像 token 计算，
    与掩码 [1,T,1,60,90] 的语义对齐（掩码只作用在图像 token 上）。
  - 判据关键字不变：CAPTURED / SAVED / GID_OK。
"""
import argparse
import os
import sys

import numpy as np
import torch

SAIG_DIR = "/root/saig/saig_project/saig"
sys.path.insert(0, SAIG_DIR)
import core  # noqa: E402

TPF = core.FRAME_TOKENS  # 1350


def as_shd(t):
    """把捕获到的 q/k 统一成 (S,H,d)；兼容 (B,H,S,d) / (S,H,d) / (S,d)。"""
    nd = len(t.shape)
    if nd == 4:
        b, h, s, d = t.shape
        # swapaxes 在 numpy 与 torch 上同名同义，所以这一层不绑定库
        t = t.reshape(b, h, s, d)[0].swapaxes(0, 1).reshape(s, h, d)
    elif nd == 3:
        return t
    elif nd == 2:
        t = t.reshape(t.shape[0], 1, t.shape[1])
    else:
        raise ValueError("unexpected q/k rank %d shape=%s" % (nd, tuple(t.shape)))
    return t


def text_prefix_len(S, frames, tpf=TPF, cap=512):
    """返回序列里文本 token 数（图像 token 都在尾部）。

    S = prefix + frames*tpf；prefix 必须落在 [0, cap]，否则说明假设被打破，
    宁可报错也不静默截断（诚实边界：数字必须可解释）。
    """
    prefix = S - frames * tpf
    if prefix < 0 or prefix > cap:
        raise ValueError(
            "sequence len %d not explainable by %d frames x %d tokens (prefix=%d)"
            % (S, frames, tpf, prefix))
    return prefix


def m_from_qk_torch(q, k, first_frame_len=TPF, scale=None, pblock=512):
    """q,k: (S,H,d) 图像 token -> (S,) 每 token 到首帧的注意力质量（逐头平均）。"""
    q = as_shd(q)
    k = as_shd(k)
    S, H, d = q.shape
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if scale is None:
        scale = 1.0 / (d ** 0.5)
    acc = torch.zeros(S, device=dev, dtype=torch.float32)
    for h in range(H):
        qh = q[:, h, :].float().to(dev)
        kh = k[:, h, :].float().to(dev)
        num = torch.zeros(S, device=dev)
        den = torch.zeros(S, device=dev)
        for i in range(0, S, pblock):
            j = min(i + pblock, S)
            s = (qh[i:j] @ kh.t()) * scale
            s = s - s.max(dim=1, keepdim=True).values
            e = torch.exp(s)
            num[i:j] = e[:, :first_frame_len].sum(dim=1)
            den[i:j] = e.sum(dim=1)
        acc += num / den.clamp_min(1e-20)
    return (acc / H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/root/saig/CogVideoX-5b")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--frames", type=int, default=13, help="潜帧数（49 帧输入 -> 13）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sys.argv = [
        "edit.py", "--model_path", args.model,
        "--init_prompt", "a subject", "--edit_prompt", "a subject",
        "--device", "cuda:0", "--dtype", "bf16",
        "--input_video_path", args.video,
        "--latent_trajectory_path", args.video.replace("input/", "inversion/").replace(".mp4", ".pt"),
        "--mask_path", args.video.replace("input/", "mask/").replace(".mp4", ".pt"),
        "--config_path", "configs/bear.yaml", "--fps", "4",
        "--num_inference_steps", str(args.steps),
        "--output_video_path", "output/_probe.mp4",
    ]
    import runpy
    sys.path.insert(0, os.getcwd())
    try:
        runpy.run_path("edit.py", run_name="__main__")
    except Exception as e:
        print("RUN_EXC", type(e).__name__, str(e)[:120], flush=True)

    from diffusion_core.guiders import opt_guiders_cogvideo as og
    sink = og.SAIG_SINK
    print("CAPTURED", len(sink), flush=True)
    if not sink:
        print("NO_SIGNAL")
        return 2

    by_layer = {}
    for item in sink:
        li = item[0]
        q = item[1]
        k = item[2]
        if li not in by_layer:
            q3 = as_shd(q)
            k3 = as_shd(k)
            S0 = q3.shape[0]
            print("QK_SHAPE layer=%s raw_q=%s raw_k=%s S0=%d" % (
                li, tuple(q.shape), tuple(k.shape), S0), flush=True)
            pfx = text_prefix_len(S0, args.frames)
            print("GID_PREFIX layer=%s prefix=%d frames=%d tpf=%d S_img=%d" % (
                li, pfx, args.frames, TPF, S0 - pfx), flush=True)
            by_layer[li] = (q3[pfx:].contiguous(), k3[pfx:].contiguous())
    ms = []
    for li in sorted(by_layer):
        q, k = by_layer[li]
        S, H, d = q.shape
        print("SHAPE_OK layer=%s S=%d H=%d d=%d" % (li, S, H, d), flush=True)
        m = m_from_qk_torch(q, k)
        ms.append(m)
        print(f"layer{li} m: min={m.min():.4f} max={m.max():.4f}", flush=True)
        del q, k
        by_layer[li] = None
        torch.cuda.empty_cache()

    g = ms[0]
    for m in ms[1:]:
        g = g + m
    g = g / len(ms)
    S = g.numel()
    T = S // TPF
    if S % TPF != 0:
        print("BAD_LEN", S, TPF)
        return 3
    g = g.cpu().numpy().reshape(T, TPF).astype(np.float32)
    np.save(args.out, g)
    print("SAVED", args.out, g.shape, float(g.min()), float(g.max()), flush=True)
    print("RESULT: GID_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
