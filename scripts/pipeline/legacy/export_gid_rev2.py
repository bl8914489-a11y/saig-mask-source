"""导出 SAIG 的 g_id（不跑完整 50 步管线，只在 1 次前向后从 FADE 内部取 q/k）。

用法（FADE 目录、venv 下）：
  python export_gid_rev1.py --video input/bear.mp4 --out saig_out/bear_gid.npy

v3 变更（2026-11-19 01:20，本机实测依据）：
  - pod 上插桩实际抓到的 q/k 是 4-D (B,H,S,d)：opt_guiders_cogvideo.py:486 与 :818
    都是 `query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)`。
    旧版 `S, H, d = q.shape` 只接受 3-D，因此必然报
    `ValueError: too many values to unpack (expected 3)`（见 /root/exp2.log 第 51-58 行）。
  - 新增 as_shd()：把 (B,H,S,d) / (S,H,d) / (S,d) 统一成 (S,H,d)，不再依赖猜测。
  - 新增 QK_SHAPE 证据行；其余 CLI、输出、判据关键字（CAPTURED / SAVED / GID_OK）不变。
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
    """把捕获到的 q/k 统一成 (S,H,d)；兼容 (B,H,S,d) / (S,H,d) / (S,d)。

    只依赖 .shape 与 reshape/transpose，torch 张量与 numpy 数组都能用。
    """
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


def m_from_qk_torch(q, k, first_frame_len=TPF, scale=None, pblock=512):
    """q,k: (S,H,d) -> (S,) 每 token 到首帧的注意力质量（逐头平均）。"""
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
            print("QK_SHAPE layer=%s raw_q=%s raw_k=%s" % (
                li, tuple(q.shape), tuple(k.shape)), flush=True)
            by_layer[li] = (as_shd(q), as_shd(k))
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
    g = g.cpu().numpy().reshape(T, TPF).astype(np.float32)
    np.save(args.out, g)
    print("SAVED", args.out, g.shape, float(g.min()), float(g.max()), flush=True)
    print("RESULT: GID_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
