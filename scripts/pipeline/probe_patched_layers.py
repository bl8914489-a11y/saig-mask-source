"""B1 hook 的最小验证探针（不改 FADE 任何算法，只证明"钩子能抓到信号"）。

目的：在真实模型上证明三件事
  1) `transformer_blocks[i].attn1` 能被挂上；
  2) pre-hook 能从输入 x 用模块自身权重重算出 q/k（形状 = (S, 48, 64)）；
  3) m(p) -> g_id 的形状与取值范围正确（S = 潜帧数 × 1350）。

用法（在 FADE 目录下）：
  python /root/saig/saig_project/scripts/probe_patched_layers.py --frames 8  --steps 1
  python /root/saig/saig_project/scripts/probe_patched_layers.py --frames 49 --steps 1

设计取舍：
  * 8 帧（3 潜帧）时**不开 offload**，显存够，先验证机制；
  * 49 帧（13 潜帧）时必须开 offload，此时 accelerate 的 hook 与本探针的
    hook 执行顺序可能让模块还停在 CPU —— 本探针会按 `to_q.weight.device`
    把 x 搬过去再算，**只搬这一次，不动权重**。
  * 本探针**不接 RoPE**（与 saig/hook.py 的 ROPE=False 一致），结果只用于
    验证接线，不用于论文数值。
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SAIG_DIR = os.path.abspath(os.path.join(HERE, "..", "saig"))
sys.path.insert(0, SAIG_DIR)

import hook as saig_hook  # noqa: E402

SINK = []


def patch_pipeline_factory(pipe_cls):
    """在 pipe 创建出来的瞬间挂 hook，然后才返回给 edit.py / 调用方。"""
    orig = pipe_cls.from_pretrained

    def patched(*args, **kwargs):
        pipe = orig(*args, **kwargs)
        detach = saig_hook.attach(pipe, layers=(2, 3, 4), sink=SINK)
        print("HOOK_ATTACHED layers=(2,3,4)", flush=True)
        patched._detach = detach
        return pipe

    pipe_cls.from_pretrained = staticmethod(patched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/saig/CogVideoX-5b")
    ap.add_argument("--video", default=None, help="mp4 路径；默认按 --frames 选")
    ap.add_argument("--frames", type=int, default=8, choices=[8, 49])
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--offload", action="store_true")
    args = ap.parse_args()

    video = args.video or ("input/bear.mp4" if args.frames == 8 else "input/bear49.mp4")
    print(f"VIDEO={video} FRAMES={args.frames} STEPS={args.steps} OFFLOAD={args.offload}", flush=True)

    from diffusers import CogVideoXPipeline
    patch_pipeline_factory(CogVideoXPipeline)

    sys.argv = [
        "edit.py",
        "--model_path", args.model,
        "--init_prompt", "a bear",
        "--edit_prompt", "a panda",
        "--device", "cuda:0",
        "--dtype", "bf16",
        "--input_video_path", video,
        "--latent_trajectory_path", video.replace("input/", "inversion/").replace(".mp4", ".pt"),
        "--mask_path", video.replace("input/", "mask/").replace(".mp4", ".pt"),
        "--config_path", "configs/bear.yaml",
        "--fps", "4",
        "--num_inference_steps", str(args.steps),
        "--local_blend_end_step", "35",
        "--output_video_path", "output/probe.mp4",
    ]

    import runpy
    sys.path.insert(0, os.getcwd())  # edit.py does `from diffusion_core...` -> needs cwd on sys.path
    try:
        runpy.run_path("edit.py", run_name="__main__")
    except Exception as e:  # 探针允许在收尾阶段失败，只要有信号就够了
        print("PROBE_RUN_EXC", type(e).__name__, str(e)[:160], flush=True)

    print(f"CAPTURED={len(SINK)}", flush=True)
    if not SINK:
        try:
            from diffusion_core.guiders import opt_guiders_cogvideo as _og
            SINK.extend(_og.SAIG_SINK)
            print(f"CAPTURED_FROM_FADE={len(SINK)}", flush=True)
        except Exception as e:
            print("SINK_READ_FAIL", type(e).__name__, str(e)[:120], flush=True)
    if not SINK:
        print("RESULT: NO_SIGNAL")
        return 2

    by_layer = {}
    for li, (q, k) in SINK:
        by_layer.setdefault(li, []).append((q, k))
    for li in sorted(by_layer):
        q, k = by_layer[li][0]
        print(f"layer{li}: q={tuple(q.shape)} k={tuple(k.shape)} dtype={q.dtype}", flush=True)

    q, k = SINK[0][1]
    qn = q.detach().float().cpu().numpy()
    kn = k.detach().float().cpu().numpy()
    m = saig_hook.m_from_qk(qn, kn, first_frame_len=saig_hook.FRAME_TOKENS)
    print(f"m: shape={m.shape} min={m.min():.4f} max={m.max():.4f} mean={m.mean():.4f}", flush=True)
    T = m.size // saig_hook.FRAME_TOKENS
    g = saig_hook.build_gid([m], T=T, tpf=saig_hook.FRAME_TOKENS)
    gf = g.reshape(T, saig_hook.FRAME_TOKENS)
    print(f"g_id: T={T} range=[{g.min():.3f},{g.max():.3f}] "
          f"per_frame_dyn_min={(gf.max(1)-gf.min(1)).min():.3f}", flush=True)
    print("RESULT: HOOK_SIGNAL_OK" if T in (3, 13) else "RESULT: T_UNEXPECTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
