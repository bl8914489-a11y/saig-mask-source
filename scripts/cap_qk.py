# -*- coding: utf-8 -*-
"""保存 q/k 供聚合方式扫描（不用重跑：一次前向把每层 q/k 存盘）。

用法（FADE 目录内，SAIG_CAPTURE=1）：
  /root/venv310/bin/python /root/cap_qk.py camel
输出：/root/qk/<seq>_L<NN>.pt   dict(q=half(4D or 3D), k=..., shape=...)
"""
import os
import sys

import torch

OUT = "/root/qk"


def main():
    seq = sys.argv[1] if len(sys.argv) > 1 else "camel"
    os.makedirs(OUT, exist_ok=True)

    sys.argv = [
        "edit.py", "--model_path", "/root/saig/CogVideoX-5b",
        "--init_prompt", "a subject", "--edit_prompt", "a subject",
        "--device", "cuda:0", "--dtype", "bf16",
        "--input_video_path", "input/%s.mp4" % seq,
        "--latent_trajectory_path", "inversion/%s.pt" % seq,
        "--mask_path", "mask/%s.pt" % seq,
        "--config_path", "configs/bear.yaml", "--fps", "4",
        "--num_inference_steps", "1",
        "--output_video_path", "output/_qk.mp4",
    ]
    import runpy
    sys.path.insert(0, os.getcwd())
    try:
        runpy.run_path("edit.py", run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:
        print("RUN_EXC", type(e).__name__, str(e)[:150], flush=True)

    from diffusion_core.guiders import opt_guiders_cogvideo as og
    sink = og.SAIG_SINK
    print("SINK", len(sink), flush=True)
    seen = {}
    for it in sink:
        li, q, k = it[0], it[1], it[2]
        if li in seen:
            continue
        seen[li] = True
        torch.save({"q": q, "k": k, "raw_q": tuple(q.shape), "raw_k": tuple(k.shape)},
                   os.path.join(OUT, "%s_L%02d.pt" % (seq, li)))
        print("SAVED_QK %s L%02d %s %s" % (seq, li, tuple(q.shape), tuple(k.shape)), flush=True)
    print("DONE", seq, "layers", sorted(seen), flush=True)
    sink.clear()


if __name__ == "__main__":
    main()
