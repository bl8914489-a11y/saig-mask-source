import sys

P1 = "/root/saig/FADE/diffusion_core/guiders/opt_guiders_cogvideo.py"
P2 = "/root/saig/FADE/edit.py"

# ---- patch 1: 在 FADE 自己的 patched_forward 内部抄一份 q/k ----
s = open(P1, encoding="utf-8").read()
if "SAIG_SINK" in s:
    print("PATCH1_SKIP")
else:
    anchor = "                hidden_states = F.scaled_dot_product_attention("
    if anchor not in s:
        print("ANCHOR1_MISSING")
        sys.exit(1)
    ins = (
        "                try:\n"
        "                    if getattr(self, \"_saig_layer\", None) in (2, 3, 4):\n"
        "                        SAIG_SINK.append((self._saig_layer, query.detach().float(), key.detach().float()))\n"
        "                except Exception:\n"
        "                    pass\n"
    )
    s = "SAIG_SINK = []\n" + s.replace(anchor, ins + anchor, 1)
    open(P1, "w", encoding="utf-8").write(s)
    print("PATCH1_OK")

# ---- patch 2: 给 attn1 打层号 ----
e = open(P2, encoding="utf-8").read()
if "_saig_layer" in e:
    print("PATCH2_SKIP")
else:
    anchor2 = "pipe.enable_model_cpu_offload()"
    if anchor2 not in e:
        print("ANCHOR2_MISSING")
        sys.exit(1)
    _i = e.index(anchor2) + len(anchor2)
    add = (
        "\n    for _i, _b in enumerate(pipe.transformer.transformer_blocks):"
        + "\n        try:"
        + "\n            _b.attn1._saig_layer = _i"
        + "\n        except Exception:"
        + "\n            pass"
    )
    open(P2, "w", encoding="utf-8").write(e[:_i] + add + e[_i:])
    print("PATCH2_OK")

print("DONE")
