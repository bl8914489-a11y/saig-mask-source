# -*- coding: utf-8 -*-
"""在优云 H20 上一次性打齐四类补丁（幂等）：
  1. edit.py：enable_model_cpu_offload
  2. edit.py：给 42 个 block 的 attn1 打 _saig_layer 标号
  3. guidance_editing_cogvideo.py：rope 帧数
  4. opt_guiders_cogvideo.py：两处 patched_forward 的 SAIG_SINK 插桩
用法：/root/venv310/bin/python /root/apply_saig_patches.py
判据：末尾 PATCHES_OK，且 counts = offload1 / tag1 / rope1 / sink2
"""
import shutil
import sys

F = "/root/saig/FADE"
EDIT = F + "/edit.py"
GUARD = F + "/diffusion_core/guiders/guidance_editing_cogvideo.py"
OPT = F + "/diffusion_core/guiders/opt_guiders_cogvideo.py"

report = {}


def backup(p):
    b = p + ".bak_saig"
    try:
        shutil.copy2(p, b)
    except Exception:
        pass


# ---------- 1+2 edit.py ----------
s = open(EDIT, encoding="utf-8").read()
backup(EDIT)
changed = False

if "enable_model_cpu_offload" not in s:
    key = ".to(args.device)"
    if key in s:
        tag_block = (
            "\n    pipe.enable_model_cpu_offload()"
            "\n    try:"
            "\n        for _i, _b in enumerate(pipe.transformer.transformer_blocks):"
            "\n            _b.attn1._saig_layer = _i"
            "\n    except Exception as _e:"
            "\n        print('SAIG_TAG_ERR', type(_e).__name__, str(_e)[:100], flush=True)"
        )
        s = s.replace(key, key + tag_block, 1)
        changed = True

if "_saig_layer" not in s and "enable_model_cpu_offload" in s:
    # offload 已在但没有 tag 块，补在 offload 行之后
    import re
    m = re.search(r"^(.*enable_model_cpu_offload\(\))\s*$", s, flags=re.M)
    if m:
        tag_block = (
            "\n    try:"
            "\n        for _i, _b in enumerate(pipe.transformer.transformer_blocks):"
            "\n            _b.attn1._saig_layer = _i"
            "\n    except Exception as _e:"
            "\n        print('SAIG_TAG_ERR', type(_e).__name__, str(_e)[:100], flush=True)"
        )
        s = s[:m.end()] + tag_block + s[m.end():]
        changed = True

if changed:
    open(EDIT, "w", encoding="utf-8").write(s)
report["offload"] = s.count("enable_model_cpu_offload")
report["tag"] = s.count("_saig_layer")

# ---------- 3 rope ----------
g = open(GUARD, encoding="utf-8").read()
backup(GUARD)
if "self.inv_latents[0].shape[1]" not in g:
    g = g.replace("_prepare_rotary_positional_embeddings(480, 720, 3,",
                  "_prepare_rotary_positional_embeddings(480, 720, self.inv_latents[0].shape[1],")
    open(GUARD, "w", encoding="utf-8").write(g)
report["rope"] = open(GUARD, encoding="utf-8").read().count("self.inv_latents[0].shape[1]")
report["gate"] = open(GUARD, encoding="utf-8").read().count("SAIG_GATE_APPLIED")

# ---------- 4 插桩 ----------
o = open(OPT, encoding="utf-8").read()
backup(OPT)
if "SAIG_SINK" not in o:
    lines = o.split("\n")
    out, hits = [], 0
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("hidden_states = F.scaled_dot_product_attention("):
            ind = ln[: len(ln) - len(ln.lstrip())]
            out.append(ind + "try:")
            out.append(ind + "    if getattr(self, '_saig_layer', None) in (2, 3, 4):")
            out.append(ind + "        SAIG_SINK.append((self._saig_layer,")
            out.append(ind + "                            query.detach().cpu().half(),")
            out.append(ind + "                            key.detach().cpu().half()))")
            out.append(ind + "except Exception as _e:")
            out.append(ind + "    print('SAIG_ERR', type(_e).__name__, str(_e)[:100], flush=True)")
            hits += 1
        out.append(ln)
    o = "SAIG_SINK = []\n\n" + "\n".join(out)
    open(OPT, "w", encoding="utf-8").write(o)
report["sink"] = open(OPT, encoding="utf-8").read().count("SAIG_SINK.append")
report["sink_sites"] = hits

print("COUNTS", report)
ok = (report["offload"] >= 1 and report["tag"] >= 1 and report["rope"] >= 1
      and report["gate"] >= 1 and report["sink"] == 2)
print("PATCHES_OK" if ok else "PATCHES_FAIL")
sys.exit(0 if ok else 3)
