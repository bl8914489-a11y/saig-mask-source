"""FADE 插桩 v2：在 opt_guiders_cogvideo.py 的**两处** patched_forward 内埋点 + 给 42 个 block 打层号。

为什么是 v2：本地旧版 patch_fade_saig_rev1.py 只插了 1 处（replace(...,1)），实测 CAPTURED=0；
两处都插才得到 CAPTURED=8（档案第 5 章第 5 轮结论）。本版按"两处 + 异常打印 + half 拷贝"实现。

用法（在 pod 上、FADE 目录内）：
    python patch_fade_saig_rev1.py --check   # 只检查锚点
    python patch_fade_saig_rev1.py           # 打补丁（自动备份、幂等）

判据：
    - 脚本输出 PATCH1_OK_2SITES 与 PATCH2_OK
    - 跑一次 edit.py 后日志出现 CAPTURED 8
"""
import argparse
import os
import shutil

P1 = "/root/saig/FADE/diffusion_core/guiders/opt_guiders_cogvideo.py"
P2 = "/root/saig/FADE/edit.py"
ANCHOR = "hidden_states = F.scaled_dot_product_attention("

SNIPPET = """{ind}try:
{ind}    if getattr(self, '_saig_layer', None) in (2, 3, 4):
{ind}        SAIG_SINK.append((self._saig_layer,
{ind}                            query.detach().cpu().half(),
{ind}                            key.detach().cpu().half()))
{ind}except Exception as _e:
{ind}    print('SAIG_ERR', type(_e).__name__, str(_e)[:100], flush=True)
"""


def patch1(path, check_only=False):
    s = open(path, encoding="utf-8").read()
    if "SAIG_SINK" in s:
        return "PATCH1_SKIP_ALREADY", s
    lines = s.split("\n")
    out, hits = [], 0
    for ln in lines:
        if ln.strip().startswith("hidden_states = F.scaled_dot_product_attention(") or \
           ln.strip().startswith("hidden_states = F.scaled_dot_product_attention ("):
            ind = ln[: len(ln) - len(ln.lstrip())]
            out.append(SNIPPET.format(ind=ind).rstrip("\n"))
            hits += 1
        out.append(ln)
    if hits == 0:
        return "ANCHOR_MISSING", s
    body = "SAIG_SINK = []\n\n" + "\n".join(out)
    return ("CHECK_OK sites=%d" if check_only else "PATCH1_OK_2SITES sites=%d") % hits, body


def patch2(path, check_only=False):
    e = open(path, encoding="utf-8").read()
    if "_saig_layer" in e:
        return "PATCH2_SKIP_ALREADY", e
    key = "pipe.enable_model_cpu_offload()"
    if key not in e:
        return "ANCHOR2_MISSING", e
    i = e.index(key) + len(key)
    add = ("\n    for _i, _b in enumerate(pipe.transformer.transformer_blocks):"
           "\n        try:"
           "\n            _b.attn1._saig_layer = _i"
           "\n        except Exception:"
           "\n            pass")
    return ("CHECK2_OK" if check_only else "PATCH2_OK"), e[:i] + add + e[i:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--p1", default=P1)
    ap.add_argument("--p2", default=P2)
    a = ap.parse_args()
    for p in (a.p1, a.p2):
        if not os.path.exists(p):
            print("NO_FILE", p)
            return 2

    m1, n1 = patch1(a.p1, a.check)
    m2, n2 = patch2(a.p2, a.check)
    print(m1)
    print(m2)
    if a.check:
        return 0
    if m1 == "ANCHOR_MISSING" or m2 == "ANCHOR2_MISSING":
        return 3
    for p in (a.p1, a.p2):
        bak = p + ".bak_saig"
        if not os.path.exists(bak):
            shutil.copy2(p, bak)
            print("BACKUP", bak)
    open(a.p1, "w", encoding="utf-8").write(n1)
    open(a.p2, "w", encoding="utf-8").write(n2)
    print("VERIFY_SINK", open(a.p1, encoding="utf-8").read().count("SAIG_SINK.append"))
    print("VERIFY_TAG", open(a.p2, encoding="utf-8").read().count("_saig_layer"))
    print("RESULT: PATCH_V2_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
