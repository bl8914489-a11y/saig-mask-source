# -*- coding: utf-8 -*-
"""计算论文表 2/表 3 的 95% CI（对 8 段做配对 bootstrap）。

用法：python ci_table.py
输出：work/ci_table.csv（供论文引用）+ 控制台

方法：B = 20000，seed = 20261121；统计量为跨段均值；配对差直接对每段差值重采样。
"""
import csv
import json
import os
import random
import statistics

B = 20000
SEED = 20261121
AUROC = r"F:\deepseek\work\auroc_a800.csv"
METRICS = r"F:\deepseek\work\metrics_all.csv"
OUT = r"F:\deepseek\work\ci_table.csv"
SEQS = ["bear", "camel", "cows", "dog", "elephant", "flamingo", "goat", "rhino"]


def rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]


def boot_ci(xs, stat=statistics.mean, b=B, seed=SEED):
    rnd = random.Random(seed)
    n = len(xs)
    vals = []
    for _ in range(b):
        vals.append(stat([xs[rnd.randrange(n)] for _ in range(n)]))
    vals.sort()
    return vals[int(0.025 * b)], vals[int(0.975 * b)]


def main():
    # ---- AUROC ----
    a = [r for r in rows(AUROC) if r.get("auc_gid")]
    seqs_a = [(r.get("seq") or "").strip() for r in a]
    gid = [float(r["auc_gid"]) for r in a]
    cg = [float(r["center_gauss"]) for r in a]
    diff = [g - c for g, c in zip(gid, cg)]

    # ---- ID-S ----
    m = {}
    for r in rows(METRICS):
        s = (r.get("seq") or "").strip()
        v = (r.get("video") or "").strip()
        if s and v:
            m.setdefault(s, {})[v] = float(r["ID_S"])
    seqs_m = [s for s in SEQS if s in m and {"b0", "gate", "center"} <= set(m[s])]
    b0 = [m[s]["b0"] for s in seqs_m]
    gate = [m[s]["gate"] for s in seqs_m]
    cen = [m[s]["center"] for s in seqs_m]
    d_gb = [g - b for g, b in zip(gate, b0)]
    d_gc = [g - c for g, c in zip(gate, cen)]

    out = []

    def add(name, val, lo, hi, n):
        out.append(dict(quantity=name, mean=round(val, 4), ci_lo=round(lo, 4),
                        ci_hi=round(hi, 4), n=n))

    for name, xs, n in (
        ("AUROC(g_id)", gid, len(gid)),
        ("AUROC(centre-Gauss)", cg, len(cg)),
        ("AUROC paired diff (g_id - prior)", diff, len(diff)),
        ("ID-S B0", b0, len(b0)),
        ("ID-S gate", gate, len(gate)),
        ("ID-S centre", cen, len(cen)),
        ("ID-S gate - B0", d_gb, len(d_gb)),
        ("ID-S gate - centre", d_gc, len(d_gc)),
    ):
        lo, hi = boot_ci(xs)
        add(name, statistics.mean(xs), lo, hi, n)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "mean", "ci_lo", "ci_hi", "n"])
        w.writeheader()
        w.writerows(out)

    for r in out:
        print("%-34s mean=%+.4f  95%%CI [%+.4f, %+.4f]  n=%d"
              % (r["quantity"], r["mean"], r["ci_lo"], r["ci_hi"], r["n"]))
    print("WROTE", OUT)
    print("RESULT: CI_TABLE_DONE")


if __name__ == "__main__":
    main()
