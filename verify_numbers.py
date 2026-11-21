# -*- coding: utf-8 -*-
"""Recompute every headline number of the paper from data/ (no GPU needed).

Usage:  python verify_numbers.py
Exit 0 and "ALL NUMBERS VERIFIED" on success.
"""
import csv
import json
import os
import random
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
B = 20000
SEED = 20261121
fails = []


def rows(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        raise SystemExit("missing data file: " + p)
    with open(p, encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]


def chk(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def boot_ci(xs, b=B, seed=SEED):
    rnd = random.Random(seed)
    n = len(xs)
    vals = sorted(statistics.mean([xs[rnd.randrange(n)] for _ in range(n)]) for _ in range(b))
    return vals[int(0.025 * b)], vals[int(0.975 * b)]


def main():
    # ---- AUROC table ----
    a = [r for r in rows("auroc_gate_metric.csv") if r.get("auc_gid")]
    gid = [float(r["auc_gid"]) for r in a]
    cg = [float(r["center_gauss"]) for r in a]
    d = [g - c for g, c in zip(gid, cg)]
    chk(len(a) == 8, "8 DAVIS sequences (got %d)" % len(a))
    chk(abs(statistics.mean(gid) - 0.6171) < 5e-4, "AUROC mean = 0.6171 (got %.4f)" % statistics.mean(gid))
    chk(abs(statistics.mean(cg) - 0.9144) < 5e-4, "prior mean = 0.9144 (got %.4f)" % statistics.mean(cg))
    chk(abs(statistics.mean(d) + 0.2973) < 5e-4, "paired diff = -0.2973 (got %.4f)" % statistics.mean(d))
    lo, hi = boot_ci(d)
    chk(lo < statistics.mean(d) < hi, "paired-diff CI brackets the mean")
    chk(abs(lo + 0.337) < 0.01 and abs(hi + 0.252) < 0.01, "paired-diff 95%% CI ~ [-0.337, -0.252] (got [%.3f, %.3f])" % (lo, hi))

    # ---- video-level table ----
    m = [r for r in rows("metrics_all.csv") if (r.get("video") or "").strip()]
    chk(len(m) == 24, "24 scored videos = 8 sequences x 3 arms (got %d)" % len(m))
    per = {}
    for r in m:
        per.setdefault(r["seq"], {})[r["video"]] = (float(r["ID_S"]), float(r["Sem_T_target"]))
    seqs = [s for s in per if {"b0", "gate", "center"} <= set(per[s])]
    chk(len(seqs) == 8, "all 8 sequences have three arms")
    b0 = [per[s]["b0"][0] for s in seqs]
    ga = [per[s]["gate"][0] for s in seqs]
    ce = [per[s]["center"][0] for s in seqs]
    chk(abs(statistics.mean(b0) - 0.462) < 1e-3, "ID-S B0 mean = 0.462")
    chk(abs(statistics.mean(ga) - 0.859) < 1e-3, "ID-S gate mean = 0.859")
    chk(abs(statistics.mean(ce) - 0.786) < 1e-3, "ID-S centre mean = 0.786")
    gain = [g - c for g, c in zip(ga, ce)]
    chk(sum(1 for x in gain if x > 0) == 7, "gate beats centre on 7/8 sequences")
    chk(abs(statistics.mean(gain) - 0.073) < 2e-3, "gate - centre = +0.073 (got %+.4f)" % statistics.mean(gain))

    # ---- sweep ----
    s = [r for r in rows("sweep_agg.csv") if r.get("auroc")]
    chk(len(s) == 108, "108-cell sweep (got %d)" % len(s))
    chk(abs(max(float(r["auroc"]) for r in s) - 0.8603) < 5e-4, "best sweep cell = 0.8603")
    frozen = [float(r["auroc"]) for r in s
              if int(r["layer"]) == 2 and int(r["ref_frame"]) == 0 and r["head_agg"] == "mean"]
    chk(abs(statistics.mean(frozen) - 0.5974) < 1e-3, "frozen-protocol cell mean = 0.5974")

    # ---- alpha sweep ----
    al = [r for r in rows("alpha_summary.csv") if (r.get("video") or "").strip()]
    chk(len(al) == 12, "alpha sweep 3 sequences x 4 arms (got %d)" % len(al))

    # ---- control arms ----
    ctl = 0
    for s_name in ("camel", "dog", "flamingo"):
        p = os.path.join(DATA, "metrics_control_%s.csv" % s_name)
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig") as f:
                ctl += len([r for r in csv.DictReader(f) if (r.get("video") or "").strip()])
    chk(ctl == 18, "control arms: 3 sequences x 6 arms (got %d)" % ctl)

    # ---- CI table consistency ----
    ci = {r["quantity"]: r for r in rows("confidence_intervals.csv")}
    chk(abs(float(ci["AUROC(g_id)"]["ci_lo"]) - 0.583) < 5e-3, "AUROC CI lower ~0.583")
    chk(abs(float(ci["ID-S gate - centre"]["ci_hi"]) - 0.116) < 5e-3, "gate-centre CI upper ~0.116")

    # ---- sensitivity ----
    sen = [r for r in rows("sensitivity_variants.csv") if r.get("variant")]
    chk(len(sen) == 4, "4 sensitivity variants")
    chk(abs(float(sen[0]["mean_auroc"]) - 0.6171) < 5e-4, "sensitivity baseline = 0.6171")

    print()
    if fails:
        print("RESULT: %d CHECK(S) FAILED" % len(fails))
        return 1
    print("RESULT: ALL NUMBERS VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
