# -*- coding: utf-8 -*-
"""从实验产物生成 n=8 的论文表格（Table 3 三臂 + 汇总），数据驱动、不写死数字。

输入：
  /root/logs/runlog_b1.csv          （编辑台账：seq,stage,wall_s,rc）
  /root/logs/metrics_all.csv      （p7 汇总）
  /root/logs/auroc_yun.csv        （闸门 AUROC）
输出：
  /root/logs/table3_8seq.csv      （段 × 臂 的 ID-S / Sem-T）
  /root/logs/table_summary.json   （给论文用的汇总数字）

用法：/root/venv310/bin/python /root/make_tables.py
"""
import csv
import json
import os

LOGS = "/root/logs"
OUT_CSV = os.path.join(LOGS, "table3_8seq.csv")
OUT_JSON = os.path.join(LOGS, "table_summary.json")
SEQS = ["bear", "camel", "cows", "dog", "elephant", "flamingo", "goat", "rhino"]


def read_csv(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = read_csv(os.path.join(LOGS, "metrics_all.csv"))
    per_seq = {}
    if rows:
        # p7 的汇总列名以实际为准，做模糊匹配
        keys = rows[0].keys()
        def pick(r, *cands):
            for c in cands:
                for k in keys:
                    if k.lower() == c.lower():
                        return r[k]
            return ""

        for r in rows:
            seq = pick(r, "seq", "sequence", "name")
            arm = pick(r, "arm", "variant")
            if not seq or not arm:
                continue
            per_seq.setdefault(seq, {})[arm] = dict(
                IDS=pick(r, "ID_S", "IDS", "id_s", "id_similarity"),
                SemT=pick(r, "Sem_T_target", "SemT", "sem_t", "sem_t_target"),
            )

    out_rows = [["Segment", "Arm", "ID-S", "Sem-T"]]
    for s in SEQS:
        for arm in ("b0", "gate", "center"):
            d = per_seq.get(s, {}).get(arm)
            if d:
                out_rows.append([s, arm, d["IDS"], d["SemT"]])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)

    # 台账统计
    led = read_csv(os.path.join(LOGS, "runlog_b1.csv"))
    edits = [r for r in led if "b1" in r.get("stage", "")]
    ok = [r for r in edits if r.get("rc") == "0"]
    walls = [int(r["wall_s"]) for r in ok if r.get("wall_s", "").isdigit()]

    auroc = read_csv(os.path.join(LOGS, "auroc_yun.csv"))
    aucs = []
    for r in auroc:
        for k in r:
            if k.lower().startswith("auc_gid"):
                try:
                    aucs.append(float(r[k]))
                except ValueError:
                    pass

    summary = dict(
        n_segments=len({r[0] for r in out_rows[1:]}),
        edits_ok=len(ok),
        edits_failed=len(edits) - len(ok),
        edit_wall_min=round(min(walls) / 60, 1) if walls else None,
        edit_wall_max=round(max(walls) / 60, 1) if walls else None,
        auroc_mean=round(sum(aucs) / len(aucs), 4) if aucs else None,
        auroc_n=len(aucs),
        table_rows=len(out_rows) - 1,
    )
    json.dump(summary, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("WROTE", OUT_CSV, len(out_rows) - 1, "rows")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    print("RESULT: TABLES_V2_DONE")


if __name__ == "__main__":
    main()
