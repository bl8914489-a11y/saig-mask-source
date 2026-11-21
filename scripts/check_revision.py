# -*- coding: utf-8 -*-
"""v1.5 校验脚本（对应修改单第 7 节 + 数字复算）。"""
import csv
import os
import re
import statistics

from docx import Document

DOCX = r"F:\deepseek\saig_project\paper\SAIG_论文_v1.5_TVC_20261121.docx"
DATA = r"F:\deepseek\work"
fails = []


def chk(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


doc = Document(DOCX)
text = "\n".join(p.text for p in doc.paragraphs)
for t in doc.tables:
    for row in t.rows:
        for c in row.cells:
            text += "\n" + c.text

# 1) 禁止残留
for bad in ["1,131", "no confidence intervals", "random-layer", "short by 0.054",
            "negative-result framing", "Draft v1", "Is Self-Attention an Object Selector"]:
    chk(bad not in text, "禁止残留: %s" % bad)

# 2) 必备内容
for need in ["95% CI", "Where Should the Keep-Mask Come From", "permuted g_id",
             "scoped to CogVideoX-5b"]:
    chk(need in text, "必备内容: %s" % need)

# 3) 结构
chk(len(doc.inline_shapes) == 4, "图数 = 4（实际 %d）" % len(doc.inline_shapes))
caps = [p.text for p in doc.paragraphs if p.text.startswith("Table ")]
nums = [int(m.group(1)) for m in (re.match(r"Table\s+(\d+)", c) for c in caps) if m]
chk(nums == list(range(1, len(nums) + 1)), "表号连续: %s" % nums)
refs = [p.text for p in doc.paragraphs if re.match(r"^\[\d+\]", p.text)]
chk(len(refs) >= 30, "参考文献 >= 30（实际 %d）" % len(refs))
decl = [p.text for p in doc.paragraphs]
chk(any("Data availability" in x for x in decl), "有 Data availability 声明")
chk(any("Ethics approval" in x for x in decl), "有 Ethics approval 声明")
chk(any("[FAMILY NAME" in x for x in decl), "标题页有作者占位符")

# 4) 数字复算
rows = [r for r in csv.DictReader(open(os.path.join(DATA, "auroc_a800.csv"), encoding="utf-8-sig"))
        if r.get("auc_gid")]
a = [float(r["auc_gid"]) for r in rows]
chk(abs(statistics.mean(a) - 0.617) < 0.001 and len(rows) == 8, "AUROC 复算 0.617 n=8")

m = [r for r in csv.DictReader(open(os.path.join(DATA, "metrics_all.csv"), encoding="utf-8-sig"))
     if (r.get("video") or "").strip()]
g = [float(r["ID_S"]) for r in m if r["video"] == "gate"]
chk(abs(statistics.mean(g) - 0.859) < 0.001 and len(m) == 24, "ID-S gate 均值 0.859，24 行")

s = [r for r in csv.DictReader(open(os.path.join(DATA, "sweep_agg.csv"), encoding="utf-8-sig"))
     if r.get("auroc")]
chk(len(s) == 108, "sweep 108 行")
chk(abs(max(float(r["auroc"]) for r in s) - 0.8603) < 0.001, "sweep 最大 0.8603")

# 5) 附录 C 引用的文件必须存在（data/ 相对路径）
data_files = set(os.listdir(DATA))
cited = re.findall(r"data/([A-Za-z0-9_\-]+\.(?:csv|json))", text)
missing = sorted({c for c in cited if c not in data_files})
chk(not missing, "附录引用的 data/ 文件都存在（缺: %s）" % missing)

print()
print("RESULT: %s  (%d 项失败)" % ("ALL_CHECKS_PASSED" if not fails else "CHECKS_FAILED", len(fails)))
