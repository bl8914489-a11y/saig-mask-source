#!/bin/bash
# 正确的 p7：8 段 × 3 臂 ID-S / Sem-T（用本机新版 metrics_compare.py）
exec > /root/p7_full.log 2>&1
set -x
F=/root/saig/FADE
S=/root/saig/saig_project/scripts
V=/root/venv310/bin/python
LOGS=/root/logs
SEQS="bear camel cows dog elephant flamingo goat rhino"

sub() { case $1 in bear) echo bear;; camel) echo camel;; cows) echo cow;; dog) echo dog;;
  elephant) echo elephant;; flamingo) echo flamingo;; goat) echo goat;; rhino) echo rhinoceros;; esac; }
tgt() { case $1 in bear) echo panda;; camel) echo horse;; cows) echo zebra;; dog) echo wolf;;
  elephant) echo rhinoceros;; flamingo) echo swan;; goat) echo sheep;; rhino) echo hippopotamus;; esac; }

cd "$F" || exit 1
echo "P7FULL_START $(date -Is)"

for s in $SEQS; do
  echo "=== $s ==="
  for v in "output/b0_$s.mp4" "output/b1_$s.mp4" "output/b1c_$s.mp4"; do
    [ -f "$v" ] || { echo "MISS $v"; continue 2; }
  done
  $V "$S/metrics_compare.py" \
    --seq "$s" --src "input/$s.mp4" --ann "/root/saig/DAVIS/Annotations/480p/$s" \
    --init "a $(sub $s)" --target "a $(tgt $s)" \
    --videos "b0=output/b0_$s.mp4" "gate=output/b1_$s.mp4" "center=output/b1c_$s.mp4" \
    --out "$LOGS/metrics_$s.csv" 2>&1 | tail -6
  echo "RC_$s=$?"
done

# 汇总
$V - <<'PY'
import csv, glob, os
rows, cols = [], None
for p in sorted(glob.glob("/root/logs/metrics_*.csv")):
    name = os.path.basename(p)[8:-4]
    if name in ("all", "pilot"):
        continue
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["seq"] = name
            if cols is None:
                cols = ["seq"] + [c for c in r if c != "seq"]
            rows.append(r)
out = "/root/logs/metrics_all.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print("WROTE", out, "rows", len(rows), "seqs", len({r["seq"] for r in rows}))
PY

echo "P7FULL_DONE $(date -Is)"
