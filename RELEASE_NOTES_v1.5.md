# v1.5 — code, measurement tables and artefact bundle

Release accompanying the manuscript **“Where Should the Keep-Mask Come From? A Controlled Study of Mask Sources in Training-Free Video Editing”** (prepared for *The Visual Computer*).

## Assets in this release

| Asset | Size | SHA256 |
|---|---|---|
| `saig_results_20261121.tgz` | 45.7 MB | `1cf07ae1221bf72675555c85f14ad55167156431fe3d264545b9a6ed77e5b9bc` |

The archive contains:

- `output/` — 39 rendered videos: `b0_*` (external mask), `b1_*` (attention gate), `b1c_*` (centred prior), `b1m_*` (middle-frame gate), `b1a0.3_*` / `b1a0.7_*` (gate strength), `m5_nomask_*`, `m6_gtmask_*`, `m3_refine_*` (control arms)
- `saig_out/` — 8 × `*_gid.npy`, per-frame attention mass (13 × 1350)
- `masks/` — 9 × `gid_*.npy`, gate maps at 60 × 90
- `logs/` — `runlog_b0.csv`, `runlog_b1.csv`, per-run logs, AUROC and metric tables

## Verify without a GPU

```bash
git clone <this repo>
cd <repo>
python verify_numbers.py      # -> ALL NUMBERS VERIFIED
```

`verify_numbers.py` re-reads `data/` and asserts every headline number in the paper (AUROC 0.6171, prior 0.9144, paired difference −0.2973, ID-S 0.462 / 0.859 / 0.786, 108-cell sweep maximum 0.8603, control arms, and the bootstrap intervals).

## Reproduce the experiments (GPU)

See `docs/REPRODUCE.md`. Expected after a correct build: `MEAN_AUC_GID 0.6171`, `MEAN_MARGIN -0.2973`, `GATE_AUROC FAIL`.

## Notes

- `scripts/metrics_compare.py` is the exact revision (2025-11-18) that produced the published tables.
- The study uses DAVIS 2017 and YouTube-VOS under their original licences; derived videos are redistributed for reproducibility only.
