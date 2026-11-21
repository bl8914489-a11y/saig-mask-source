# Data

Every number in the manuscript comes from one of these files. `verify_numbers.py` at the repository root re-reads all of them and asserts the values printed in the paper.

| File | Rows | What it contains | Paper |
|---|---|---|---|
| `auroc_gate_metric.csv` | 8 | per-sequence AUROC of `g_id` against the subject mask, the centred Gaussian, the full-frame constant and the permuted gate, plus the margin | Table 2, Appendix A |
| `metrics_all.csv` | 24 | ID-S and Sem-T for 8 sequences × 3 arms (B0 / gate / centre) | Table 3, Fig. 2 |
| `pixel_metrics.csv` | 18 | MAE, SSIM and the fraction of pixels with \|Δ\| > 15 for every arm pair and timestamp | Table 4 |
| `alpha_summary.csv` | 12 | ID-S and Sem-T at α = 0.3 / 0.5 / 0.7 and for B0 | Table 5 |
| `sweep_agg.csv` | 108 | the 108-cell sweep: 4 layers × 3 reference frames × 3 head aggregations, per sequence | Table 6, Fig. 4 |
| `metrics_middle_frame_camel.csv`, `metrics_middle_frame_dog.csv`, `metrics_middle_frame_flamingo.csv` | 3 each | middle-reference-frame gate vs the first-frame gate at the video level | Table 7 |
| `metrics_rerun_camel.csv` | 4 | cross-machine re-run of the four arms for camel only (b0 / gate / centre / middle-frame), used as a consistency check; Table 7 reports the values in `metrics_middle_frame_camel.csv` | §4.4.7 |
| `metrics_control_camel.csv`, `metrics_control_dog.csv`, `metrics_control_flamingo.csv` | 6 each | control arms: B0, gate, centre, M5 no mask, M6 ground-truth mask, M3 boundary-refined mask | Table 8 |
| `runlog_b0.csv` | 16 | per-stage wall time for the B0 run (inversion / mask / object / style) | Table 1 |
| `runlog_b1.csv` | 18 | per-stage wall time for the B1 runs, all rc = 0 | Table 1 |
| `confidence_intervals.csv` | 8 | paired bootstrap confidence intervals (B = 20 000, seed 20261121) | Tables 2, 3 |
| `bootstrap_n8.json` | — | raw bootstrap output behind `confidence_intervals.csv` | Tables 2, 3 |
| `sensitivity_variants.csv` | 4 | the four protocol variants of Appendix A | Appendix A |
| `README.md` | — | this file | — |

## Conventions

- **ID-S** is the DINOv2 CLS cosine between the source and edited subject crops; higher means the subject is better preserved.
- **Sem-T** is the CLIP image–text cosine between the edited frame and the target prompt; ΔSem is reported relative to B0.
- **AUROC** is the Mann–Whitney statistic with tie averaging, comparing `g_id` at 60×90 against the downsampled first-frame subject mask.
- Rows sharing a `seq` / `video` key are per-timestamp; the paper always reports the mean over the scored frames (K = 13).
- Arm keys: `b0` external mask, `gate` attention gate, `center` centred Gaussian prior, `nomask` control M5, `gtmask` control M6, `refine` control M3.
- File names carry the quantity, not the machine or the experiment code; the mapping from the original run names is in the git history.

## What is deliberately not here

The rendered videos, `g_id` arrays, gate maps and per-run logs are 45.7 MB and are attached to release `v1.5.1` as `saig_results.tgz` (SHA256 `1cf07ae1…b9bc`). See `docs/DATA.md`.
