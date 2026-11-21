# Scripts

Index of every script in this directory. The five entry points that the paper and `docs/REPRODUCE.md` name explicitly are kept at this level; everything else is grouped by role.

## Entry points (named in the paper)

| Script | What it does | Paper artefact |
|---|---|---|
| `apply_saig_patches.py` | Patches FADE: CPU offload, rotary frame count, layer tagging, q/k instrumentation | §3.4, pitfalls 1–2 and 5 |
| `patch_guider_saig_gate.py` | Injects `mask ⊙ (1 − α·g_id)` at the single mask-consumption point | §3.2, Eq. 3 |
| `export_gid.py` | Exports `g_id` from the captured attention | §3.3, Fig. 1 |
| `auroc_gid.py` | AUROC of `g_id` against the subject mask, plus the three zero models | Table 2, Appendix A |
| `metrics_compare.py` | ID-S / Sem-T for the video arms — **the revision that produced the published tables (2025-11-18)** | Tables 3, 5, 7, 8 |

## `pipeline/` — patches, capture and mask construction

| Script | Purpose |
|---|---|
| `patch_fade_saig.py` | applies the FADE patch set (revision used for the reported runs) |
| `capture_qk.py` | q/k capture under the `SAIG_CAPTURE` guard |
| `probe_patched_layers.py` | finds which blocks actually use the patched forward — this is how the "4 of 42 layers" limitation was discovered |
| `inspect_gid.py` | summary statistics of an exported `g_id` array |
| `build_gate_masks.py` | builds `g_id↑` and the gate mask at 60×90 |
| `build_middle_frame_gid.py` | rebuilds the gate from the middle latent frame (§4.4.7) |
| `build_control_masks.py` | builds the M5 (no mask), M6 (ground-truth) and M3 (boundary-refined) control masks (§4.4.8) |
| `ab_test_offload.py` | A/B test of CPU offload (79 GiB peak vs OOM) |
| `legacy/export_gid_rev1.py`, `legacy/export_gid_rev2.py` | superseded revisions of the exporter, kept for provenance |
| `legacy/patch_fade_saig_rev1.py` | first revision of the patch script |

## `analysis/` — post-hoc computation on the exported data

| Script | Purpose | Paper artefact |
|---|---|---|
| `sweep_agg.py` | aggregates the 108-cell sweep (4 layers × 3 reference frames × 3 head aggregations) | Table 6, Fig. 4 |
| `ci_table.py` | paired bootstrap confidence intervals (B = 20 000, seed 20261121) | Tables 2, 3 |
| `make_tables.py` | regenerates the paper tables from `data/` | Tables 2–8 |
| `pixel_compare.py` | MAE / SSIM / difference maps between arms | Table 4 |
| `metrics_single_sequence.py` | single-sequence metric run, used before the batch version | — |
| `check_manuscript.py` | 22-point consistency check of the manuscript against `data/` | Appendix C |
| `legacy/pixel_compare_rev1.py` | first revision of the pixel comparison | — |

## `figures/`

| Script | Purpose | Paper artefact |
|---|---|---|
| `make_paper_figures.py` | draws Fig. 1–4 with PIL (no matplotlib dependency) | Fig. 1–4 |

## `ops/` — run orchestration and environment

The actual shell scripts used on the rented GPUs. They contain machine paths, instance names and cost guards, so they are included for transparency only — you do not need them to reproduce anything.

| Group | Scripts |
|---|---|
| environment | `setup_env_part1.sh`, `setup_env_part2.sh`, `rebuild_env_step1.sh`, `rebuild_env_step2.sh`, `rebuild_env_step3_resume.sh`, `download_assets.sh` |
| pre-flight | `check_env.py`, `check_gpu.sh`, `precheck.sh`, `smoke_test.sh`, `verify_downloads.py` |
| run chains | `run_main_arms.sh`, `run_control_arms.sh`, `run_alpha_sweep.sh`, `run_middle_frame_arm.sh`, `run_night_chain.sh`, `run_all_night.sh` |
| guard / bookkeeping | `budget_guard.sh`, `make_output_manifest.py` |

## Naming conventions

- No version suffixes: the revision that produced the reported numbers carries the plain name, superseded revisions live under `legacy/` with a `_revN` suffix.
- No machine names or internal experiment codes in file names. The mapping from an old name to the current one is in the git history of this file.
- `data/` follows the same rule; each table is named after the quantity it holds (see `data/README.md`).
