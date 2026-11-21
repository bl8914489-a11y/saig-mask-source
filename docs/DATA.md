# Artefact bundle (distributed separately)

The repository ships code and measurement tables only. The heavy artefacts live in one archive:

| Item | Value |
|---|---|
| File | `saig_results_20261121.tgz` |
| Size | 45.7 MB |
| SHA256 | `1cf07ae1221bf72675555c85f14ad55167156431fe3d264545b9a6ed77e5b9bc` |
| Contents | 39 rendered videos, 17 `.npy` (g_id arrays and gate maps), 13 `.csv`, 39 per-run logs |

## What is inside

```
output/     39 videos
            b0_<seq>.mp4                 B0 arm (external mask), 8 sequences
            b1_<seq>.mp4                 B1-gate arm
            b1c_<seq>.mp4                B1-centre arm
            b1m_<seq>.mp4                middle-frame gate (3 sequences)
            b1a0.3_<seq>.mp4, b1a0.7_<seq>.mp4   alpha sweep (3 sequences)
            m5_nomask_<seq>.mp4          control arm: no mask
            m6_gtmask_<seq>.mp4          control arm: ground-truth mask
            m3_refine_<seq>.mp4          control arm: boundary-refined mask
saig_out/   8 x *_gid.npy       per-frame attention mass, (13, 1350)
masks/      9 x gid_*.npy       gate maps at 60x90 (+ gid_center.npy, gid_mid_*.npy)
logs/       runlog_b0.csv, runlog_b1.csv, per-run logs, AUROC and metric tables
```

## Where to put it

Two options, both acceptable for the Data Availability statement:

**Option A — GitHub release (fastest).** Create a release tagged `v1.5` and attach the `.tgz`. The statement then reads:

> The artefact bundle (45.7 MB, SHA256 `1cf07ae1…`) is attached to release v1.5 of the repository at https://github.com/<user>/<repo>, together with the analysis scripts.

**Option B — Zenodo DOI (recommended for a journal).** Upload the `.tgz` (plus this repository as a zip) to Zenodo, get a DOI, and cite it:

> The artefact bundle and code are archived at https://doi.org/10.5281/zenodo.<id> (45.7 MB, SHA256 `1cf07ae1…`).

Whichever option you choose, keep the SHA256 in the statement — it is what lets a reader confirm they have the same files the paper used.

## Licence note

The bundle contains frames and videos derived from DAVIS 2017. DAVIS is released for research use; redistributing derived videos for reproducibility is customary, but the dataset licence still applies to any reuse of the underlying footage. Do not re-license the videos.
