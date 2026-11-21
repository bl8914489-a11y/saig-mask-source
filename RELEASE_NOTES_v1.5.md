v1.5.1 - code, measurement tables and artefact bundle

Release accompanying the manuscript "Where Should the Keep-Mask Come From? A Controlled Study of Mask Sources in Training-Free Video Editing" (prepared for The Visual Computer).

ASSET

saig_results.tgz  ·  45.7 MB (45,651,488 bytes)
SHA256: 1cf07ae1221bf72675555c85f14ad55167156431fe3d264545b9a6ed77e5b9bc

Contents: 39 rendered videos (b0_*, b1_*, b1c_*, b1m_*, alpha sweep, m5_nomask_*, m6_gtmask_*, m3_refine_*), 8 g_id arrays, 9 gate maps at 60x90, and the per-run logs.

VERIFY WITHOUT A GPU

    git clone https://github.com/bl8914489-a11y/saig-mask-source
    cd saig-mask-source
    python verify_numbers.py

Expected output: ALL NUMBERS VERIFIED

REPRODUCE THE EXPERIMENTS (GPU)

See docs/REPRODUCE.md. Expected after a correct build: MEAN_AUC_GID 0.6171, MEAN_MARGIN -0.2973, GATE_AUROC FAIL.

NOTES

- scripts/metrics_compare.py is the exact revision (2025-11-18) that produced the published tables; later internal revisions shift ID-S by up to 0.02.
- The study uses DAVIS 2017 and YouTube-VOS under their original licences; derived videos are redistributed for reproducibility only.
