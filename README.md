# Where Should the Keep-Mask Come From?

Code and measurement tables for the study **“Where Should the Keep-Mask Come From? A Controlled Study of Mask Sources in Training-Free Video Editing”** (manuscript prepared for *The Visual Computer*).

The repository contains the analysis scripts, the per-sequence measurement tables, and a verification script that recomputes every headline number in the paper **without a GPU**. The large artefact bundle (39 output videos, `g_id` arrays, gate maps, per-run logs, 45.7 MB) is distributed separately — see [`docs/DATA.md`](docs/DATA.md).

---

## What the study asks

Training-free video editors such as FADE delegate the decision of *what must be preserved* to an external segmentation mask. We treat that mask source as the experimental variable and compare three sources under an otherwise identical pipeline (same prompts, input clip, DDIM inversion trajectory, step count, resolution):

| Arm | Mask source |
|---|---|
| **B0** | external text-referenced segmentation (the original FADE mask) |
| **B1-gate** | self-attention mass to the first latent frame, `mask ⊙ (1 − α·g_id)`, α = 0.5 |
| **B1-centre** | parameter-free centred Gaussian (sees neither video nor model) |

## Headline numbers

| Quantity | Value |
|---|---|
| AUROC(gate) vs subject mask, 8 DAVIS sequences | **0.617** [95% CI 0.583, 0.655] |
| AUROC(centred Gaussian) | **0.914** [0.871, 0.951] |
| Paired difference (gate − prior) | **−0.297** [−0.337, −0.252] |
| ID-S: B0 / gate / centre | **0.462 / 0.859 / 0.786** |
| gate − centre (per-sequence) | **+0.073** [0.035, 0.116], 7/8 sequences, sign test p = 0.035 |
| 108-cell sweep (layers × reference frames × head aggregations) | max **0.860**; no cell beats the prior |

The conclusion is scoped to CogVideoX-5b, the four patched layers ({2,3} kept), and DAVIS-style centred subjects.

## Verify the numbers in one command (no GPU)

```bash
python verify_numbers.py
```

It re-reads `data/` and asserts the means, ranges, row counts and confidence intervals used in the paper. Expected output ends with `ALL NUMBERS VERIFIED`.

## Reproduce the experiments (GPU required)

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md). Short version:

1. Build the environment (`requirements.txt`, CogVideoX-5b, DAVIS-2017 480p).
2. Apply the five patches with `scripts/apply_saig_patches.py` and `scripts/patch_guider_saig_gate.py`.
3. Run `scripts/auroc_gid.py` on the exported `g_id` arrays → expect `MEAN_AUC_GID 0.6171`, `GATE_AUROC FAIL`.
4. Run `scripts/metrics_compare.py` for the video-level arms.

> **Version note.** `scripts/metrics_compare.py` is the exact revision (2025-11-18) that produced the published numbers. A later internal revision changed the annotation-enumeration order and the crop list, which shifts ID-S by up to 0.02; the released tables were produced with the file shipped here.

## Repository layout

```
scripts/     analysis + experiment scripts (id export, AUROC, masks, gate injection, metrics, sweep, controls)
data/        measurement tables behind every number and figure (CSV/JSON)
figures/     paper figures (Fig. 1-4)
docs/        REPRODUCE.md (full pipeline) · DATA.md (artefact bundle)
verify_numbers.py   recomputes the paper numbers from data/
```

## Data and artefacts

All primary measurements are in `data/`. The artefact bundle with the 39 rendered videos, `g_id` arrays, gate maps and per-run logs is archived separately; its contents, size and SHA256 are listed in [`docs/DATA.md`](docs/DATA.md).

The study uses the publicly released **DAVIS 2017** and **YouTube-VOS** datasets under their original licences. Derived frames and videos are redistributed for research reproducibility only; please observe the dataset licences.

## Licence

Code: MIT (see `LICENSE`). The pipeline builds on FADE (MIT) and HybridGL; the text of the manuscript is not part of this repository.

## Contact

Corresponding author: *[to be filled]*
