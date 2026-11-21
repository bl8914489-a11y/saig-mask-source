# Reproducing the experiments

Everything below was run on Ubuntu 22.04 with an 80 GB GPU (A100 / A800 / H20, driver 595.71.05). The measured numbers reproduce exactly across those three machines.

## 0. Sanity check without a GPU

```bash
python verify_numbers.py          # expects: ALL NUMBERS VERIFIED
```

If this fails, the `data/` tables are inconsistent with the paper — stop here.

## 1. Environment

```bash
python3 -m venv /root/venv310 && . /root/venv310/bin/activate
pip install -r requirements.txt
pip install "en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl"
pip install segment-anything-py
```

Then fetch:

| Asset | Size | Where |
|---|---|---|
| CogVideoX-5b | 21 GB | `modelscope` → `ZhipuAI/CogVideoX-5b` (or Hugging Face `zai-org/CogVideoX-5b`) |
| DAVIS-2017 480p | 1.9 GB | Hugging Face dataset `rg4q33/DAVIS-2017-Unsupervised-trainval-480p` |
| SAM ViT-H weights | 2.4 GB | `ybelkada/segment-anything` → `checkpoints/sam_vit_h_4b8939.pth` |
| FADE + HybridGL | — | `git clone https://github.com/EternalEvan/FADE`, then HybridGL into `FADE/HybridGL` |

## 2. Apply the patches (five of them)

```bash
cd FADE
python <repo>/scripts/apply_saig_patches.py          # offload + rope + layer tagging + q/k instrumentation
python <repo>/scripts/patch_guider_saig_gate.py      # gate injection at the single mask consumption point
python -m py_compile edit.py diffusion_core/guiders/*.py
```

Expected: `offload 1 / tag 1 / rope 1 / gate 1 / sink 2`.

## 3. Export g_id and check the gate metric (CPU-only after export)

```bash
export SAIG_CAPTURE=1
cd FADE
python <repo>/scripts/export_gid_rev1.py --video input/camel.mp4 --out saig_out/camel_gid.npy
unset SAIG_CAPTURE

python <repo>/scripts/auroc_gid.py --gid_dir saig_out --ann DAVIS/Annotations/480p --out auroc.csv
```

Expected: `MEAN_AUC_GID 0.6171`, `MEAN_MARGIN -0.2973`, `GATE_AUROC FAIL`.
If you get a different value, check the pitfalls in the next section before anything else.

## 4. Video-level arms

```bash
python <repo>/scripts/metrics_compare.py --seq camel --src input/camel.mp4 \
  --ann DAVIS/Annotations/480p/camel --init "a camel" --target "a horse" \
  --videos b0=output/b0_camel.mp4 gate=output/b1_camel.mp4 center=output/b1c_camel.mp4 \
  --out metrics_camel.csv
```

Expected for camel: `b0 0.3919 / gate 0.9195 / centre 0.8184`.

## 5. Six pitfalls that cost us days

1. **Rotary embedding frame count.** The FADE helper is called with `3` latent frames; at 13 latents the sequence becomes 17,776 tokens and you must pass the actual latent count (`self.inv_latents[0].shape[1]`), otherwise you get a 17,550 vs 4,050 mismatch.
2. **`pipe.enable_model_cpu_offload()` is mandatory** — without it the peak is 78.90 GiB and you OOM on an 80 GB card.
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is mandatory** — otherwise the frequency branch OOMs at 78.96 GiB.
4. **The gate must be injected as a float multiply.** Casting the mask with `.bool()` binarises the soft weights and the gate silently does nothing.
5. **Attention capture must instrument both `patched_forward` sites.** With one site you get `CAPTURED=0`.
6. **DAVIS annotations are palette images.** Reading them with `convert('L')` maps index 1 to grey 38, so any `>127` threshold yields an all-background mask. Read raw indices instead.

## 6. Determinism

Same GPU model + same torch build → identical numbers (we verified `0.6171` on A100, H20 and A800). Different `torch` builds can shift ID-S by ~1e-3 through bf16 kernels; the reported differences (≥0.02) are well above that.
