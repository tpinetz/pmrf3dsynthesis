# pmrf3dsynthesis

Inference and evaluation code for PRFM, which synthesises 3D volumes from a single 2D image in two stages:

1. **MMSE stage.** A TransUpNet predicts the posterior-mean (MMSE) volume from the 2D input.
2. **Rectified flow.** A 2D slice-wise rectified flow, conditioned on the MMSE estimate, restores texture and detail.
   Through-plane noise coupling (`--noise_type`) controls slice-to-slice consistency.

Two tasks are supported:

| Modality | Input | Output | Test set |
|---|---|---|---|
| `oct` | colour fundus photograph | 3D macular OCT volume (128 × 256 × 256) | 81 eyes |
| `ct` | synthetic X-ray (DRR) | 3D chest CT volume | 100 LIDC-IDRI cases |

The package also contains fundus-to-OCT-MIP registration (to build paired training data), retinal vessel segmentation, and the evaluation metrics used in the paper.
Models are trained with the companion repository [eye-screen](https://github.com/tpinetz/eye-screen) (`scripts/train.py`). This repository loads those Lightning checkpoints.

## Installation

```sh
git clone https://github.com/tpinetz/pmrf3dsynthesis.git
cd pmrf3dsynthesis
pip install -e .
```

Dependencies are listed in `requirements.txt` and include PyTorch 2.9 and Lightning 2.6. A CUDA GPU is strongly recommended.
Optional extras for some evaluations:

- RadImageNet ResNet50 weights for CT FID (`--radimagenet_path`).
- RETFound weights for OCT FID, downloaded via `transformers`.
- `lungmask` / `TotalSegmentator` for the CT organ volume scripts.
- `eye-screen` on `PYTHONPATH` (or `EYE_SCREEN_ROOT`) only for the Stable EDM baseline script.

## Command-line tools

`pip install -e .` installs four commands. Run any of them with `--help` for the full option list.

### `run-registration`: fundus ↔ OCT-MIP registration

```sh
run-registration <fundus_dir> <oct_dir> <out_dir> [--common_size N]
```

### `eval-model`: evaluate a model on a test split

```sh
# MMSE stage only
eval-model --model_name mmse_model --ckpt_path mmse.ckpt \
    --df test_keys.csv --res_path results/mmse --modality oct --device cuda --eval_vessels

# Full PRFM (MMSE + rectified flow, 20 steps)
eval-model --model_name flow_model --ckpt_path flow.ckpt --mmse_path mmse.ckpt \
    --df test_keys.csv --res_path results/prfm --modality oct --T 20 --device cuda
```

- Metrics: PSNR, 3D-SSIM, LPIPS, SDIF (slice difference), FID, plus vessel Dice/IoU for OCT.
- FID features: RETFound for OCT, RadImageNet for CT, or Inception (`--fid_feature`).
- CT volumes are clipped to [−1350, 150] HU and scaled to [0, 1] (`--hu_min`, `--hu_max`).
- Noise couplings (`--noise_type`, default `slerp` as in the paper): `normal` (independent), `constant` (shared), `linear`, `linear_vm` (variance-matched linear), `slerp`.
- Results are written to `--res_path` as `eval_results.pkl` / `eval_results.csv` and per-case volumes.

### `eval-model-only-fundus`: synthesise OCT from fundus images alone

This tool detects the fovea, crops the fundus image and synthesises the volume, so no OCT input is needed.

```sh
eval-model-only-fundus --fundus_path fundus.png --res_path out/ \
    --ckpt_path flow.ckpt --model_name flow --mmse_path mmse.ckpt --T 20 --save_mip
```

Use `--crop_only` to export the fovea-centred crops only. Pass `--gt_vol_dir` to score the results against ground-truth volumes.

### `view-eval`: browse evaluation results

```sh
view-eval --res_path results/prfm --modality oct --sort_by psnr_asc
```

## Scripts for paper experiments

| Script | Purpose |
|---|---|
| `scripts/run_noise_coupling_ablation.py`, `scripts/analyze_noise_coupling.py` | Noise-coupling ablation over seeds, with Wilcoxon/Holm tests and bootstrap CIs |
| `scripts/eval_stable_edm_oct.py` | Stable EDM (WF-VAE latent) baseline on OCT. Use `--num-steps 100 --seed 42` for the 100-step run |
| `scripts/lung_volume_eval.py`, `scripts/lung_volume_summary.py` | Lung/heart volume error on generated CT (`--segmenter threshold\|lungmask\|totalseg`) |
| `scripts/mask_flicker_eval.py` | Slice-to-slice lung-mask boundary flicker ("pop rate") |
| `scripts/recompute_all_metrics.py`, `scripts/recompute_oct_vessels.py`, `scripts/extract_metrics.py` | Recompute and collect metrics from saved result folders |
| `scripts/generate_pngs_for_rflow.py` | Export MMSE predictions as PNG slices for rectified-flow training |
| `scripts/generate_comparison_gifs.py` | GT / MMSE / PRFM comparison GIFs, written to `assets/` |
| `scripts/plot_vessel_architecture.py` | Vessel network architecture figure |

Several scripts default to paths on the authors' machines. Pass your own paths through their CLI arguments.

## Repository layout

```
pmrf3dsynthesis/
  cli_fundus_oct.py        run-registration entry point
  eval_model.py            eval-model entry point (OCT and CT)
  eval_model_only_fundus.py
  visualize_eval.py        view-eval entry point
  models/                  MMSE (TransUpNet/TransUNet), flow model, EDM networks (2D, 3D-conditioned), slice noise
  evaluation/              FID (RETFound, RadImageNet), SSIM, inter-slice and Wasserstein metrics
  registration/            fundus ↔ MIP registration
  vessel_segmentation/     pretrained retinal vessel segmenter (model.ckpt)
  data/, utils/
scripts/                   experiment and figure scripts (see above)
tests/                     pytest unit tests
```

## Tests

```sh
pip install pytest
pytest tests
```

## License

MIT. See `LICENSE`.
