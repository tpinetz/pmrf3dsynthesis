#!/usr/bin/env python3
"""
Controlled ablation experiment comparing 5 through-plane noise couplings:
  1. normal (Independent noise)
  2. constant (Shared noise)
  3. linear_vm (Variance-matched linear mixing)
  4. slerp (Spherical Linear Interpolation)
  5. linear (Plain linear mixing)

All arms use:
  - The exact same initial estimator (MMSE: 2026-01-23_11-33-50_ms_mmse_added_augs_transupnet)
  - The exact same 2D flow checkpoint (2026-02-20_09-09-18_ms__rflow_cleaned_song_sigma_01)
  - The exact same training conditions (same weights, sigma_start=0.1 from model)
  - Paired randomness / Common Random Numbers (CRN) with matched endpoints (z0, z1)
"""

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from skimage.transform import resize

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from pmrf3dsynthesis.data import constants as C
from pmrf3dsynthesis.models.flow_model import FlowModel
from pmrf3dsynthesis.models.mmse_model import MMSEModel
from pmrf3dsynthesis.models.slice_noise import draw_slice_endpoints, make_slice_noise
from pmrf3dsynthesis.evaluation.ssim import SSIM3D
from pmrf3dsynthesis.evaluation.retfound_fid import RETFoundFeatureExtractor
from pmrf3dsynthesis.evaluation.interslice import (
    evaluate_interslice_structure,
    compute_lag_ssim_comparison,
    compute_spectral_metrics,
    compute_orthogonal_reslice_metrics,
    compute_flow_displacement
)
from pmrf3dsynthesis.vessel_segmentation.utils import load_model as load_vessel_model, extract_vessels_oct
from pmrf3dsynthesis.eval_model import (
    load_sample,
    compute_sdif,
    compute_vessel_metrics,
    compute_volume_lpips,
    setup_fid_metric
)
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def integrate_2d_flow_batched(
    network: torch.nn.Module,
    init_vol: torch.Tensor,
    T: int = 20,
    t_eps: float = 0.05,
    slice_batch_size: int = 32
) -> torch.Tensor:
    """
    Integrates 2D rectified flow ODE across all slices in parallel mini-batches.
    Since SongUNet uses GroupNorm (not BatchNorm), mini-batching slices produces
    mathematically exact results with ~30x faster inference.
    
    Args:
        network: 2D SongUNet model (x, t, class_labels)
        init_vol: (1, 1, D, H, W) initial noisy volume
        T: Number of Euler integration steps
        t_eps: Minimum step clamping factor
        slice_batch_size: Mini-batch size of slices
    
    Returns:
        (1, 1, D, H, W) synthesized volume
    """
    device = init_vol.device
    D, H, W = init_vol.shape[2], init_vol.shape[3], init_vol.shape[4]
    
    # Reshape (1, 1, D, H, W) -> (D, 1, H, W)
    slices = init_vol[0].permute(1, 0, 2, 3).clone()
    
    for s_start in range(0, D, slice_batch_size):
        s_end = min(s_start + slice_batch_size, D)
        b_slices = slices[s_start:s_end].clone()
        B_cur = b_slices.shape[0]

        for i in range(T):
            t_val = i / float(T)
            t_tensor = torch.full((B_cur,), t_val, dtype=torch.float32, device=device)
            c_tensor = torch.zeros((B_cur,), dtype=torch.int, device=device)

            pred = network(b_slices, t_tensor, c_tensor)
            vel = (pred - b_slices) / max(1.0 - t_val, t_eps)
            b_slices += (1.0 / float(T)) * vel

        slices[s_start:s_end] = b_slices

    # Reshape back (D, 1, H, W) -> (1, 1, D, H, W)
    return slices.permute(1, 0, 2, 3).unsqueeze(0)


def run_ablation(args):
    device = args.device
    if device != 'cpu' and torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    logging.info(f"Using device: {device}")
    out_dir = Path(args.res_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load 2D Flow Model
    logging.info(f"Loading FlowModel from: {args.flow_ckpt}")
    flow_model = FlowModel.load_from_checkpoint(args.flow_ckpt, map_location=device)
    flow_model = flow_model.to(device).eval()
    sigma_start = flow_model.sigma_start
    t_eps = flow_model.t_eps
    network_2d = flow_model.network
    logging.info(f"FlowModel loaded: sigma_start={sigma_start}, t_eps={t_eps}")

    # 2. Setup MMSE guidance (check cache first)
    mmse_model = None
    mmse_cache_dir = Path(args.mmse_cached_dir) if args.mmse_cached_dir else None
    if mmse_cache_dir and mmse_cache_dir.exists():
        logging.info(f"Using pre-cached MMSE predictions from: {mmse_cache_dir}")
    else:
        logging.info(f"Loading MMSE model from: {args.mmse_ckpt}")
        mmse_model = MMSEModel.load_from_checkpoint(args.mmse_ckpt, map_location=device)
        mmse_model = mmse_model.to(device).eval()

    # 3. Load dataset keys
    keys_df = pd.read_csv(args.df)
    cand_cols = ['Keys', 'keys', 'key', 'patient_id']
    match_col = next((c for c in cand_cols if c in keys_df.columns), keys_df.columns[0])
    keys = list(keys_df[match_col].astype(str).str.strip())
    if args.max_cases is not None:
        keys = keys[:args.max_cases]
    logging.info(f"Total evaluation cases: {len(keys)}")

    # 4. Initialize evaluation metric modules
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = SSIM3D().to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval()
    ves_model = load_vessel_model().to(device).eval() if args.eval_vessels else None

    # Single shared RETFound feature extractor for FID (avoids duplicating 5+ ViT backbones on GPU)
    feature_extractor = None
    if not args.no_fid:
        logging.info("Initializing single shared RETFound feature extractor for OCT FID...")
        feature_extractor = RETFoundFeatureExtractor().to(device).eval()

    arms = args.arms
    seeds = args.seeds
    logging.info(f"Ablation arms: {arms}")
    logging.info(f"Random seeds: {seeds}")

    # Per-arm, per-seed results accumulator: arm -> seed -> metric_dict
    all_results = {arm: {s: {"PSNR": {}, "SSIM": {}, "LPIPS": {}, "SDIF": {}, "Vessel_Dice": {},
                             "Adjacent_SSIM_Gap": {}, "Log_Spectral_Distance": {},
                             "Orthogonal_SSIM_mean": {}, "Orthogonal_LPIPS_mean": {},
                             "Flow_Displacement_L1": {}, "FID": []} for s in seeds} for arm in arms}

    # Feature collectors for FID
    gt_features_list = []
    arm_features = {arm: {s: [] for s in seeds} for arm in arms}

    vol_path = Path(args.vol_path) if args.vol_path else None
    img_path = Path(args.img_path) if args.img_path else None

    start_time = time.time()

    # Evaluation loop
    for case_idx, key in enumerate(tqdm(keys, desc="Processing cases")):
        # Load input fundus and GT volume
        in_img, gt_vol = load_sample(
            key=key,
            modality="oct",
            vol_path=vol_path,
            img_path=img_path
        )
        gt_t = torch.from_numpy(gt_vol[None, None]).float().to(device)
        gt_01 = torch.clamp(gt_t, 0.0, 1.0)
        gt_np_01 = gt_01.cpu().numpy()[0, 0]

        # Get initial MMSE estimate
        cached_file = mmse_cache_dir / f"{key}.npy" if mmse_cache_dir else None
        if cached_file and cached_file.exists():
            mmse_raw = np.load(cached_file)
            if mmse_raw.shape != (128, 256, 256):
                mmse_raw = resize(mmse_raw, (128, 256, 256))
            x_mmse = torch.from_numpy(mmse_raw[None, None]).float().to(device)
        else:
            in_t = torch.from_numpy(in_img.transpose((2, 0, 1))[None] * 2.0 - 1.0).float().to(device)
            with torch.no_grad():
                x_mmse = mmse_model.predict(in_t)
                if isinstance(x_mmse, list):
                    x_mmse = x_mmse[-1]

        # Standardize MMSE volume range to [-1, 1] for flow integration
        if x_mmse.min() >= 0.0 and x_mmse.max() <= 1.0:
            x_mmse_flow = x_mmse * 2.0 - 1.0
        else:
            x_mmse_flow = x_mmse

        x_shape = x_mmse_flow.shape  # (1, 1, 128, 256, 256)

        # Iterate over random seeds
        for s_idx, seed_val in enumerate(seeds):
            # Seed generator specifically for this (case, seed)
            gen_seed = int(seed_val + case_idx * 1000)
            g = torch.Generator(device=device).manual_seed(gen_seed)

            # Common Random Numbers: draw endpoints z0, z1
            z0, z1 = draw_slice_endpoints(x_shape, generator=g, device=device, dtype=x_mmse_flow.dtype)

            # Generate noise for all arms using the exact same endpoints
            arm_noises = {}
            for arm in arms:
                arm_noises[arm] = make_slice_noise(
                    noise_type=arm,
                    shape=x_shape,
                    generator=g,
                    z0=z0,
                    z1=z1,
                    device=device,
                    dtype=x_mmse_flow.dtype
                )

            # Sanity check: verify endpoint identity across arms (slice 0 and slice D-1)
            if args.verify_endpoints:
                for comp_arm in [C.NOISE_TYPE_LINEAR, C.NOISE_TYPE_LINEAR_VM, C.NOISE_TYPE_SLERP]:
                    if comp_arm in arm_noises and C.NOISE_TYPE_NORMAL in arm_noises:
                        diff_0 = (arm_noises[comp_arm][:, :, 0] - arm_noises[C.NOISE_TYPE_NORMAL][:, :, 0]).abs().max().item()
                        diff_end = (arm_noises[comp_arm][:, :, -1] - arm_noises[C.NOISE_TYPE_NORMAL][:, :, -1]).abs().max().item()
                        assert diff_0 < 1e-5, f"Endpoint 0 mismatch between {comp_arm} and normal: {diff_0}"
                        assert diff_end < 1e-5, f"Endpoint D-1 mismatch between {comp_arm} and normal: {diff_end}"

            # Evaluate each arm
            for arm in arms:
                noise = arm_noises[arm]
                init_noisy = x_mmse_flow + sigma_start * noise

                with torch.no_grad():
                    y_hat = integrate_2d_flow_batched(
                        network=network_2d,
                        init_vol=init_noisy,
                        T=args.T,
                        t_eps=t_eps,
                        slice_batch_size=args.slice_batch_size
                    )

                # Rescale from [-1, 1] to [0, 1]
                pred_01 = torch.clamp(y_hat * 0.5 + 0.5, 0.0, 1.0)
                pred_np_01 = pred_01.cpu().numpy()[0, 0]

                # Compute standard in-plane metrics
                psnr_val = float(psnr_metric(pred_01, gt_01).item())
                ssim_val = float(ssim_metric(pred_01, gt_01).item())
                lpips_val = float(compute_volume_lpips(pred_np_01, gt_np_01, lpips_fn, device=device))
                sdif_val = float(compute_sdif(pred_np_01, gt_np_01)["SDIF"])

                all_results[arm][seed_val]["PSNR"][key] = psnr_val
                all_results[arm][seed_val]["SSIM"][key] = ssim_val
                all_results[arm][seed_val]["LPIPS"][key] = lpips_val
                all_results[arm][seed_val]["SDIF"][key] = sdif_val

                # Vessel segmentation metrics (OCT)
                if ves_model is not None:
                    p_ves = extract_vessels_oct(pred_np_01, ves_model, device=device)
                    g_ves = extract_vessels_oct(gt_np_01, ves_model, device=device)
                    v_mets = compute_vessel_metrics(p_ves, g_ves)
                    all_results[arm][seed_val]["Vessel_Dice"][key] = v_mets["Vessel_Dice"]

                # Inter-slice structural metrics
                mmse_np_01 = np.clip(x_mmse.cpu().numpy()[0, 0], 0.0, 1.0)
                is_metrics = evaluate_interslice_structure(
                    pred=pred_np_01,
                    gt=gt_np_01,
                    mmse_pred=mmse_np_01,
                    lpips_fn=lpips_fn,
                    device=device
                )

                all_results[arm][seed_val]["Adjacent_SSIM_Gap"][key] = is_metrics["Adjacent_SSIM_Gap"]
                all_results[arm][seed_val]["Log_Spectral_Distance"][key] = is_metrics["Log_Spectral_Distance"]
                all_results[arm][seed_val]["Orthogonal_SSIM_mean"][key] = is_metrics["Orthogonal_SSIM_mean"]
                all_results[arm][seed_val]["Orthogonal_LPIPS_mean"][key] = is_metrics["Orthogonal_LPIPS_mean"]
                all_results[arm][seed_val]["Flow_Displacement_L1"][key] = is_metrics["Flow_Displacement_L1"]

                # Store additional lag-k SSIMs
                for k in (1, 2, 4, 8, 16):
                    k_name = f"LagSSIM_Gap_k{k}"
                    if k_name in is_metrics:
                        if k_name not in all_results[arm][seed_val]:
                            all_results[arm][seed_val][k_name] = {}
                        all_results[arm][seed_val][k_name][key] = is_metrics[k_name]

                # Collect features for FID
                if feature_extractor is not None:
                    p_slices = pred_01[0].transpose(1, 0)
                    with torch.no_grad():
                        p_feats = []
                        for b_i in range(0, p_slices.shape[0], 32):
                            p_feats.append(feature_extractor(p_slices[b_i:b_i + 32]).cpu().numpy())
                        arm_features[arm][seed_val].append(np.concatenate(p_feats, axis=0))

                # Save predicted volume if requested
                if args.save_volumes:
                    arm_save_dir = out_dir / arm / f"seed_{seed_val}"
                    arm_save_dir.mkdir(parents=True, exist_ok=True)
                    np.save(arm_save_dir / f"{key}.npy", pred_np_01)

        # Extract GT features once per case
        if feature_extractor is not None:
            g_slices = gt_01[0].transpose(1, 0)
            with torch.no_grad():
                g_feats = []
                for b_i in range(0, g_slices.shape[0], 32):
                    g_feats.append(feature_extractor(g_slices[b_i:b_i + 32]).cpu().numpy())
                gt_features_list.append(np.concatenate(g_feats, axis=0))

    # Compute FID for each arm and seed
    if feature_extractor is not None and gt_features_list:
        import scipy.linalg
        all_gt_feats = np.concatenate(gt_features_list, axis=0)
        mu_gt = np.mean(all_gt_feats, axis=0)
        sigma_gt = np.atleast_2d(np.cov(all_gt_feats, rowvar=False))

        for arm in arms:
            for seed_val in seeds:
                if arm_features[arm][seed_val]:
                    all_pred_feats = np.concatenate(arm_features[arm][seed_val], axis=0)
                    mu_pred = np.mean(all_pred_feats, axis=0)
                    sigma_pred = np.atleast_2d(np.cov(all_pred_feats, rowvar=False))

                    diff = mu_pred - mu_gt
                    covmean, _ = scipy.linalg.sqrtm(sigma_pred.dot(sigma_gt), disp=False)
                    if np.iscomplexobj(covmean):
                        covmean = covmean.real
                    fid_score = float(diff.dot(diff) + np.trace(sigma_pred) + np.trace(sigma_gt) - 2.0 * np.trace(covmean))
                    all_results[arm][seed_val]["FID"] = [fid_score]
                    logging.info(f"Arm '{arm}' Seed {seed_val}: FID = {fid_score:.4f}")

    # Save all raw results
    results_file = out_dir / "ablation_results_raw.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(all_results, f)
    logging.info(f"Saved raw ablation results to: {results_file}")

    elapsed = time.time() - start_time
    logging.info(f"Ablation run complete in {elapsed / 60.0:.2f} minutes.")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run noise coupling ablation for 2D rectified flow.")
    parser.add_argument(
        "--flow_ckpt",
        type=str,
        default="/home/optima/tpinetz/data/results/iscreen/gen/2026-02-20_09-09-18_ms__rflow_cleaned_song_sigma_01/version_0/checkpoints/last.ckpt",
        help="Path to FlowModel checkpoint."
    )
    parser.add_argument(
        "--mmse_ckpt",
        type=str,
        default="/home/optima/tpinetz/data/results/iscreen/gen/2026-01-23_11-33-50_ms_mmse_added_augs_transupnet/version_0/checkpoints/last.ckpt",
        help="Path to MMSE checkpoint."
    )
    parser.add_argument(
        "--mmse_cached_dir",
        type=str,
        default="/media/thomas-pinetz/data/data/OCT/results/mmse/2026-01-23_11-33-50_ms_mmse_added_augs_transupnet",
        help="Path to precomputed MMSE volumes directory."
    )
    parser.add_argument(
        "--df",
        type=str,
        default="/home/thomas-pinetz/repos/eye-screen/data_splits/fund_mip_251211/test_keys.csv",
        help="Path to test keys csv."
    )
    parser.add_argument(
        "--vol_path",
        type=str,
        default="/home/optima/tpinetz/data/OCT/retinsight/FundusMIP/flattened_volumes",
        help="OCT volumes ground truth directory."
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="/home/optima/tpinetz/data/OCT/retinsight/FundusMIP/Fundus_cropped_color",
        help="OCT fundus images directory."
    )
    parser.add_argument(
        "--res_path",
        type=str,
        default="/media/thomas-pinetz/data/data/OCT/results/noise_coupling_ablation_2026-02-20_09-09-18",
        help="Directory to save ablation evaluation outputs."
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[C.NOISE_TYPE_NORMAL, C.NOISE_TYPE_CONSTANT, C.NOISE_TYPE_LINEAR_VM, C.NOISE_TYPE_SLERP, C.NOISE_TYPE_LINEAR],
        help="Noise coupling arms to evaluate."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1042, 2042], help="Random seeds.")
    parser.add_argument("--T", type=int, default=20, help="Euler flow steps.")
    parser.add_argument("--slice_batch_size", type=int, default=16, help="Mini-batch size of slices.")
    parser.add_argument("--max_cases", type=int, default=None, help="Optional max cases to evaluate (e.g. 2 for smoke test).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_vessels", action="store_true", default=True)
    parser.add_argument("--no_fid", action="store_true", default=False, help="Disable FID computation.")
    parser.add_argument("--save_volumes", action="store_true", default=True)
    parser.add_argument("--verify_endpoints", action="store_true", default=True)

    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
