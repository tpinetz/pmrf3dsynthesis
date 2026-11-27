#!/usr/bin/env python3
"""
Compute full-dataset RadImageNet FID on the saved 3D CT ablation outputs.

Extracts 2048-dimensional RadImageNet ResNet50 features across all 100 cases
(all slices) for ground truth and each of the 5 ablation arms:
  - normal (Independent Gaussian noise)
  - constant (Shared noise)
  - linear_vm (Variance-matched linear mixing)
  - slerp (Spherical linear interpolation)
  - linear (Plain linear mixing)

Updates ablation_results_raw.pkl with the full-dataset FID scores and regenerates
the analysis summary tables and report.
"""

import os
import sys
import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import scipy.linalg
from tqdm import tqdm

from pmrf3dsynthesis.evaluation.radimagenet_fid import RadImageNetFeatureExtractor
from pmrf3dsynthesis.eval_model import load_sample
from pmrf3dsynthesis.data import constants as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        logging.warning("FID calculation produced non-finite values; adding epsilon regularization.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def extract_volume_features(vol_np, feature_extractor, device='cuda', batch_size=32):
    """
    Extracts RadImageNet features for all 2D slices in a 3D volume.
    vol_np: (D, H, W) in [0, 1]
    Returns: (D, 2048) float32 numpy array
    """
    D = vol_np.shape[0]
    # (D, H, W) -> (D, 1, H, W)
    t_vol = torch.from_numpy(vol_np[:, None, :, :]).float().to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, D, batch_size):
            chunk = t_vol[i:i + batch_size]
            f = feature_extractor(chunk)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Paths
    split_csv = Path("/home/thomas-pinetz/repos/eye-screen/data_splits/LIDC_IDRI/val_keys.csv")
    gt_vol_dir = Path("/home/optima/tpinetz/data/CT/LIDC-IDRI/CT")
    gt_img_dir = Path("/home/optima/tpinetz/data/CT/LIDC-IDRI/synth_xray")
    results_dir = Path("/media/thomas-pinetz/data/data/CT/results/noise_coupling_ablation_full")
    radimagenet_weights = Path("/home/thomas-pinetz/tools/RadImageNet/RadImageNet_pytorch/ResNet50.pt")
    raw_pkl_path = results_dir / "ablation_results_raw.pkl"

    # Load keys
    df = pd.read_csv(split_csv)
    keys = list(df['Keys'].astype(str).str.strip())
    logging.info(f"Evaluating {len(keys)} test cases for RadImageNet FID.")

    # Initialize RadImageNet feature extractor
    logging.info("Initializing RadImageNet ResNet50 feature extractor...")
    feature_extractor = RadImageNetFeatureExtractor(checkpoint_path=radimagenet_weights).to(device).eval()

    # 1. Ground truth feature extraction (with disk cache)
    gt_cache_file = results_dir / "gt_radimagenet_features.npy"
    if gt_cache_file.exists():
        logging.info(f"Loading cached ground truth features from: {gt_cache_file}")
        gt_features = np.load(gt_cache_file)
    else:
        logging.info("Extracting ground truth features across all cases...")
        gt_feats_list = []
        for key in tqdm(keys, desc="Ground Truth Volumes"):
            _, gt_vol = load_sample(
                key=key,
                modality=C.MODALITY_CT,
                vol_path=gt_vol_dir,
                img_path=gt_img_dir
            )
            feats = extract_volume_features(gt_vol, feature_extractor, device=device)
            gt_feats_list.append(feats)
        gt_features = np.concatenate(gt_feats_list, axis=0)
        np.save(gt_cache_file, gt_features)
        logging.info(f"Saved ground truth features to {gt_cache_file}. Shape: {gt_features.shape}")

    mu_gt = np.mean(gt_features, axis=0)
    sigma_gt = np.cov(gt_features, rowvar=False)

    # 2. Extract features for each arm and compute FID
    arms = [C.NOISE_TYPE_NORMAL, C.NOISE_TYPE_CONSTANT, C.NOISE_TYPE_LINEAR_VM, C.NOISE_TYPE_SLERP, C.NOISE_TYPE_LINEAR]
    arm_fids = {}

    for arm in arms:
        arm_cache_file = results_dir / f"{arm}_radimagenet_features.npy"
        if arm_cache_file.exists():
            logging.info(f"Loading cached features for arm '{arm}' from: {arm_cache_file}")
            arm_feats = np.load(arm_cache_file)
        else:
            logging.info(f"Extracting features for arm '{arm}' across all cases...")
            arm_feats_list = []
            arm_dir = results_dir / arm / "seed_42"
            for key in tqdm(keys, desc=f"Arm {arm}"):
                vol_file = arm_dir / f"{key}.npy"
                if not vol_file.exists():
                    raise FileNotFoundError(f"Missing prediction volume: {vol_file}")
                vol = np.load(vol_file)
                feats = extract_volume_features(vol, feature_extractor, device=device)
                arm_feats_list.append(feats)
            arm_feats = np.concatenate(arm_feats_list, axis=0)
            np.save(arm_cache_file, arm_feats)
            logging.info(f"Saved features for arm '{arm}' to {arm_cache_file}. Shape: {arm_feats.shape}")

        mu_arm = np.mean(arm_feats, axis=0)
        sigma_arm = np.cov(arm_feats, rowvar=False)

        fid_val = calculate_frechet_distance(mu_arm, sigma_arm, mu_gt, sigma_gt)
        arm_fids[arm] = fid_val
        logging.info(f"==> Arm '{arm}' RadImageNet FID: {fid_val:.4f}")

    print("\n" + "=" * 60)
    print("FULL-DATASET RADIMAGENET FID RESULTS (N=100, seed 42):")
    print("=" * 60)
    for arm in arms:
        print(f"  {arm:15s}: FID = {arm_fids[arm]:.4f}")
    print("=" * 60 + "\n")

    # 3. Update ablation_results_raw.pkl
    if raw_pkl_path.exists():
        with open(raw_pkl_path, "rb") as f:
            data = pickle.load(f)
        for arm in arms:
            if arm in data and 42 in data[arm]:
                data[arm][42]["FID"] = [arm_fids[arm]]
        with open(raw_pkl_path, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"Updated {raw_pkl_path} with full-dataset FID scores.")

    # 4. Run analyze_noise_coupling to update tables and reports
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.analyze_noise_coupling import analyze
    analyze(raw_pkl_path, results_dir, modality="ct")


if __name__ == "__main__":
    main()
