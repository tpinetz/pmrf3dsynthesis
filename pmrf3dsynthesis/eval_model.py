import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm
from skimage.transform import resize
from pmrf3dsynthesis.data import constants as C
from pmrf3dsynthesis.evaluation.ssim import SSIM3D
from pmrf3dsynthesis.evaluation.retfound_fid import RETFoundFeatureExtractor
from pmrf3dsynthesis.evaluation.radimagenet_fid import RadImageNetFeatureExtractor
from pmrf3dsynthesis.models.flow_model import FlowModel
from pmrf3dsynthesis.models.mmse_model import MMSEModel
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pmrf3dsynthesis.vessel_segmentation.utils import load_model as load_vessel_model, extract_vessels_oct

import pickle
import logging
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_name: str, ckpt_path: str, device: str = 'cpu') -> torch.nn.Module:
    """Loads trained checkpoint for MMSEModel or FlowModel."""
    if model_name == C.MMSE_MODEL:
        model = MMSEModel.load_from_checkpoint(ckpt_path, map_location=device)
    elif model_name == C.FLOW_MODEL:
        model = FlowModel.load_from_checkpoint(ckpt_path, map_location=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choices: {[C.MMSE_MODEL, C.FLOW_MODEL]}")
    return model


def load_dataset(df_path: Union[str, Path]) -> List[str]:
    """
    Loads dataset keys from a .csv, .txt, or .json file.
    Supports columns: 'Keys', 'keys', 'key', 'patient_id', 'id', etc.
    """
    df_path = Path(df_path)
    if not df_path.exists():
        raise FileNotFoundError(f"Dataset split file not found at: {df_path}")

    suffix = df_path.suffix.lower()
    if suffix == '.txt':
        with open(df_path, 'r') as f:
            keys = [line.strip() for line in f if line.strip()]
    elif suffix == '.json':
        with open(df_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                keys = [str(k).strip() for k in data]
            elif isinstance(data, dict):
                keys = [str(k).strip() for k in data.keys()]
            else:
                raise ValueError(f"Unsupported JSON format in {df_path}")
    else:
        # Default to CSV / table
        df = pd.read_csv(df_path)
        candidate_cols = ['Keys', 'keys', 'Key', 'key', 'patient_id', 'PatientID', 'id', 'ID', 'case', 'Case']
        matched_col = None
        for col in candidate_cols:
            if col in df.columns:
                matched_col = col
                break

        if matched_col is not None:
            keys = list(df[matched_col].astype(str).str.strip())
        else:
            keys = list(df.iloc[:, 0].astype(str).str.strip())

    logging.info(f"Loaded {len(keys)} cases from {df_path}")
    return keys


def slice_volume_ssim(y_hat: torch.Tensor, gt: torch.Tensor) -> float:
    """Computes average 2D slice-wise SSIM across depth dimension."""
    if y_hat.ndim == 3:
        # (D, H, W) -> (1, 1, D, H, W)
        y_hat = y_hat[None, None]
        gt = gt[None, None]
    elif y_hat.ndim == 4:
        # (C, D, H, W) -> (1, C, D, H, W)
        y_hat = y_hat[None]
        gt = gt[None]

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    scores = []
    for i in range(y_hat.shape[2]):
        scores.append(ssim(y_hat[:, :, i], gt[:, :, i]).item())
    return float(np.mean(scores))


def compute_vessel_metrics(
    pred_vessels: np.ndarray,
    gt_vessels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes Dice, IoU, and Soft Dice between predicted and ground truth vessel probability maps.
    """
    pred_bin = pred_vessels > threshold
    gt_bin = gt_vessels > threshold

    intersection = float((pred_bin & gt_bin).sum())
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())
    union = float((pred_bin | gt_bin).sum())

    dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)
    iou = intersection / (union + 1e-8)

    # Continuous soft dice
    soft_dice = (2.0 * float((pred_vessels * gt_vessels).sum())) / (
        float((pred_vessels ** 2).sum()) + float((gt_vessels ** 2).sum()) + 1e-8
    )

    return {
        "Vessel_Dice": dice,
        "Vessel_IoU": iou,
        "Vessel_Soft_Dice": soft_dice
    }


def compute_sdif(pred: np.ndarray, gt: np.ndarray, axis: int = 0) -> Dict[str, float]:
    """
    Computes Slice Difference (SDiff) metrics as defined in ISCS (Du et al., ICLR 2026):
      SDiff(V) = mean(|V[i+1] - V[i]|) along depth/slice axis.
      SDIF (Gap |Delta|) = |SDiff(pred) - SDiff(gt)|
    """
    if pred.ndim == 4:
        pred = pred[0]
    if gt.ndim == 4:
        gt = gt[0]
    if pred.ndim == 5:
        pred = pred[0, 0]
    if gt.ndim == 5:
        gt = gt[0, 0]

    diff_pred = np.abs(np.diff(pred, axis=axis))
    diff_gt = np.abs(np.diff(gt, axis=axis))

    sdiff_pred = float(np.mean(diff_pred))
    sdiff_gt = float(np.mean(diff_gt))
    sdiff_gap = float(np.abs(sdiff_pred - sdiff_gt))

    return {
        "SDIF": sdiff_gap,
        "SDIF_pred": sdiff_pred,
        "SDIF_gt": sdiff_gt
    }


def compute_volume_lpips(
    pred: np.ndarray,
    gt: np.ndarray,
    lpips_fn: torch.nn.Module,
    device: str = 'cuda',
    batch_size: int = 32
) -> float:
    """
    Computes slice-wise LPIPS (Learned Perceptual Image Patch Similarity) averaged over 3D volume slices.
    Expects pred and gt in range [0, 1] with shape (S, H, W).
    """
    if pred.ndim == 4:
        pred = pred[0]
    if gt.ndim == 4:
        gt = gt[0]
    if pred.ndim == 5:
        pred = pred[0, 0]
    if gt.ndim == 5:
        gt = gt[0, 0]

    S = pred.shape[0]
    scores = []

    for i in range(0, S, batch_size):
        p_chunk = pred[i:i + batch_size]
        g_chunk = gt[i:i + batch_size]

        # Convert grayscale (B, H, W) to (B, 3, H, W)
        p_t = torch.from_numpy(p_chunk)[:, None, :, :].repeat(1, 3, 1, 1).float().to(device)
        g_t = torch.from_numpy(g_chunk)[:, None, :, :].repeat(1, 3, 1, 1).float().to(device)

        with torch.no_grad():
            val = lpips_fn(p_t, g_t)
            scores.append(val.item() * len(p_chunk))

    return float(sum(scores) / max(S, 1))


def load_sample(
    key: str,
    modality: str,
    vol_path: Optional[Path] = None,
    img_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    data_format: str = C.FORMAT_AUTO,
    h5_name: str = 'ct_xray_data.h5',
    hu_min: float = -1350.0,
    hu_max: float = 150.0,
    ct_scale: float = 5800.0,
    xray_scale: float = 1200.0,
    target_vol_shape: Optional[Tuple[int, int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads 2D input image and 3D ground truth volume for a given case key.

    Returns:
        input_img: np.ndarray in [0, 1], shape (H, W) or (H, W, C)
        volume: np.ndarray in [0, 1], shape (slices, H, W)
    """
    if modality == C.MODALITY_OCT:
        # Default OCT paths if not provided
        if vol_path is None:
            vol_path = Path(os.path.expandvars("$HOME/data/OCT/retinsight/FundusMIP/flattened_volumes/"))
        if img_path is None:
            img_path = Path(os.path.expandvars("$HOME/data/OCT/retinsight/FundusMIP/Fundus_cropped_color/"))

        vol_file = vol_path / f"{key}.npy"
        img_file = img_path / f"{key}.png"
        if not img_file.exists():
            img_file = img_path / f"{key}.jpg"

        if not vol_file.exists():
            raise FileNotFoundError(f"OCT volume file not found: {vol_file}")
        if not img_file.exists():
            raise FileNotFoundError(f"OCT fundus image file not found: {img_file}")

        volume = np.load(vol_file)
        if volume.shape[1] > 256:
            volume = volume[:, 100:356, :]
        volume = resize(volume, (128, 256, 256))

        input_img = iio.imread(img_file)
        if input_img.dtype == np.uint8:
            input_img = input_img / 255.0
        input_img = resize(input_img, (128, 256))

        return input_img, volume

    elif modality == C.MODALITY_CT:
        # Check HDF5 format
        is_h5 = False
        h5_file = None

        if data_dir is None:
            default_h5_dir = Path(os.path.expandvars("$HOME/data/CT/data/LIDC-HDF5-256"))
            if default_h5_dir.exists():
                data_dir = default_h5_dir

        if data_format == C.FORMAT_H5:
            is_h5 = True
        elif data_format == C.FORMAT_AUTO:
            if data_dir is not None:
                cand1 = data_dir / key / h5_name
                cand2 = data_dir / f"{key}.h5"
                if cand1.exists():
                    h5_file = cand1
                    is_h5 = True
                elif cand2.exists():
                    h5_file = cand2
                    is_h5 = True
            if not is_h5 and vol_path is not None:
                cand3 = vol_path / f"{key}.h5"
                if cand3.exists():
                    h5_file = cand3
                    is_h5 = True

        if is_h5:
            import h5py
            if h5_file is None:
                if data_dir is not None:
                    h5_file = data_dir / key / h5_name
                elif vol_path is not None:
                    h5_file = vol_path / f"{key}.h5"
                else:
                    raise FileNotFoundError(f"Cannot resolve HDF5 path for case {key}")

            with h5py.File(h5_file, 'r') as f:
                # Find CT volume dataset
                ct_ds_name = None
                for name in ['ct', 'CT', 'volume', 'data']:
                    if name in f:
                        ct_ds_name = name
                        break
                if ct_ds_name is None:
                    raise KeyError(f"No CT dataset found in {h5_file}. Available: {list(f.keys())}")

                # Find X-ray projection dataset
                xray_ds_name = None
                for name in ['xray1', 'xray', 'synth_xray', 'drr', 'image', 'xray2']:
                    if name in f:
                        xray_ds_name = name
                        break
                if xray_ds_name is None:
                    raise KeyError(f"No X-ray dataset found in {h5_file}. Available: {list(f.keys())}")

                ct_raw = f[ct_ds_name][:]
                xray_raw = f[xray_ds_name][:]

            volume = (np.clip(ct_raw, 0.0, ct_scale) / ct_scale).astype(np.float32)
            input_img = (np.clip(xray_raw, 0.0, xray_scale) / xray_scale).astype(np.float32)

            if input_img.ndim == 3 and input_img.shape[0] == 1:
                input_img = input_img[0]

            return input_img, volume

        # Check NIfTI / NumPy format
        if vol_path is None:
            vol_path = Path(os.path.expandvars("$HOME/data/CT/LIDC-IDRI/CT/"))
        if img_path is None:
            img_path = Path(os.path.expandvars("$HOME/data/CT/LIDC-IDRI/synth_xray/"))

        # Volume loading (.nii.gz, .nii, or .npy)
        vol_file_nii_gz = vol_path / f"{key}.nii.gz"
        vol_file_nii = vol_path / f"{key}.nii"
        vol_file_npy = vol_path / f"{key}.npy"

        if vol_file_nii_gz.exists() or vol_file_nii.exists():
            import nibabel as nib
            target_file = vol_file_nii_gz if vol_file_nii_gz.exists() else vol_file_nii
            raw_vol = nib.load(str(target_file)).get_fdata()
            # Transpose from (H, W, slices) to (slices, H, W) if necessary
            if raw_vol.ndim == 3:
                volume = raw_vol.transpose((2, 0, 1))
            else:
                volume = raw_vol
            # Apply HU windowing
            hu_range = max(hu_max - hu_min, 1e-6)
            volume = (np.clip(volume, hu_min, hu_max) - hu_min) / hu_range
            volume = volume.astype(np.float32)
        elif vol_file_npy.exists():
            volume = np.load(vol_file_npy).astype(np.float32)
            if volume.max() > 1.0:
                volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-6)
        else:
            raise FileNotFoundError(f"CT volume not found for {key} in {vol_path} (checked .nii.gz, .nii, .npy)")

        # 2D X-ray loading
        img_file_png = img_path / f"{key}.png"
        img_file_jpg = img_path / f"{key}.jpg"
        img_file_npy = img_path / f"{key}.npy"

        if img_file_png.exists() or img_file_jpg.exists():
            target_img_file = img_file_png if img_file_png.exists() else img_file_jpg
            input_img = iio.imread(target_img_file)
            if input_img.dtype == np.uint8:
                input_img = input_img / 255.0
            input_img = input_img.astype(np.float32)
        elif img_file_npy.exists():
            input_img = np.load(img_file_npy).astype(np.float32)
            if input_img.max() > 1.0:
                input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-6)
        else:
            raise FileNotFoundError(f"CT X-ray image not found for {key} in {img_path} (checked .png, .jpg, .npy)")

        if target_vol_shape is not None and volume.shape != target_vol_shape:
            volume = resize(volume, target_vol_shape)

        return input_img, volume

    else:
        raise ValueError(f"Unknown modality: {modality}. Choices: {[C.MODALITY_OCT, C.MODALITY_CT]}")


def setup_fid_metric(
    modality: str,
    fid_feature: str = 'auto',
    radimagenet_path: Optional[Union[str, Path]] = None,
    device: str = 'cpu'
) -> FrechetInceptionDistance:
    """
    Initializes FrechetInceptionDistance metric with appropriate feature extractor.
    - OCT: RETFoundFeatureExtractor (or InceptionV3)
    - CT: RadImageNetFeatureExtractor (or InceptionV3)
    """
    if fid_feature == 'auto':
        if modality == C.MODALITY_OCT:
            feature_extractor = RETFoundFeatureExtractor()
            logging.info("Initialized FID with RETFound feature extractor for OCT.")
        else:
            feature_extractor = RadImageNetFeatureExtractor(checkpoint_path=radimagenet_path)
            logging.info("Initialized FID with RadImageNet ResNet50 feature extractor for CT.")
        fid_metric = FrechetInceptionDistance(feature=feature_extractor, normalize=True).to(device)
    elif fid_feature == 'radimagenet':
        feature_extractor = RadImageNetFeatureExtractor(checkpoint_path=radimagenet_path)
        logging.info("Initialized FID with RadImageNet ResNet50 feature extractor.")
        fid_metric = FrechetInceptionDistance(feature=feature_extractor, normalize=True).to(device)
    elif fid_feature == 'retfound':
        feature_extractor = RETFoundFeatureExtractor()
        logging.info("Initialized FID with RETFound feature extractor.")
        fid_metric = FrechetInceptionDistance(feature=feature_extractor, normalize=True).to(device)
    elif fid_feature == 'inception':
        logging.info("Initialized FID with standard InceptionV3 feature extractor.")
        fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    else:
        raise ValueError(f"Unknown fid_feature: {fid_feature}. Choices: ['auto', 'radimagenet', 'retfound', 'inception']")

    return fid_metric


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    T: int,
    keys: List[str],
    res_path: Path,
    modality: str = C.MODALITY_OCT,
    noise_type: str = C.NOISE_TYPE_NORMAL,
    vol_path: Optional[Path] = None,
    img_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    data_format: str = C.FORMAT_AUTO,
    h5_name: str = 'ct_xray_data.h5',
    radimagenet_path: Optional[Union[str, Path]] = None,
    fid_feature: str = 'auto',
    hu_min: float = -1350.0,
    hu_max: float = 150.0,
    ct_scale: float = 5800.0,
    xray_scale: float = 1200.0,
    eval_vessels: bool = False,
    vessel_threshold: float = 0.5,
    save_vessels: bool = False,
    eval_lpips: bool = True,
    eval_sdif: bool = True,
    save_nii: bool = False,
    device: str = 'cpu'
) -> Dict:
    """Evaluates model over dataset cases and computes PSNR, SSIM, LPIPS, SDIF, FID, and optional vessel metrics."""
    res_path.mkdir(parents=True, exist_ok=True)

    metrics = {
        "PSNR": PeakSignalNoiseRatio(data_range=1.0).to(device),
        "SSIM": SSIM3D().to(device),
        "FID": setup_fid_metric(modality, fid_feature, radimagenet_path, device)
    }
    output = {
        "PSNR": {},
        "SSIM": {},
        "FID": []
    }

    lpips_fn = None
    if eval_lpips:
        try:
            logging.info("Initializing LPIPS metric (AlexNet)...")
            lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
            lpips_fn.eval()
            output["LPIPS"] = {}
        except Exception as e:
            logging.warning(f"Could not initialize LPIPS metric: {e}")

    if eval_sdif:
        output["SDIF"] = {}

    ves_model = None
    if eval_vessels and modality == C.MODALITY_OCT:
        logging.info("Loading pretrained vessel segmentation model for OCT...")
        ves_model = load_vessel_model().to(device)
        ves_model.eval()
        output["Vessel_Dice"] = {}
        output["Vessel_IoU"] = {}
        output["Vessel_Soft_Dice"] = {}

    evaluated_count = 0

    for key in tqdm(keys, desc=f"Evaluating {modality.upper()} model"):
        try:
            input_img, volume = load_sample(
                key=key,
                modality=modality,
                vol_path=vol_path,
                img_path=img_path,
                data_dir=data_dir,
                data_format=data_format,
                h5_name=h5_name,
                hu_min=hu_min,
                hu_max=hu_max,
                ct_scale=ct_scale,
                xray_scale=xray_scale
            )
        except Exception as e:
            logging.error(f"Error loading sample {key}: {e}. Skipping.")
            continue

        # Prepare 2D input tensor in [-1, 1] range
        input_tensor = input_img.astype(np.float32)
        if input_tensor.ndim == 2:
            # (H, W) -> (1, 1, H, W)
            x = torch.from_numpy(input_tensor[None, None] * 2.0 - 1.0).to(device)
        elif input_tensor.ndim == 3:
            if input_tensor.shape[2] in [1, 3]:
                # (H, W, C) -> (1, C, H, W)
                x = torch.from_numpy(input_tensor.transpose((2, 0, 1))[None] * 2.0 - 1.0).to(device)
            elif input_tensor.shape[0] in [1, 3]:
                # (C, H, W) -> (1, C, H, W)
                x = torch.from_numpy(input_tensor[None] * 2.0 - 1.0).to(device)
            else:
                x = torch.from_numpy(input_tensor[None, None] * 2.0 - 1.0).to(device)
        else:
            raise ValueError(f"Unexpected input_img shape: {input_img.shape}")

        # Model prediction
        if isinstance(model, FlowModel):
            y_hat = model(x, T, noise_type=noise_type)
        else:
            y_hat = model(x)

        if isinstance(y_hat, list):
            y_hat = y_hat[-1]

        # Ground truth volume in [0, 1] range: shape (1, 1, D, H, W)
        gt = torch.from_numpy(volume[None, None]).float().to(device)

        # Scale model output from [-1, 1] to [0, 1]
        pred_01 = torch.clamp(y_hat * 0.5 + 0.5, 0.0, 1.0)
        gt_01 = torch.clamp(gt, 0.0, 1.0)

        # Handle resolution mismatch if any
        if pred_01.shape[2:] != gt_01.shape[2:]:
            logging.warning(f"Shape mismatch for {key}: pred {pred_01.shape} vs gt {gt_01.shape}. Interpolating prediction.")
            pred_01 = torch.nn.functional.interpolate(pred_01, size=gt_01.shape[2:], mode='trilinear', align_corners=True)

        psnr_val = metrics["PSNR"](pred_01, gt_01).item()
        ssim_val = metrics["SSIM"](pred_01, gt_01).item()
        output["PSNR"][key] = psnr_val
        output["SSIM"][key] = ssim_val

        pred_np_01 = pred_01.detach().cpu().numpy()[0, 0]
        gt_np_01 = gt_01.detach().cpu().numpy()[0, 0]

        # LPIPS perceptual metric
        if lpips_fn is not None:
            lpips_val = compute_volume_lpips(pred_np_01, gt_np_01, lpips_fn, device=device)
            output["LPIPS"][key] = lpips_val
        else:
            lpips_val = None

        # SDIF (Slice Difference gap) metric
        if eval_sdif:
            sdif_metrics = compute_sdif(pred_np_01, gt_np_01, axis=0)
            output["SDIF"][key] = sdif_metrics["SDIF"]
            sdif_val = sdif_metrics["SDIF"]
        else:
            sdif_val = None

        # Vessel Segmentation evaluation (OCT)
        if ves_model is not None:
            pred_vessels = extract_vessels_oct(pred_np_01, ves_model, device=device)
            gt_vessels = extract_vessels_oct(gt_np_01, ves_model, device=device)

            v_metrics = compute_vessel_metrics(pred_vessels, gt_vessels, threshold=vessel_threshold)
            output["Vessel_Dice"][key] = v_metrics["Vessel_Dice"]
            output["Vessel_IoU"][key] = v_metrics["Vessel_IoU"]
            output["Vessel_Soft_Dice"][key] = v_metrics["Vessel_Soft_Dice"]

            if save_vessels:
                vessels_dir = res_path / "vessels"
                vessels_dir.mkdir(parents=True, exist_ok=True)
                np.save(vessels_dir / f"{key}_pred_vessels.npy", pred_vessels)
                np.save(vessels_dir / f"{key}_gt_vessels.npy", gt_vessels)
                iio.imwrite(vessels_dir / f"{key}_pred_vessels.png", (np.clip(pred_vessels, 0, 1) * 255).astype(np.uint8))
                iio.imwrite(vessels_dir / f"{key}_gt_vessels.png", (np.clip(gt_vessels, 0, 1) * 255).astype(np.uint8))

            log_str = f"Evaluated {key}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}, Vessel_Dice={v_metrics['Vessel_Dice']:.4f}"
            if lpips_val is not None:
                log_str += f", LPIPS={lpips_val:.4f}"
            if sdif_val is not None:
                log_str += f", SDIF={sdif_val:.6f}"
            logging.info(log_str)
        else:
            log_str = f"Evaluated {key}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}"
            if lpips_val is not None:
                log_str += f", LPIPS={lpips_val:.4f}"
            if sdif_val is not None:
                log_str += f", SDIF={sdif_val:.6f}"
            logging.info(log_str)

        # Update FID statistics (expects slices in (N, C, H, W) with values in [0, 1])
        pred_slices = pred_01[0].transpose(1, 0)  # (D, C, H, W)
        gt_slices = gt_01[0].transpose(1, 0)      # (D, C, H, W)

        if fid_feature == 'inception':
            # Standard InceptionV3 requires 3 channels
            if pred_slices.shape[1] == 1:
                pred_slices = pred_slices.repeat(1, 3, 1, 1)
            if gt_slices.shape[1] == 1:
                gt_slices = gt_slices.repeat(1, 3, 1, 1)

        metrics["FID"].update(pred_slices, real=False)
        metrics["FID"].update(gt_slices, real=True)

        # Save prediction
        pred_np = y_hat.detach().cpu().numpy()[0, 0]
        np.save(res_path / f"{key}.npy", pred_np)

        if save_nii and modality == C.MODALITY_CT:
            import nibabel as nib
            # Transpose back to (H, W, slices) for NIfTI convention
            nii_img = nib.Nifti1Image(pred_np.transpose((1, 2, 0)), np.eye(4))
            nib.save(nii_img, str(res_path / f"{key}.nii.gz"))

        evaluated_count += 1

    if evaluated_count == 0:
        raise RuntimeError("No cases were successfully evaluated. Please verify paths and dataset split.")

    fid_score = metrics["FID"].compute().item()
    output["FID"] = [fid_score]

    psnr_values = list(output["PSNR"].values())
    ssim_values = list(output["SSIM"].values())

    logging.info(f"--- Evaluation Complete ({evaluated_count} cases) ---")
    logging.info(f"PSNR:  {np.mean(psnr_values):.4f} +/- {np.std(psnr_values, ddof=1) if len(psnr_values)>1 else 0.0:.4f}")
    logging.info(f"SSIM:  {np.mean(ssim_values):.4f} +/- {np.std(ssim_values, ddof=1) if len(ssim_values)>1 else 0.0:.4f}")
    if "LPIPS" in output and output["LPIPS"]:
        lpips_vals = list(output["LPIPS"].values())
        logging.info(f"LPIPS: {np.mean(lpips_vals):.4f} +/- {np.std(lpips_vals, ddof=1) if len(lpips_vals)>1 else 0.0:.4f}")
    if "SDIF" in output and output["SDIF"]:
        sdif_vals = list(output["SDIF"].values())
        logging.info(f"SDIF:  {np.mean(sdif_vals):.6f} +/- {np.std(sdif_vals, ddof=1) if len(sdif_vals)>1 else 0.0:.6f}")
    logging.info(f"FID:   {fid_score:.4f}")

    if "Vessel_Dice" in output and output["Vessel_Dice"]:
        v_dices = list(output["Vessel_Dice"].values())
        v_ious = list(output["Vessel_IoU"].values())
        v_soft = list(output["Vessel_Soft_Dice"].values())
        logging.info(f"Vessel Dice:      {np.mean(v_dices):.4f} +/- {np.std(v_dices, ddof=1) if len(v_dices)>1 else 0.0:.4f}")
        logging.info(f"Vessel IoU:       {np.mean(v_ious):.4f} +/- {np.std(v_ious, ddof=1) if len(v_ious)>1 else 0.0:.4f}")
        logging.info(f"Vessel Soft Dice: {np.mean(v_soft):.4f} +/- {np.std(v_soft, ddof=1) if len(v_soft)>1 else 0.0:.4f}")

    return output


def save_summary_table(eval_results: Dict, res_path: Path):
    """Exports tabular CSV summary with per-case metrics and summary statistics."""
    has_lpips = "LPIPS" in eval_results and len(eval_results["LPIPS"]) > 0
    has_sdif = "SDIF" in eval_results and len(eval_results["SDIF"]) > 0
    has_vessels = "Vessel_Dice" in eval_results and len(eval_results["Vessel_Dice"]) > 0

    rows = []
    for key in eval_results["PSNR"].keys():
        row = {
            "Key": key,
            "PSNR": eval_results["PSNR"][key],
            "SSIM": eval_results["SSIM"][key]
        }
        if has_lpips:
            row["LPIPS"] = eval_results["LPIPS"].get(key, np.nan)
        if has_sdif:
            row["SDIF"] = eval_results["SDIF"].get(key, np.nan)
        if has_vessels:
            row["Vessel_Dice"] = eval_results.get("Vessel_Dice", {}).get(key, np.nan)
            if "Vessel_IoU" in eval_results:
                row["Vessel_IoU"] = eval_results["Vessel_IoU"].get(key, np.nan)
            if "Vessel_Soft_Dice" in eval_results:
                row["Vessel_Soft_Dice"] = eval_results["Vessel_Soft_Dice"].get(key, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)

    summary_rows = []
    mean_row = {"Key": "MEAN", "PSNR": df["PSNR"].mean(), "SSIM": df["SSIM"].mean()}
    std_row = {
        "Key": "STD",
        "PSNR": df["PSNR"].std(ddof=1) if len(df) > 1 else 0.0,
        "SSIM": df["SSIM"].std(ddof=1) if len(df) > 1 else 0.0
    }
    if has_lpips:
        mean_row["LPIPS"] = df["LPIPS"].mean()
        std_row["LPIPS"] = df["LPIPS"].std(ddof=1) if len(df) > 1 else 0.0
    if has_sdif:
        mean_row["SDIF"] = df["SDIF"].mean()
        std_row["SDIF"] = df["SDIF"].std(ddof=1) if len(df) > 1 else 0.0
    if has_vessels:
        mean_row["Vessel_Dice"] = df["Vessel_Dice"].mean()
        std_row["Vessel_Dice"] = df["Vessel_Dice"].std(ddof=1) if len(df) > 1 else 0.0
        if "Vessel_IoU" in df.columns:
            mean_row["Vessel_IoU"] = df["Vessel_IoU"].mean()
            std_row["Vessel_IoU"] = df["Vessel_IoU"].std(ddof=1) if len(df) > 1 else 0.0
        if "Vessel_Soft_Dice" in df.columns:
            mean_row["Vessel_Soft_Dice"] = df["Vessel_Soft_Dice"].mean()
            std_row["Vessel_Soft_Dice"] = df["Vessel_Soft_Dice"].std(ddof=1) if len(df) > 1 else 0.0

    summary_rows.append(mean_row)
    summary_rows.append(std_row)

    fid_val = eval_results["FID"][0] if eval_results.get("FID") else np.nan
    fid_row = {"Key": "FID", "PSNR": fid_val, "SSIM": fid_val}
    if has_lpips:
        fid_row["LPIPS"] = np.nan
    if has_sdif:
        fid_row["SDIF"] = np.nan
    if has_vessels:
        fid_row["Vessel_Dice"] = np.nan
        fid_row["Vessel_IoU"] = np.nan
        fid_row["Vessel_Soft_Dice"] = np.nan
    summary_rows.append(fid_row)

    summary_df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
    summary_csv = res_path / "eval_results.csv"
    summary_df.to_csv(summary_csv, index=False)
    logging.info(f"Saved tabular metrics to: {summary_csv}")


def run_cli():
    parser = argparse.ArgumentParser(description="Evaluate 3D Synthesis Models on OCT or CT datasets.")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--model_name', type=str, choices=[C.MMSE_MODEL, C.FLOW_MODEL], required=True, help='Name of the model to evaluate.')
    parser.add_argument('--df', type=str, required=True, help='Path to the dataset split file (.csv, .txt, or .json).')
    parser.add_argument('--res_path', type=str, required=True, help='Path to save evaluation results.')
    parser.add_argument('--mmse_path', type=str, default='', help='Path to the MMSE model file (if applicable for FlowModel).')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the evaluation on (e.g. cuda, cpu).')
    parser.add_argument('--T', type=int, default=20, help='Number of flow steps (for FlowModel).')
    parser.add_argument('--modality', type=str, choices=[C.MODALITY_OCT, C.MODALITY_CT], default=C.MODALITY_OCT, help='Modality to evaluate on (oct or ct).')
    parser.add_argument('--noise_type', type=str, choices=[C.NOISE_TYPE_NORMAL, C.NOISE_TYPE_CONSTANT, C.NOISE_TYPE_SLERP, C.NOISE_TYPE_LINEAR_VM, C.NOISE_TYPE_LINEAR], default=C.NOISE_TYPE_NORMAL, help='Type of noise to apply.')

    # Custom path and format arguments
    parser.add_argument('--vol_path', type=str, default=None, help='Custom path to ground truth 3D volume directory.')
    parser.add_argument('--img_path', type=str, default=None, help='Custom path to input 2D projection/image directory.')
    parser.add_argument('--data_dir', type=str, default=None, help='Base data directory (e.g. for HDF5 dataset root).')
    parser.add_argument('--data_format', type=str, choices=[C.FORMAT_AUTO, C.FORMAT_NII, C.FORMAT_H5, C.FORMAT_NPY], default=C.FORMAT_AUTO, help='Data storage format.')
    parser.add_argument('--h5_name', type=str, default='ct_xray_data.h5', help='HDF5 filename within case directory.')

    # FID and RadImageNet parameters
    parser.add_argument('--radimagenet_path', type=str, default=None, help='Path to pretrained RadImageNet ResNet50 checkpoint.')
    parser.add_argument('--fid_feature', type=str, choices=['auto', 'radimagenet', 'retfound', 'inception'], default='auto', help='Feature extractor for FID calculation.')

    # CT normalization parameters
    parser.add_argument('--hu_min', type=float, default=-1350.0, help='Minimum HU value for CT clipping (default: -1350).')
    parser.add_argument('--hu_max', type=float, default=150.0, help='Maximum HU value for CT clipping (default: 150).')
    parser.add_argument('--ct_scale', type=float, default=5800.0, help='Maximum value for HDF5 CT normalization (default: 5800).')
    parser.add_argument('--xray_scale', type=float, default=1200.0, help='Maximum value for HDF5 X-ray normalization (default: 1200).')
    parser.add_argument('--save_nii', action='store_true', help='Also save predicted volumes as NIfTI (.nii.gz) for CT.')

    # Vessel segmentation parameters (OCT)
    parser.add_argument('--eval_vessels', action='store_true', help='Evaluate 2D vessel segmentation for OCT using pretrained CustomNet.')
    parser.add_argument('--vessel_threshold', type=float, default=0.5, help='Binary threshold for vessel segmentation (default: 0.5).')
    parser.add_argument('--save_vessels', action='store_true', help='Save predicted and GT vessel probability maps to res_path/vessels.')

    # LPIPS and SDIF parameters
    parser.add_argument('--no_lpips', action='store_true', help='Disable LPIPS perceptual similarity evaluation.')
    parser.add_argument('--no_sdif', action='store_true', help='Disable SDIF (Slice Difference gap) evaluation.')

    args = parser.parse_args()

    model = load_model(args.model_name, args.ckpt_path, args.device)

    if args.device != 'cpu':
        torch.set_float32_matmul_precision('high')

    if args.mmse_path:
        mmse_model = load_model(C.MMSE_MODEL, args.mmse_path, device=args.device)
        model.set_mmse_model(mmse_model)

    model = model.to(args.device)

    keys = load_dataset(args.df)

    eval_name = Path(args.ckpt_path).parent.parent.parent.name
    if eval_name in ["", ".", "/"]:
        eval_name = Path(args.ckpt_path).stem

    if args.model_name == C.FLOW_MODEL:
        eval_name += f"_with_{args.T}_steps"

    if args.noise_type != C.NOISE_TYPE_NORMAL:
        eval_name += f"_{args.noise_type}_noise"

    logging.info(f"Starting evaluation: {eval_name} (Modality: {args.modality.upper()})")

    res_path = Path(args.res_path) / eval_name
    res_path.mkdir(parents=True, exist_ok=True)

    vol_path = Path(args.vol_path) if args.vol_path else None
    img_path = Path(args.img_path) if args.img_path else None
    data_dir = Path(args.data_dir) if args.data_dir else None

    eval_results = evaluate_model(
        model=model.eval(),
        T=args.T,
        keys=keys,
        res_path=res_path,
        modality=args.modality,
        noise_type=args.noise_type,
        vol_path=vol_path,
        img_path=img_path,
        data_dir=data_dir,
        data_format=args.data_format,
        h5_name=args.h5_name,
        radimagenet_path=args.radimagenet_path,
        fid_feature=args.fid_feature,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        ct_scale=args.ct_scale,
        xray_scale=args.xray_scale,
        eval_vessels=args.eval_vessels,
        vessel_threshold=args.vessel_threshold,
        save_vessels=args.save_vessels,
        eval_lpips=not args.no_lpips,
        eval_sdif=not args.no_sdif,
        save_nii=args.save_nii,
        device=args.device
    )

    with open(res_path / "eval_results.pkl", "wb") as f:
        pickle.dump(eval_results, f)

    save_summary_table(eval_results, res_path)
    print(f"\nFinal Evaluation Results for {eval_name}:")
    psnrs = list(eval_results['PSNR'].values())
    ssims = list(eval_results['SSIM'].values())
    print(f"PSNR:  {np.mean(psnrs):.4f} +/- {np.std(psnrs, ddof=1) if len(psnrs)>1 else 0.0:.4f}")
    print(f"SSIM:  {np.mean(ssims):.4f} +/- {np.std(ssims, ddof=1) if len(ssims)>1 else 0.0:.4f}")
    if "LPIPS" in eval_results and eval_results["LPIPS"]:
        lpips_vals = list(eval_results['LPIPS'].values())
        print(f"LPIPS: {np.mean(lpips_vals):.4f} +/- {np.std(lpips_vals, ddof=1) if len(lpips_vals)>1 else 0.0:.4f}")
    if "SDIF" in eval_results and eval_results["SDIF"]:
        sdif_vals = list(eval_results['SDIF'].values())
        print(f"SDIF:  {np.mean(sdif_vals):.6f} +/- {np.std(sdif_vals, ddof=1) if len(sdif_vals)>1 else 0.0:.6f}")
    print(f"FID:   {eval_results['FID'][0]:.4f}")
    if "Vessel_Dice" in eval_results and eval_results["Vessel_Dice"]:
        v_dices = list(eval_results['Vessel_Dice'].values())
        print(f"Vessel Dice: {np.mean(v_dices):.4f} +/- {np.std(v_dices, ddof=1) if len(v_dices)>1 else 0.0:.4f}")


if __name__ == '__main__':
    run_cli()

