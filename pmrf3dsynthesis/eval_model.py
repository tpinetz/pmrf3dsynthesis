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
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    scores = []
    for i in range(y_hat.shape[2]):
        scores.append(ssim(y_hat[:, :, i], gt[:, :, i]).item())
    return float(np.mean(scores))


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
    save_nii: bool = False,
    device: str = 'cpu'
) -> Dict:
    """Evaluates model over dataset cases and computes PSNR, SSIM, and FID."""
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

        logging.info(f"Evaluated {key}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")
        evaluated_count += 1

    if evaluated_count == 0:
        raise RuntimeError("No cases were successfully evaluated. Please verify paths and dataset split.")

    fid_score = metrics["FID"].compute().item()
    output["FID"] = [fid_score]

    psnr_values = list(output["PSNR"].values())
    ssim_values = list(output["SSIM"].values())

    logging.info(f"--- Evaluation Complete ({evaluated_count} cases) ---")
    logging.info(f"PSNR: {np.mean(psnr_values):.4f} +/- {np.std(psnr_values, ddof=1) if len(psnr_values)>1 else 0.0:.4f}")
    logging.info(f"SSIM: {np.mean(ssim_values):.4f} +/- {np.std(ssim_values, ddof=1) if len(ssim_values)>1 else 0.0:.4f}")
    logging.info(f"FID:  {fid_score:.4f}")

    return output


def save_summary_table(eval_results: Dict, res_path: Path):
    """Exports tabular CSV summary with per-case metrics and summary statistics."""
    rows = []
    for key in eval_results["PSNR"].keys():
        rows.append({
            "Key": key,
            "PSNR": eval_results["PSNR"][key],
            "SSIM": eval_results["SSIM"][key]
        })

    df = pd.DataFrame(rows)

    psnr_vals = df["PSNR"].values
    ssim_vals = df["SSIM"].values
    fid_val = eval_results["FID"][0] if eval_results["FID"] else np.nan

    summary_rows = [
        {"Key": "MEAN", "PSNR": np.mean(psnr_vals), "SSIM": np.mean(ssim_vals)},
        {"Key": "STD", "PSNR": np.std(psnr_vals, ddof=1) if len(psnr_vals) > 1 else 0.0, "SSIM": np.std(ssim_vals, ddof=1) if len(ssim_vals) > 1 else 0.0},
        {"Key": "FID", "PSNR": fid_val, "SSIM": fid_val}
    ]
    df_summary = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
    csv_path = res_path / "eval_results.csv"
    df_summary.to_csv(csv_path, index=False)
    logging.info(f"Saved tabular metrics to: {csv_path}")


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
    parser.add_argument('--noise_type', type=str, choices=[C.NOISE_TYPE_NORMAL, C.NOISE_TYPE_CONSTANT, C.NOISE_TYPE_SLERP], default=C.NOISE_TYPE_NORMAL, help='Type of noise to apply.')

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
        save_nii=args.save_nii,
        device=args.device
    )

    with open(res_path / "eval_results.pkl", "wb") as f:
        pickle.dump(eval_results, f)

    save_summary_table(eval_results, res_path)
    print(f"\nFinal Evaluation Results for {eval_name}:")
    psnrs = list(eval_results['PSNR'].values())
    ssims = list(eval_results['SSIM'].values())
    print(f"PSNR: {np.mean(psnrs):.4f} +/- {np.std(psnrs, ddof=1) if len(psnrs)>1 else 0.0:.4f}")
    print(f"SSIM: {np.mean(ssims):.4f} +/- {np.std(ssims, ddof=1) if len(ssims)>1 else 0.0:.4f}")
    print(f"FID:  {eval_results['FID'][0]:.4f}")


if __name__ == '__main__':
    run_cli()
