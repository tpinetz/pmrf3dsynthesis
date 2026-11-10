from pathlib import Path
import argparse
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd
import pydicom
from skimage.transform import resize
from tqdm import tqdm
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from pmrf3dsynthesis.data import constants as C
from pmrf3dsynthesis.evaluation.ssim import SSIM3D
from pmrf3dsynthesis.models.flow_model import FlowModel
from pmrf3dsynthesis.models.mmse_model import MMSEModel
from pmrf3dsynthesis.eval_model import compute_sdif, compute_volume_lpips, compute_vessel_metrics, setup_fid_metric
from pmrf3dsynthesis.vessel_segmentation.utils import load_model as load_vessel_model, extract_vessels_oct

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_fundus_image(path: Union[str, Path]) -> np.ndarray:
    """
    Loads a fundus image from file (.png, .jpg, .jpeg, .tif, .dcm, etc.).
    Returns uint8 RGB image of shape (H, W, 3).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fundus image file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".dcm":
        dcm = pydicom.dcmread(str(path))
        img = dcm.pixel_array
    else:
        img = iio.imread(str(path))

    # Convert grayscale (H, W) or (H, W, 1) to (H, W, 3)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.ndim == 3 and img.shape[2] >= 4:
        # Discard alpha channel
        img = img[:, :, :3]

    # Normalize dtype to uint8 [0, 255]
    if img.dtype != np.uint8:
        img_min = float(img.min())
        img_max = float(img.max())
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
        else:
            img = np.zeros(img.shape, dtype=np.uint8)

    return img


def resolve_checkpoint_path(path: Union[str, Path]) -> Path:
    """
    Resolves checkpoint file from given file or directory path.
    If a directory is passed, recursively searches for .ckpt files, prioritizing 'last.ckpt'.
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        ckpt_files = sorted(list(p.rglob("*.ckpt")))
        if ckpt_files:
            last_ckpts = [f for f in ckpt_files if "last" in f.name.lower()]
            if last_ckpts:
                return last_ckpts[0]
            return ckpt_files[-1]
    raise FileNotFoundError(f"No checkpoint (.ckpt) file found at: {path}")


def load_synthesis_model(
    model_name: str,
    ckpt_path: Union[str, Path],
    mmse_path: Optional[Union[str, Path]] = None,
    device: str = "cpu"
) -> torch.nn.Module:
    """Loads trained checkpoint for MMSEModel or FlowModel."""
    resolved_ckpt = resolve_checkpoint_path(ckpt_path)
    logging.info(f"Loading {model_name} from {resolved_ckpt}...")

    model_name_lower = model_name.lower().replace("-", "_")
    if model_name_lower in [C.MMSE_MODEL, "mmse"]:
        model = MMSEModel.load_from_checkpoint(str(resolved_ckpt), map_location=device)
    elif model_name_lower in [C.FLOW_MODEL, "flow_model", "flow", "rflow"]:
        model = FlowModel.load_from_checkpoint(str(resolved_ckpt), map_location=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choices: ['mmse', 'flow_model', 'rflow']")

    if mmse_path is not None and hasattr(model, "set_mmse_model"):
        resolved_mmse_ckpt = resolve_checkpoint_path(mmse_path)
        logging.info(f"Loading auxiliary MMSE guidance model from {resolved_mmse_ckpt}...")
        mmse_model = MMSEModel.load_from_checkpoint(str(resolved_mmse_ckpt), map_location=device)
        model.set_mmse_model(mmse_model)

    model = model.to(device)
    model.eval()
    return model


def crop_fundus_fovea(
    fundus_img: np.ndarray,
    fov_model: Optional[object] = None,
    crop_radius: Optional[int] = None,
    crop_size: Union[int, Tuple[int, int]] = 850,
    fovea_coords: Optional[Tuple[float, float]] = None,
    device: str = "cpu"
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Extracts an anatomically centered crop around the fovea with reflection/edge boundary padding.
    
    Args:
        fundus_img: uint8 RGB fundus image of shape (H, W, 3).
        fov_model: Optional fovea detector from fundus_image_toolbox.load_fovea_od_model.
        crop_radius: Radius in pixels around fovea to crop (default None -> crop_size // 2, e.g. 425 for 850x850).
        crop_size: Output resolution to resize cropped window to (default 850, yielding (850, 850)).
                   If None or <= 0, preserves the unscaled crop size (2 * crop_radius, 2 * crop_radius).
        fovea_coords: Optional manual (fovea_x, fovea_y) coordinates.
        device: Device used for fovea detection if fov_model is loaded dynamically.

    Returns:
        cropped_fundus: Cropped and resized fundus image of shape (crop_size, crop_size, 3).
        (fovea_x, fovea_y): Fovea coordinates used for cropping.
    """
    H, W = fundus_img.shape[:2]

    # Resolve target crop size and radius
    if isinstance(crop_size, (tuple, list)):
        target_h, target_w = int(crop_size[0]), int(crop_size[1])
    elif crop_size is not None and crop_size > 0:
        target_h, target_w = int(crop_size), int(crop_size)
    else:
        target_h, target_w = 850, 850

    if crop_radius is None:
        crop_radius = max(target_h, target_w) // 2

    # 1. Determine fovea coordinates
    if fovea_coords is not None:
        fovea_x, fovea_y = float(fovea_coords[0]), float(fovea_coords[1])
    elif fov_model is not None:
        try:
            res = fov_model.predict(fundus_img)
            fovea_x, fovea_y = float(res[0]), float(res[1])
            # Validate detection sanity
            if not (0 <= fovea_x < W and 0 <= fovea_y < H):
                logging.warning(
                    f"Fovea prediction out of bounds ({fovea_x:.1f}, {fovea_y:.1f}) for image size ({W}x{H}). "
                    "Falling back to image center."
                )
                fovea_x, fovea_y = W / 2.0, H / 2.0
        except Exception as e:
            logging.warning(f"Fovea detection failed with error: {e}. Falling back to image center.")
            fovea_x, fovea_y = W / 2.0, H / 2.0
    else:
        # Default fallback to center of the image
        fovea_x, fovea_y = W / 2.0, H / 2.0

    # 2. Compute bounding box coordinates centered at fovea
    fx = int(round(fovea_x))
    fy = int(round(fovea_y))

    x_min = fx - crop_radius
    x_max = fx + crop_radius
    y_min = fy - crop_radius
    y_max = fy + crop_radius

    slice_y1 = max(0, y_min)
    slice_y2 = min(H, y_max)
    slice_x1 = max(0, x_min)
    slice_x2 = min(W, x_max)

    valid_crop = fundus_img[slice_y1:slice_y2, slice_x1:slice_x2]

    # Calculate padding required to reach exact (2*crop_radius, 2*crop_radius)
    pad_top = max(0, -y_min)
    pad_bottom = max(0, y_max - H)
    pad_left = max(0, -x_min)
    pad_right = max(0, x_max - W)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        # Prefer reflection padding; fallback to edge padding if valid crop is too small for reflect
        try:
            if valid_crop.ndim == 3:
                padded = np.pad(
                    valid_crop,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="reflect"
                )
            else:
                padded = np.pad(
                    valid_crop,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="reflect"
                )
        except ValueError:
            if valid_crop.ndim == 3:
                padded = np.pad(
                    valid_crop,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="edge"
                )
            else:
                padded = np.pad(
                    valid_crop,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="edge"
                )
    else:
        padded = valid_crop

    # 3. Resize to target crop size
    if isinstance(crop_size, (tuple, list)):
        target_h, target_w = int(crop_size[0]), int(crop_size[1])
    elif crop_size is not None and crop_size > 0:
        target_h, target_w = int(crop_size), int(crop_size)
    else:
        target_h, target_w = padded.shape[0], padded.shape[1]

    if (padded.shape[0], padded.shape[1]) != (target_h, target_w):
        cropped_fundus = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        cropped_fundus = padded

    return cropped_fundus, (fovea_x, fovea_y)


def prepare_model_input(cropped_fundus: np.ndarray, target_shape: Tuple[int, int] = (128, 256)) -> torch.Tensor:
    """
    Converts a uint8 cropped fundus image (e.g. 850x850x3) to a model input tensor (1, 3, 128, 256)
    normalized to the [-1.0, 1.0] range.
    """
    img_float = cropped_fundus.astype(np.float32)
    if img_float.max() > 1.0:
        img_float = img_float / 255.0

    # Resize to (128, 256) matching RetInsight dataset model inputs
    img_resized = resize(img_float, target_shape, preserve_range=True)

    if img_resized.ndim == 2:
        # Grayscale (H, W) -> (1, 1, H, W)
        x = torch.from_numpy(img_resized[None, None] * 2.0 - 1.0).float()
    elif img_resized.ndim == 3:
        if img_resized.shape[2] in [1, 3]:
            # (H, W, C) -> (1, C, H, W)
            x = torch.from_numpy(img_resized.transpose((2, 0, 1))[None] * 2.0 - 1.0).float()
        elif img_resized.shape[0] in [1, 3]:
            # (C, H, W) -> (1, C, H, W)
            x = torch.from_numpy(img_resized[None] * 2.0 - 1.0).float()
        else:
            x = torch.from_numpy(img_resized[None, None] * 2.0 - 1.0).float()
    else:
        raise ValueError(f"Unexpected image shape: {img_resized.shape}")

    return x


def collect_fundus_inputs(
    fundus_path: Union[str, Path],
    fundus_dir: Optional[Union[str, Path]] = None
) -> List[Tuple[str, Path]]:
    """
    Collects list of (case_key, file_path) from a single file, directory, or CSV/split file.
    """
    fundus_path = Path(fundus_path)
    if not fundus_path.exists():
        raise FileNotFoundError(f"Provided fundus path does not exist: {fundus_path}")

    valid_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"]

    if fundus_path.is_file():
        suffix = fundus_path.suffix.lower()
        if suffix in [".csv", ".txt", ".json"]:
            # Load dataset keys/paths from file
            if suffix == ".txt":
                with open(fundus_path, "r") as f:
                    keys = [line.strip() for line in f if line.strip()]
            elif suffix == ".json":
                with open(fundus_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        keys = [str(k).strip() for k in data]
                    elif isinstance(data, dict):
                        keys = [str(k).strip() for k in data.keys()]
                    else:
                        raise ValueError(f"Unsupported JSON structure in {fundus_path}")
            else:
                df = pd.read_csv(fundus_path)
                candidate_cols = ["Keys", "keys", "Key", "key", "patient_id", "PatientID", "id", "ID", "path", "file"]
                matched = next((c for c in candidate_cols if c in df.columns), None)
                if matched is not None:
                    keys = list(df[matched].astype(str).str.strip())
                else:
                    keys = list(df.iloc[:, 0].astype(str).str.strip())

            base_dir = Path(fundus_dir) if fundus_dir else fundus_path.parent
            items = []
            for k in keys:
                # Check if k is a direct file path
                cand_path = Path(k)
                if cand_path.exists():
                    items.append((cand_path.stem, cand_path))
                    continue
                # Search in base_dir
                found = None
                for ext in valid_exts:
                    c = base_dir / f"{k}{ext}"
                    if c.exists():
                        found = c
                        break
                if found:
                    items.append((k, found))
                else:
                    logging.warning(f"Could not find fundus file for key '{k}' in {base_dir}")
            return items
        elif suffix in valid_exts:
            return [(fundus_path.stem, fundus_path)]
        else:
            raise ValueError(f"Unsupported file extension: {fundus_path.suffix}")

    elif fundus_path.is_dir():
        all_files = sorted([f for f in fundus_path.rglob("*") if f.suffix.lower() in valid_exts])
        return [(f.stem, f) for f in all_files]

    else:
        raise ValueError(f"Invalid fundus path: {fundus_path}")


@torch.no_grad()
def run_fundus_synthesis(
    fundus_inputs: List[Tuple[str, Path]],
    res_path: Union[str, Path],
    ckpt_path: Optional[Union[str, Path]] = None,
    model_name: str = "mmse",
    mmse_path: Optional[Union[str, Path]] = None,
    crop_only: bool = False,
    crop_radius: Optional[int] = None,
    crop_size: int = 850,
    save_crops: bool = True,
    save_crops_dir: Optional[Union[str, Path]] = None,
    save_mip: bool = False,
    fovea_coords: Optional[Tuple[float, float]] = None,
    detect_fovea: bool = True,
    T: int = 10,
    noise_type: str = C.NOISE_TYPE_NORMAL,
    device: str = "cpu",
    gt_vol_dir: Optional[Union[str, Path]] = None,
    eval_vessels: bool = False,
    eval_fid: bool = True,
    fid_feature: str = "all",
) -> Dict[str, Union[int, List[str], Dict]]:
    """
    Main execution pipeline:
    1. Extracts fovea-centered fundus crops.
    2. Saves crops for pre-processing / downstream tasks.
    3. Unless crop_only is specified, runs 3D OCT synthesis model and saves .npy volumes.
    4. If gt_vol_dir is provided, computes quantitative evaluation metrics (including FID) against GT volumes.
    """
    res_path = Path(res_path)
    res_path.mkdir(parents=True, exist_ok=True)

    # Determine crops directory
    if save_crops_dir is not None:
        crops_dir = Path(save_crops_dir)
    elif crop_only:
        crops_dir = res_path
    else:
        crops_dir = res_path / "crops"

    if save_crops or crop_only:
        crops_dir.mkdir(parents=True, exist_ok=True)

    # Load fovea detection model if enabled (kept on CPU to conserve GPU VRAM for 3D synthesis)
    fov_model = None
    if detect_fovea and fovea_coords is None:
        try:
            import fundus_image_toolbox as fit
            logging.info("Loading fovea localization model from fundus_image_toolbox (on CPU)...")
            fov_model, _ = fit.load_fovea_od_model(device="cpu", return_test_dataloader=False)
        except Exception as e:
            logging.warning(f"Could not load fovea model ({e}). Using image center as fallback.")
            fov_model = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load synthesis model if not crop_only
    model = None
    if not crop_only:
        if ckpt_path is None:
            raise ValueError("ckpt_path must be specified when running synthesis (or pass --crop_only).")
        model = load_synthesis_model(
            model_name=model_name,
            ckpt_path=ckpt_path,
            mmse_path=mmse_path,
            device=device
        )
        if device != "cpu" and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

    # Setup metrics if ground truth volume directory is provided
    gt_path_obj = Path(gt_vol_dir) if gt_vol_dir is not None else None
    psnr_metric = None
    ssim3d_metric = None
    ssim2d_metric = None
    lpips_metric = None
    ves_model = None
    fid_inception = None
    fid_retfound = None
    eval_records = []

    if gt_path_obj is not None and not crop_only:
        logging.info(f"Ground truth OCT directory specified: {gt_path_obj}. Quantitative evaluation enabled.")
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim3d_metric = SSIM3D().to(device)
        ssim2d_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        try:
            lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
            lpips_metric.eval()
        except Exception as e:
            logging.warning(f"Could not initialize LPIPS metric ({e}). Skipping LPIPS.")
            lpips_metric = None

        if eval_vessels:
            try:
                logging.info("Loading pre-trained OCT vessel segmentation model...")
                ves_model = load_vessel_model().to(device)
                ves_model.eval()
            except Exception as e:
                logging.warning(f"Could not load vessel segmentation model ({e}). Skipping vessel metrics.")
                ves_model = None

        if eval_fid:
            if fid_feature in ["all", "inception"]:
                try:
                    logging.info("Initializing FID with standard InceptionV3 feature extractor...")
                    fid_inception = setup_fid_metric(C.MODALITY_OCT, fid_feature="inception", device=device)
                except Exception as e:
                    logging.warning(f"Could not initialize Inception FID ({e}). Skipping Inception FID.")
                    fid_inception = None

            if fid_feature in ["all", "retfound", "auto"]:
                try:
                    logging.info("Initializing FID with RETFound feature extractor...")
                    fid_retfound = setup_fid_metric(C.MODALITY_OCT, fid_feature="retfound", device=device)
                except Exception as e:
                    logging.warning(f"Could not initialize RETFound FID ({e}). Skipping RETFound FID.")
                    fid_retfound = None


    metadata = []
    processed_keys = []

    logging.info(f"Processing {len(fundus_inputs)} fundus image(s)...")
    for key, img_path in tqdm(fundus_inputs, desc="Processing fundus"):
        try:
            # 1. Load fundus image
            fundus_img = load_fundus_image(img_path)
            orig_h, orig_w = fundus_img.shape[:2]

            # 2. Extract macular crop centered on fovea
            cropped_fundus, (fx, fy) = crop_fundus_fovea(
                fundus_img=fundus_img,
                fov_model=fov_model,
                crop_radius=crop_radius,
                crop_size=crop_size,
                fovea_coords=fovea_coords,
                device=device
            )

            # 3. Save cropped fundus if requested
            if save_crops or crop_only:
                crop_out_path = crops_dir / f"{key}.png"
                iio.imwrite(str(crop_out_path), cropped_fundus)

            metadata.append({
                "key": key,
                "fundus_path": str(img_path),
                "orig_width": orig_w,
                "orig_height": orig_h,
                "fovea_x": fx,
                "fovea_y": fy,
                "crop_radius": crop_radius,
                "crop_size": crop_size
            })

            # 4. Perform 3D OCT synthesis if not crop_only
            if not crop_only and model is not None:
                x = prepare_model_input(cropped_fundus, target_shape=(128, 256)).to(device)

                if isinstance(model, FlowModel):
                    y_hat = model(x, T, noise_type=noise_type)
                else:
                    y_hat = model(x)

                if isinstance(y_hat, list):
                    y_hat = y_hat[-1]

                pred_np = y_hat.detach().cpu().numpy()[0, 0]  # shape: (128, 256, 256) in [-1, 1] range

                vol_out_path = res_path / f"{key}.npy"
                np.save(vol_out_path, pred_np)

                pred_01 = np.clip(pred_np * 0.5 + 0.5, 0.0, 1.0)

                if save_mip:
                    # En-face projection across axial depth (axis 1)
                    enface_proj = np.max(pred_01, axis=1)
                    mip_proj = cv2.resize((enface_proj * 255.0).astype(np.uint8), (256, 256), interpolation=cv2.INTER_LINEAR)
                    iio.imwrite(str(res_path / f"{key}_mip.png"), mip_proj)

                # Quantitative evaluation against ground truth OCT volume if available
                if gt_path_obj is not None:
                    gt_file = gt_path_obj / f"{key}.npy"
                    if gt_file.exists():
                        try:
                            gt_vol = np.load(gt_file)
                            if gt_vol.shape[1] > 256:
                                gt_vol = gt_vol[:, 100:356, :]
                            gt_01 = resize(gt_vol, (128, 256, 256)).astype(np.float32)

                            p_t = torch.from_numpy(pred_01[None, None]).float().to(device)
                            g_t = torch.from_numpy(gt_01[None, None]).float().to(device)

                            psnr_val = psnr_metric(p_t, g_t).item()
                            ssim3d_val = ssim3d_metric(p_t, g_t).item()
                            ssim2d_val = float(np.mean([ssim2d_metric(p_t[:, :, i], g_t[:, :, i]).item() for i in range(128)]))

                            rec = {
                                "Key": key,
                                "PSNR": psnr_val,
                                "SSIM3D": ssim3d_val,
                                "SSIM2D": ssim2d_val,
                            }

                            if lpips_metric is not None:
                                lpips_val = compute_volume_lpips(pred_01, gt_01, lpips_metric, device=device)
                                rec["LPIPS"] = lpips_val

                            sdif_res = compute_sdif(pred_01, gt_01, axis=0)
                            rec["SDIF"] = sdif_res["SDIF"]

                            if ves_model is not None:
                                pred_ves = extract_vessels_oct(pred_01, ves_model, device=device)
                                gt_ves = extract_vessels_oct(gt_01, ves_model, device=device)
                                v_metrics = compute_vessel_metrics(pred_ves, gt_ves)
                                rec["Vessel_Dice"] = v_metrics["Vessel_Dice"]
                                rec["Vessel_IoU"] = v_metrics["Vessel_IoU"]
                                rec["Vessel_Soft_Dice"] = v_metrics["Vessel_Soft_Dice"]

                            # Update FID statistics across slices
                            if fid_inception is not None or fid_retfound is not None:
                                p_slices = p_t[0].transpose(1, 0)
                                g_slices = g_t[0].transpose(1, 0)
                                if fid_inception is not None:
                                    fid_inception.update(p_slices.repeat(1, 3, 1, 1), real=False)
                                    fid_inception.update(g_slices.repeat(1, 3, 1, 1), real=True)
                                if fid_retfound is not None:
                                    fid_retfound.update(p_slices, real=False)
                                    fid_retfound.update(g_slices, real=True)

                            eval_records.append(rec)
                            log_msg = f"Eval {key}: PSNR={psnr_val:.4f}, SSIM3D={ssim3d_val:.4f}, SSIM2D={ssim2d_val:.4f}"
                            if "LPIPS" in rec:
                                log_msg += f", LPIPS={rec['LPIPS']:.4f}"
                            if "SDIF" in rec:
                                log_msg += f", SDIF={rec['SDIF']:.6f}"
                            if "Vessel_Dice" in rec:
                                log_msg += f", Vessel_Dice={rec['Vessel_Dice']:.4f}"
                            logging.info(log_msg)
                        except Exception as eval_err:
                            logging.error(f"Error evaluating {key} against GT: {eval_err}")
                    else:
                        logging.warning(f"GT volume not found for {key} at {gt_file}")

                del x, y_hat
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            processed_keys.append(key)

        except Exception as e:
            logging.error(f"Error processing {key} ({img_path}): {e}")
            continue

    # Save metadata CSV
    if metadata:
        meta_df = pd.DataFrame(metadata)
        meta_out_path = res_path / "fundus_crops_metadata.csv"
        meta_df.to_csv(meta_out_path, index=False)
        logging.info(f"Saved crops metadata to {meta_out_path}")

    # Save evaluation summary table if evaluation was run
    if eval_records:
        eval_df = pd.DataFrame(eval_records)
        num_evaluated = len(eval_df)
        mean_row = {"Key": "MEAN"}
        std_row = {"Key": "STD"}
        numeric_cols = [c for c in eval_df.columns if c != "Key"]
        for c in numeric_cols:
            mean_row[c] = float(eval_df[c].mean())
            std_row[c] = float(eval_df[c].std(ddof=1)) if num_evaluated > 1 else 0.0

        fid_rows = []
        if fid_inception is not None:
            try:
                fid_inc_score = fid_inception.compute().item()
                logging.info(f"FID (Inception): {fid_inc_score:.4f}")
                fid_rows.append({"Key": "FID_Inception", "PSNR": fid_inc_score})
            except Exception as e:
                logging.warning(f"Could not compute Inception FID: {e}")

        if fid_retfound is not None:
            try:
                fid_ret_score = fid_retfound.compute().item()
                logging.info(f"FID (RETFound) : {fid_ret_score:.4f}")
                fid_rows.append({"Key": "FID_RETFound", "PSNR": fid_ret_score})
            except Exception as e:
                logging.warning(f"Could not compute RETFound FID: {e}")

        summary_dfs = [eval_df, pd.DataFrame([mean_row, std_row])]
        if fid_rows:
            summary_dfs.append(pd.DataFrame(fid_rows))
        eval_summary_df = pd.concat(summary_dfs, ignore_index=True)
        eval_csv_path = res_path / "eval_results.csv"
        eval_summary_df.to_csv(eval_csv_path, index=False)
        logging.info(f"Saved evaluation results table to {eval_csv_path}")

        logging.info(f"--- Quantitative Evaluation Summary ({num_evaluated} cases) ---")
        for c in numeric_cols:
            logging.info(f"{c:16s}: {mean_row[c]:.4f} +/- {std_row[c]:.4f}")
        for f_row in fid_rows:
            logging.info(f"{f_row['Key']:16s}: {f_row['PSNR']:.4f}")


    logging.info(f"Successfully processed {len(processed_keys)}/{len(fundus_inputs)} cases.")
    return {
        "num_processed": len(processed_keys),
        "keys": processed_keys,
        "eval_records": eval_records
    }



def run_cli():
    """Command-line interface for eval_model_only_fundus."""
    parser = argparse.ArgumentParser(
        description="Evaluate 3D OCT synthesis directly from fundus images or export fundus crops for pre-processing (no OCT required)."
    )
    # Positional fallbacks for intuitive usage
    parser.add_argument("pos_fundus_path", nargs="?", default=None, type=str, help="(Optional positional) Path to fundus image, directory, or CSV file.")
    parser.add_argument("pos_res_path", nargs="?", default=None, type=str, help="(Optional positional) Output directory to save results.")

    # Explicit arguments
    parser.add_argument("--fundus_path", "-f", type=str, default=None, help="Path to fundus image (.png, .jpg, .dcm), directory, or CSV split file.")
    parser.add_argument("--fundus_dir", type=str, default=None, help="Directory containing fundus images if keys/paths are given in a CSV file.")
    parser.add_argument("--res_path", "-r", type=str, default=None, help="Output directory to save results.")

    # Crop parameters
    parser.add_argument("--crop_only", action="store_true", help="Only generate and save fundus crops for further pre-processing (skips 3D model synthesis).")
    parser.add_argument("--crop_size", type=int, default=850, help="Output dimension for square fundus crop in pixels (default: 850).")
    parser.add_argument("--crop_radius", type=int, default=None, help="Radius in pixels around fovea to crop (default: crop_size // 2, e.g. 425 for 850x850 window matching OCT macula field of view).")
    parser.add_argument("--no_save_crops", action="store_true", help="Do not save fundus crops when running 3D synthesis.")
    parser.add_argument("--save_crops_dir", type=str, default=None, help="Custom directory to save crops (default: res_path/crops or res_path if --crop_only).")

    # Fovea detection parameters
    parser.add_argument("--no_fovea_detect", action="store_true", help="Disable deep-learning fovea detector; fallback to image center.")
    parser.add_argument("--fovea_coords", type=float, nargs=2, default=None, help="Manual fovea coordinates: --fovea_coords X Y.")

    # Model synthesis parameters
    parser.add_argument("--ckpt_path", "-c", type=str, default=None, help="Path to model checkpoint (.ckpt file or directory containing checkpoints).")
    parser.add_argument("--model_name", "-m", type=str, default="mmse", choices=["mmse", "mmse_model", "flow_model", "flow", "rflow"], help="Synthesis model type (default: mmse).")
    parser.add_argument("--mmse_path", type=str, default=None, help="Optional checkpoint path for auxiliary MMSE guidance model.")
    parser.add_argument("--T", type=int, default=10, help="Number of flow integration steps (default: 10).")
    parser.add_argument("--noise_type", type=str, default=C.NOISE_TYPE_NORMAL, choices=[C.NOISE_TYPE_NORMAL, C.NOISE_TYPE_CONSTANT, C.NOISE_TYPE_SLERP], help="Noise distribution for flow model.")
    parser.add_argument("--save_mip", action="store_true", help="Save maximum intensity projection (MIP) preview PNG for synthesized volumes.")
    parser.add_argument("--gt_vol_dir", type=str, default=None, help="Directory containing ground truth OCT volumes (.npy) for quantitative evaluation.")
    parser.add_argument("--eval_vessels", action="store_true", help="Compute vessel segmentation metrics (Dice, IoU) using pre-trained vessel model.")
    parser.add_argument("--eval_fid", action="store_true", default=True, help="Compute FID score against ground truth OCT volume slices (default: True).")
    parser.add_argument("--no_fid", dest="eval_fid", action="store_false", help="Disable FID computation.")
    parser.add_argument("--fid_feature", type=str, default="all", choices=["all", "inception", "retfound", "auto"], help="Feature extractor for FID calculation (default: all).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on ('cuda' or 'cpu').")

    args = parser.parse_args()

    target_fundus_path = args.fundus_path or args.pos_fundus_path
    target_res_path = args.res_path or args.pos_res_path

    if target_fundus_path is None:
        parser.error("Please provide --fundus_path (or positional fundus_path).")
    if target_res_path is None:
        parser.error("Please provide --res_path (or positional res_path).")

    # Collect inputs
    fundus_inputs = collect_fundus_inputs(
        fundus_path=target_fundus_path,
        fundus_dir=args.fundus_dir
    )
    if not fundus_inputs:
        logging.error(f"No valid fundus images found for {target_fundus_path}.")
        return

    run_fundus_synthesis(
        fundus_inputs=fundus_inputs,
        res_path=target_res_path,
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        mmse_path=args.mmse_path,
        crop_only=args.crop_only,
        crop_radius=args.crop_radius,
        crop_size=args.crop_size,
        save_crops=not args.no_save_crops,
        save_crops_dir=args.save_crops_dir,
        save_mip=args.save_mip,
        fovea_coords=tuple(args.fovea_coords) if args.fovea_coords is not None else None,
        detect_fovea=not args.no_fovea_detect,
        T=args.T,
        noise_type=args.noise_type,
        device=args.device,
        gt_vol_dir=args.gt_vol_dir,
        eval_vessels=args.eval_vessels,
        eval_fid=args.eval_fid,
        fid_feature=args.fid_feature
    )




if __name__ == "__main__":
    run_cli()
