"""Batch preprocess AI-READI Topcon Maestro2 pairs: 3/4-height OCT flattening + physical-unit CFP registration."""

import argparse
import gc
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import cv2
import fundus_image_toolbox as fit
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from skimage.transform import resize
import tifffile
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmrf3dsynthesis.cli_fundus_oct import get_physical_common_size, load_fundus_image
from pmrf3dsynthesis.data.flattening import extract_and_flatten_oct
from pmrf3dsynthesis.registration.registration import adapt_large_to_small
from pmrf3dsynthesis.vessel_segmentation.utils import extract_vessels_oct, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Suppress noisy pydicom JPEG2000 fallback logs when SimpleITK handles decoding
logging.getLogger("pydicom").setLevel(logging.ERROR)

DEFAULT_CSV = Path(
    "/home/thomas-pinetz/tools/botonds_segmentation_results/AI_ready_topcon/paired_macula_maestro2_only.csv"
)
DEFAULT_SEG_ROOT = Path(
    "/home/thomas-pinetz/tools/botonds_segmentation_results/AI_ready_topcon"
)
DEFAULT_FUNDUS_OUT = Path("/home/optima/tpinetz/data/AI_ready_processed/Fundus")
DEFAULT_OCT_OUT = Path("/home/optima/tpinetz/data/AI_ready_processed/flattened_volumes")
DEFAULT_OUTPUT_CSV = Path(
    "/home/optima/tpinetz/data/AI_ready_processed/maestro2_preprocessed_pairs.csv"
)
DEFAULT_LOCAL_CSV = Path(
    "/home/thomas-pinetz/tools/botonds_segmentation_results/AI_ready_topcon/maestro2_preprocessed_pairs.csv"
)


def compute_dice_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[float, float]:
    a = mask_a > 0
    b = mask_b > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    denom = float(a.sum() + b.sum())
    dice = (2.0 * inter / denom) if denom > 0 else 0.0
    iou = (inter / union) if union > 0 else 0.0
    return dice, iou


def atomic_write_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = csv_path.parent / f".tmp_{csv_path.name}_{os.getpid()}"
    try:
        df.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, csv_path)
    except Exception:
        if tmp_csv.exists():
            try:
                tmp_csv.unlink()
            except OSError:
                pass
        raise


def atomic_save_oct_tif(arr: np.ndarray, out_file: Path) -> None:
    tmp_file = out_file.parent / f".tmp_{out_file.stem}_{os.getpid()}.tif"
    try:
        tifffile.imwrite(str(tmp_file), arr.astype(np.uint8), compression="zlib")
        os.replace(tmp_file, out_file)
    except Exception:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        raise


def atomic_save_png(img: np.ndarray, out_file: Path) -> None:
    tmp_file = out_file.parent / f".tmp_{out_file.stem}_{os.getpid()}.png"
    try:
        imageio.imwrite(str(tmp_file), img.astype(np.uint8))
        os.replace(tmp_file, out_file)
    except Exception:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        raise


def flush_results(
    base_df: pd.DataFrame,
    records: Dict[str, Dict[str, Any]],
    local_csv: Path,
    output_csv: Optional[Path],
) -> None:
    extra_cols = [
        "cropped_cfp_path",
        "flattened_oct_path",
        "common_size_h",
        "common_size_w",
        "fovea_x",
        "fovea_y",
        "reg_scale",
        "reg_angle_deg",
        "vessel_dice",
        "vessel_iou",
        "status",
        "error",
    ]
    rows = []
    for _, row in base_df.iterrows():
        key = Path(str(row["oct_path"])).stem
        rec = records.get(key, {})
        out_row = row.to_dict()
        for col in extra_cols:
            out_row[col] = rec.get(col, None)
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    atomic_write_csv(out_df, local_csv)
    if output_csv is not None and output_csv != local_csv:
        try:
            atomic_write_csv(out_df, output_csv)
        except OSError as exc:
            logging.warning("Could not sync CSV to %s: %s", output_csv, exc)


def load_existing_records(local_csv: Path, output_csv: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for candidate in (local_csv, output_csv):
        if candidate is not None and candidate.exists() and candidate.stat().st_size > 0:
            try:
                prev_df = pd.read_csv(candidate)
                if "oct_path" not in prev_df.columns:
                    continue
                for _, r in prev_df.iterrows():
                    key = Path(str(r["oct_path"])).stem
                    if pd.notna(r.get("status")) and str(r.get("status")) in ("ok", "skipped"):
                        records[key] = r.to_dict()
                logging.info("Loaded %d previously completed records from %s", len(records), candidate)
                break
            except Exception as exc:
                logging.warning("Could not read checkpoint CSV %s: %s", candidate, exc)
    return records


def main(args: argparse.Namespace) -> None:
    args.fundus_out_dir.mkdir(parents=True, exist_ok=True)
    args.oct_out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    if args.limit is not None and args.limit > 0:
        df = df.iloc[: args.limit].copy()

    logging.info("Loaded %d Maestro2 pairs from %s", len(df), args.csv_path)
    logging.info("Fundus output dir: %s", args.fundus_out_dir)
    logging.info("Flattened OCT output dir: %s", args.oct_out_dir)

    records = load_existing_records(args.local_csv, args.output_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Loading registration & vessel segmentation models on %s...", device)
    fov_model, _ = fit.load_fovea_od_model("default", device=device, return_test_dataloader=False)
    ensemble = fit.load_segmentation_ensemble(device=device)
    ves_model = load_model().to(device)
    ves_model.eval()

    counts = {"processed": 0, "skipped": 0, "failed": 0}
    processed_since_flush = 0

    for idx, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Maestro2 preprocess"), start=1):
        pid = str(row.person_id)
        cfp_path = Path(str(row.raw_cfp_path))
        oct_path = Path(str(row.oct_path))
        cfp_stem = cfp_path.stem
        oct_stem = oct_path.stem
        seg_dir = args.seg_root / pid / oct_stem

        fundus_out_file = args.fundus_out_dir / f"{cfp_stem}.png"
        oct_out_file = args.oct_out_dir / f"{oct_stem}.tif"

        fundus_exists = fundus_out_file.exists() and fundus_out_file.stat().st_size > 0
        oct_exists = oct_out_file.exists() and oct_out_file.stat().st_size > 0

        if not args.overwrite and fundus_exists and oct_exists:
            if oct_stem not in records:
                records[oct_stem] = {
                    "cropped_cfp_path": str(fundus_out_file),
                    "flattened_oct_path": str(oct_out_file),
                    "status": "skipped",
                    "error": "",
                }
            else:
                records[oct_stem]["flattened_oct_path"] = str(oct_out_file)
            counts["skipped"] += 1
            continue

        try:
            # 1. Flatten OCT (or reuse existing flattened volume if only fundus is missing)
            if not args.overwrite and oct_exists:
                flattened_oct = tifffile.imread(str(oct_out_file))
            else:
                flattened_oct = extract_and_flatten_oct(
                    oct_path,
                    seg_dir,
                    final_res=args.final_res,
                    flatten_frac=args.flatten_frac,
                )
                atomic_save_oct_tif(flattened_oct, oct_out_file)

            # 2. Register CFP to flattened OCT using physical DICOM common_size
            if not args.overwrite and fundus_exists:
                rec = records.get(oct_stem, {})
                rec.update(
                    {
                        "cropped_cfp_path": str(fundus_out_file),
                        "flattened_oct_path": str(oct_out_file),
                        "status": "ok",
                        "error": "",
                    }
                )
                records[oct_stem] = rec
            else:
                oct_float = flattened_oct.astype(np.float32) / 255.0
                fundus_img = load_fundus_image(cfp_path)

                phys_size = get_physical_common_size(cfp_path, oct_path)
                if phys_size is not None:
                    phys_h, phys_w = phys_size
                else:
                    phys_h, phys_w = args.default_common_size, args.default_common_size

                with torch.no_grad():
                    fovea_x, fovea_y, _, _ = fov_model.predict(fundus_img)
                    fovea_x = int(np.round(fovea_x))
                    fovea_y = int(np.round(fovea_y))
                    moving_vessels = fit.ensemble_predict_segmentation(
                        ensemble, [fundus_img], threshold=0.5, size=(512, 512)
                    )
                    fixed_vessels = extract_vessels_oct(oct_float, ves_model, device=device)

                fixed_vessels_uint = (resize(fixed_vessels, (phys_h, phys_w)) * 255).astype(np.uint8)
                moving_vessels_uint = (moving_vessels * 255).astype(np.uint8)

                y0 = max(fovea_y - args.rad, 0)
                y1 = fovea_y + args.rad
                x0 = max(fovea_x - args.rad, 0)
                x1 = fovea_x + args.rad

                matched_vessels, M = adapt_large_to_small(
                    fixed_vessels_uint,
                    moving_vessels_uint[y0:y1, x0:x1],
                )
                cropped_fundus = cv2.warpAffine(
                    fundus_img[y0:y1, x0:x1], M, (phys_w, phys_h)
                )
                atomic_save_png(cropped_fundus, fundus_out_file)

                dice, iou = compute_dice_iou(fixed_vessels_uint > 127, matched_vessels > 127)
                scale = float(np.hypot(M[0, 0], M[1, 0]))
                angle_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))

                records[oct_stem] = {
                    "cropped_cfp_path": str(fundus_out_file),
                    "flattened_oct_path": str(oct_out_file),
                    "common_size_h": phys_h,
                    "common_size_w": phys_w,
                    "fovea_x": fovea_x,
                    "fovea_y": fovea_y,
                    "reg_scale": round(scale, 5),
                    "reg_angle_deg": round(angle_deg, 4),
                    "vessel_dice": round(dice, 5),
                    "vessel_iou": round(iou, 5),
                    "status": "ok",
                    "error": "",
                }

            counts["processed"] += 1
            processed_since_flush += 1

        except Exception as exc:
            logging.error("Failed %s (pid=%s): %s", oct_stem, pid, exc)
            records[oct_stem] = {
                "cropped_cfp_path": str(fundus_out_file) if fundus_out_file.exists() else "",
                "flattened_oct_path": str(oct_out_file) if oct_out_file.exists() else "",
                "status": "failed",
                "error": str(exc),
            }
            counts["failed"] += 1
            processed_since_flush += 1

        if processed_since_flush >= args.flush_every:
            flush_results(df, records, args.local_csv, args.output_csv)
            processed_since_flush = 0
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    flush_results(df, records, args.local_csv, args.output_csv)
    logging.info(
        "Finished Maestro2 preprocessing: processed=%d, skipped=%d, failed=%d",
        counts["processed"],
        counts["skipped"],
        counts["failed"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten Topcon Maestro2 OCTs (3/4 height) and register paired CFPs using physical DICOM units."
    )
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--seg_root", type=Path, default=DEFAULT_SEG_ROOT)
    parser.add_argument("--fundus_out_dir", type=Path, default=DEFAULT_FUNDUS_OUT)
    parser.add_argument("--oct_out_dir", type=Path, default=DEFAULT_OCT_OUT)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--local_csv", type=Path, default=DEFAULT_LOCAL_CSV)
    parser.add_argument("--final_res", type=int, default=512)
    parser.add_argument("--flatten_frac", type=float, default=0.75)
    parser.add_argument("--rad", type=int, default=700)
    parser.add_argument("--default_common_size", type=int, default=874)
    parser.add_argument("--flush_every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
