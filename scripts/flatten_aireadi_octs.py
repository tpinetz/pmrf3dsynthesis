"""Batch extract and flatten AI-READI OCT volumes using Botond's retinal layer segmentations."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmrf3dsynthesis.data.flattening import extract_and_flatten_oct

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_CROPPED_CSV = Path("/home/optima/tpinetz/data/AI_ready_processed/cropped_cfp_to_oct_pairs.csv")
DEFAULT_ALL_CSV = Path("/home/optima/tpinetz/data/AI_ready_processed/all_cfp_to_oct_pairs.csv")
DEFAULT_SEG_ROOT = Path("/home/thomas-pinetz/tools/botonds_segmentation_results/AI_ready")
DEFAULT_OUT_DIR = Path("/home/optima/tpinetz/data/AI_ready_processed/flattened_volumes")
DEFAULT_CSV_VOL_DIR = Path("/home/thomas-pinetz/data/AI_ready_processed/flattened_volumes")


def _process_one_oct(task: Tuple[str, str, str, int, bool]) -> Tuple[str, str, Optional[str]]:
    """Worker function to extract, flatten, and atomically save one OCT volume."""
    oct_path_str, seg_root_str, out_dir_str, final_res, overwrite = task
    oct_file = Path(oct_path_str)
    person_id = oct_file.parent.name
    oct_stem = oct_file.stem
    seg_dir = Path(seg_root_str) / person_id / oct_stem
    out_dir = Path(out_dir_str)
    out_file = out_dir / f"{oct_stem}.npy"

    if not overwrite and out_file.exists() and out_file.stat().st_size > 0:
        return oct_stem, "skipped", None

    tmp_file = out_dir / f".tmp_{oct_stem}_{os.getpid()}.npy"
    try:
        flattened = extract_and_flatten_oct(oct_file, seg_dir, final_res=final_res)
        np.save(tmp_file, flattened)
        os.replace(tmp_file, out_file)
        return oct_stem, "processed", None
    except Exception as exc:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        return oct_stem, "failed", str(exc)


def _atomic_write_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Writes a DataFrame atomically to CSV without corrupting the target on failure."""
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


def _update_csv_flattened_paths(
    csv_path: Path,
    out_dir: Path,
    csv_vol_dir: Path,
    only_existing: bool = False,
) -> int:
    """Adds/updates `flattened_oct_path` column in a pairs CSV and returns count of populated rows."""
    df = pd.read_csv(csv_path)
    existing_stems = {
        p.stem for p in out_dir.glob("*.npy") if p.stat().st_size > 0
    } if only_existing else None

    def resolve_flat_path(oct_val: object) -> Optional[str]:
        if not isinstance(oct_val, str) or not oct_val.strip():
            return None
        stem = Path(oct_val).stem
        if existing_stems is not None and stem not in existing_stems:
            return None
        return str(csv_vol_dir / f"{stem}.npy")

    df["flattened_oct_path"] = df["oct_path"].apply(resolve_flat_path)
    _atomic_write_csv(df, csv_path)
    return int(df["flattened_oct_path"].notna().sum())


def collect_unique_oct_paths(csv_path: Path) -> List[str]:
    """Extracts ordered unique non-null `oct_path` values from the pairs CSV."""
    df = pd.read_csv(csv_path)
    seen = set()
    unique_paths = []
    for val in df["oct_path"].dropna():
        val_str = str(val).strip()
        if val_str and val_str not in seen:
            seen.add(val_str)
            unique_paths.append(val_str)
    return unique_paths


def main(
    csv_path: Path,
    all_csv_path: Optional[Path],
    seg_root: Path,
    out_dir: Path,
    csv_vol_dir: Path,
    final_res: int = 512,
    workers: int = 8,
    overwrite: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    oct_paths = collect_unique_oct_paths(csv_path)
    logging.info("Found %d unique OCT volumes in %s", len(oct_paths), csv_path)

    tasks = [
        (oct_p, str(seg_root), str(out_dir), final_res, overwrite)
        for oct_p in oct_paths
    ]

    counts: Dict[str, int] = {"processed": 0, "skipped": 0, "failed": 0}
    failures: List[Tuple[str, str]] = []

    if workers <= 1:
        for task in tqdm(tasks, desc="Flattening OCT volumes"):
            stem, status, err = _process_one_oct(task)
            counts[status] += 1
            if err is not None:
                failures.append((stem, err))
                logging.error("Failed %s: %s", stem, err)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_one_oct, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Flattening OCT volumes"):
                stem, status, err = fut.result()
                counts[status] += 1
                if err is not None:
                    failures.append((stem, err))
                    logging.error("Failed %s: %s", stem, err)

    logging.info(
        "Extraction complete: processed=%d, skipped=%d, failed=%d",
        counts["processed"],
        counts["skipped"],
        counts["failed"],
    )

    if failures:
        sample = "; ".join(f"{s}: {e}" for s, e in failures[:5])
        raise RuntimeError(f"{len(failures)} OCT volume(s) failed to flatten. Sample errors: {sample}")

    crop_populated = _update_csv_flattened_paths(csv_path, out_dir, csv_vol_dir, only_existing=False)
    logging.info("Updated %s with flattened_oct_path (%d populated rows).", csv_path, crop_populated)

    if all_csv_path is not None and all_csv_path.exists():
        all_populated = _update_csv_flattened_paths(all_csv_path, out_dir, csv_vol_dir, only_existing=True)
        logging.info("Updated %s with flattened_oct_path (%d populated rows).", all_csv_path, all_populated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and flatten AI-READI OCT volumes.")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CROPPED_CSV, help="Path to cropped pairs CSV.")
    parser.add_argument("--all_csv_path", type=Path, default=DEFAULT_ALL_CSV, help="Path to all pairs CSV.")
    parser.add_argument("--seg_root", type=Path, default=DEFAULT_SEG_ROOT, help="Path to Botond segmentation root.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for flattened .npy files.")
    parser.add_argument(
        "--csv_vol_dir",
        type=Path,
        default=None,
        help="Path prefix recorded in CSV flattened_oct_path (defaults to symlinked /home/thomas-pinetz/data/... when using default out_dir).",
    )
    parser.add_argument("--final_res", type=int, default=512, help="Final axial resolution after flattening.")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing .npy files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    csv_vol_dir = args.csv_vol_dir
    if csv_vol_dir is None:
        csv_vol_dir = DEFAULT_CSV_VOL_DIR if args.out_dir == DEFAULT_OUT_DIR else args.out_dir
    main(
        csv_path=args.csv_path,
        all_csv_path=args.all_csv_path,
        seg_root=args.seg_root,
        out_dir=args.out_dir,
        csv_vol_dir=csv_vol_dir,
        final_res=args.final_res,
        workers=args.workers,
        overwrite=args.overwrite,
    )
