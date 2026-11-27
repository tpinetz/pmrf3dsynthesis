"""Score AI-READI raw CFP image quality using the Fundus Image Toolbox (FIT) 10-model ensemble."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import fundus_image_toolbox as fit
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmrf3dsynthesis.cli_fundus_oct import load_fundus_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("pydicom").setLevel(logging.ERROR)

DEFAULT_CSV = Path(
    "/home/thomas-pinetz/tools/botonds_segmentation_results/AI_ready_topcon/paired_macula_maestro2_only.csv"
)
DEFAULT_OUTPUT_CSV = Path(
    "/media/thomas-pinetz/data/data/OCT/AI_ready_processed/maestro2_fundus_quality.csv"
)


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


def _load_one_cfp(path_str: str) -> Tuple[str, Optional[np.ndarray], Optional[str]]:
    try:
        img = load_fundus_image(Path(path_str))
        return path_str, img, None
    except Exception as exc:
        return path_str, None, str(exc)


def flush_quality_csv(
    base_df: pd.DataFrame,
    records: Dict[str, Dict[str, Any]],
    output_csv: Path,
) -> pd.DataFrame:
    cols = ["fundus_quality_score", "fundus_gradable", "quality_status", "quality_error"]
    rows = []
    for _, row in base_df.iterrows():
        key = str(row["raw_cfp_path"])
        rec = records.get(key, {})
        out_row = {
            "person_id": row["person_id"],
            "laterality": row["laterality"],
            "raw_cfp_path": key,
            "oct_path": row["oct_path"],
        }
        for c in cols:
            out_row[c] = rec.get(c, None)
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    atomic_write_csv(out_df, output_csv)
    return out_df


def merge_quality_into_csv(quality_df: pd.DataFrame, target_csv: Path) -> None:
    if not target_csv.exists():
        logging.warning("Merge target %s does not exist; skipping merge.", target_csv)
        return
    target_df = pd.read_csv(target_csv)
    q_cols = ["raw_cfp_path", "fundus_quality_score", "fundus_gradable"]
    q_sub = quality_df[q_cols].drop_duplicates(subset=["raw_cfp_path"])
    for c in ("fundus_quality_score", "fundus_gradable"):
        if c in target_df.columns:
            target_df = target_df.drop(columns=[c])
    merged = target_df.merge(q_sub, on="raw_cfp_path", how="left")
    atomic_write_csv(merged, target_csv)
    populated = int(merged["fundus_quality_score"].notna().sum())
    logging.info("Merged fundus quality into %s (%d / %d rows populated).", target_csv, populated, len(merged))


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.csv_path)
    if args.limit is not None and args.limit > 0:
        df = df.iloc[: args.limit].copy()

    records: Dict[str, Dict[str, Any]] = {}
    if not args.overwrite and args.output_csv.exists() and args.output_csv.stat().st_size > 0:
        prev_df = pd.read_csv(args.output_csv)
        for _, r in prev_df.iterrows():
            if pd.notna(r.get("fundus_quality_score")) and str(r.get("quality_status")) == "ok":
                records[str(r["raw_cfp_path"])] = {
                    "fundus_quality_score": float(r["fundus_quality_score"]),
                    "fundus_gradable": int(r["fundus_gradable"]),
                    "quality_status": "ok",
                    "quality_error": "",
                }
        logging.info("Loaded %d already-scored CFPs from %s", len(records), args.output_csv)

    pending_paths = [
        str(p) for p in df["raw_cfp_path"].tolist() if str(p) not in records
    ]
    logging.info("Total rows: %d | Pending to score: %d", len(df), len(pending_paths))

    if pending_paths:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Loading FIT quality ensemble on %s...", device)
        ensemble = fit.load_quality_ensemble(device=device)

        batches: List[List[str]] = [
            pending_paths[i : i + args.batch_size]
            for i in range(0, len(pending_paths), args.batch_size)
        ]

        scored_since_flush = 0
        with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
            for batch_paths in tqdm(batches, desc="FIT quality scoring"):
                loaded = list(pool.map(_load_one_cfp, batch_paths))
                valid_paths: List[str] = []
                valid_imgs: List[np.ndarray] = []

                for path_str, img, err in loaded:
                    if img is None:
                        logging.error("Failed loading CFP %s: %s", path_str, err)
                        records[path_str] = {
                            "fundus_quality_score": None,
                            "fundus_gradable": None,
                            "quality_status": "failed",
                            "quality_error": err or "unknown load error",
                        }
                    else:
                        valid_paths.append(path_str)
                        valid_imgs.append(img)

                if valid_imgs:
                    try:
                        with torch.no_grad():
                            confs, preds = fit.ensemble_predict_quality(
                                ensemble, valid_imgs, threshold=args.threshold
                            )
                        confs_arr = np.atleast_1d(confs)
                        preds_arr = np.atleast_1d(preds)
                        for path_str, conf, pred in zip(valid_paths, confs_arr, preds_arr):
                            records[path_str] = {
                                "fundus_quality_score": round(float(conf), 5),
                                "fundus_gradable": int(pred),
                                "quality_status": "ok",
                                "quality_error": "",
                            }
                    except Exception as exc:
                        logging.error("Batch inference failed, falling back to per-image: %s", exc)
                        for path_str, img in zip(valid_paths, valid_imgs):
                            try:
                                with torch.no_grad():
                                    conf, pred = fit.ensemble_predict_quality(
                                        ensemble, img, threshold=args.threshold
                                    )
                                records[path_str] = {
                                    "fundus_quality_score": round(float(conf), 5),
                                    "fundus_gradable": int(pred),
                                    "quality_status": "ok",
                                    "quality_error": "",
                                }
                            except Exception as img_exc:
                                records[path_str] = {
                                    "fundus_quality_score": None,
                                    "fundus_gradable": None,
                                    "quality_status": "failed",
                                    "quality_error": str(img_exc),
                                }

                scored_since_flush += len(batch_paths)
                if scored_since_flush >= args.flush_every:
                    flush_quality_csv(df, records, args.output_csv)
                    scored_since_flush = 0
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

    out_df = flush_quality_csv(df, records, args.output_csv)
    ok_mask = out_df["quality_status"] == "ok"
    n_ok = int(ok_mask.sum())
    n_gradable = int((out_df.loc[ok_mask, "fundus_gradable"] == 1).sum()) if n_ok > 0 else 0
    mean_score = float(out_df.loc[ok_mask, "fundus_quality_score"].mean()) if n_ok > 0 else 0.0
    logging.info(
        "Quality scoring complete: %d/%d ok | gradable=%d (%.1f%%) | mean_score=%.4f",
        n_ok,
        len(out_df),
        n_gradable,
        100.0 * n_gradable / max(n_ok, 1),
        mean_score,
    )

    if args.merge_into is not None:
        merge_quality_into_csv(out_df, args.merge_into)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FIT fundus image quality scores for AI-READI CFPs and save/merge to CSV."
    )
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--merge_into",
        type=Path,
        default=None,
        help="Optional target CSV (e.g. maestro2_preprocessed_pairs.csv) to left-join quality columns into.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--io_workers", type=int, default=4)
    parser.add_argument("--flush_every", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
