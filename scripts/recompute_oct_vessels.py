import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
from skimage.transform import resize
from tqdm import tqdm

from pmrf3dsynthesis.eval_model import compute_vessel_metrics, save_summary_table
from pmrf3dsynthesis.vessel_segmentation.utils import load_model, extract_vessels_oct
from scripts.extract_metrics import extract_metrics_from_directory


def precompute_gt_vessel_cache(
    keys: set,
    gt_vol_dir: Path,
    cache_dir: Path,
    ves_model: torch.nn.Module,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    Computes (or loads from disk cache) the 2D vessel segmentation maps for all ground truth volumes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    gt_vessels_cache = {}

    to_compute = []
    for k in sorted(list(keys)):
        cache_file = cache_dir / f"{k}_vessels.npy"
        if cache_file.exists():
            gt_vessels_cache[k] = np.load(cache_file)
        else:
            to_compute.append(k)

    print(f"GT Vessel Cache: {len(gt_vessels_cache)} loaded from disk, {len(to_compute)} to compute.")

    if to_compute:
        for k in tqdm(to_compute, desc="Precomputing GT vessel maps"):
            vol_file = gt_vol_dir / f"{k}.npy"
            if not vol_file.exists():
                print(f"Warning: Ground truth volume file not found: {vol_file}")
                continue

            raw_vol = np.load(vol_file)
            if raw_vol.shape[1] > 256:
                raw_vol = raw_vol[:, 100:356, :]
            vol_resized = resize(raw_vol, (128, 256, 256))

            vessels_map = extract_vessels_oct(vol_resized, ves_model, device=device)
            cache_file = cache_dir / f"{k}_vessels.npy"
            np.save(cache_file, vessels_map)
            gt_vessels_cache[k] = vessels_map

    return gt_vessels_cache


def recompute_vessels_for_runs(
    results_base: Path,
    gt_vol_dir: Path,
    cache_dir: Path,
    threshold: float = 0.5,
    device: str = "cuda",
    force_recompute: bool = False
):
    print(f"Initializing vessel model on {device}...")
    ves_model = load_model().to(device)
    ves_model.eval()

    # Find all runs with eval_results.pkl
    pkl_files = sorted(list(results_base.glob("**/eval_results.pkl")))
    print(f"Found {len(pkl_files)} evaluation runs in {results_base}\n")

    # Collect all unique keys
    all_keys = set()
    run_data_map = {}
    for p in pkl_files:
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            keys = list(data.get("PSNR", {}).keys())
            all_keys.update(keys)
            run_data_map[p] = data
        except Exception as e:
            print(f"Error reading {p}: {e}")

    print(f"Total unique keys across runs: {len(all_keys)}")

    # Precompute / load all GT vessel maps
    gt_cache = precompute_gt_vessel_cache(all_keys, gt_vol_dir, cache_dir, ves_model, device)

    # Now process each run
    print("\n--- Processing Runs ---")
    processed_count = 0

    for run_idx, (pkl_path, eval_results) in enumerate(run_data_map.items()):
        run_dir = pkl_path.parent
        run_rel = run_dir.relative_to(results_base)

        has_vessels = (
            not force_recompute
            and "Vessel_Dice" in eval_results
            and len(eval_results["Vessel_Dice"]) == len(eval_results.get("PSNR", {}))
            and len(eval_results["Vessel_Dice"]) > 0
        )
        if has_vessels:
            v_vals = list(eval_results["Vessel_Dice"].values())
            print(f"[{run_idx + 1}/{len(run_data_map)}] {run_rel}: Already has vessel metrics (Mean Dice: {np.mean(v_vals):.4f})")
            continue

        case_keys = list(eval_results.get("PSNR", {}).keys())
        if not case_keys:
            continue

        if "Vessel_Dice" not in eval_results:
            eval_results["Vessel_Dice"] = {}
            eval_results["Vessel_IoU"] = {}
            eval_results["Vessel_Soft_Dice"] = {}

        t0 = time.time()
        computed_cases = 0

        for k in case_keys:
            if k not in gt_cache:
                continue

            pred_file = run_dir / f"{k}.npy"
            if not pred_file.exists():
                continue

            # Load prediction
            pred_raw = np.load(pred_file)
            if pred_raw.min() < 0:
                pred_01 = np.clip(pred_raw * 0.5 + 0.5, 0.0, 1.0)
            else:
                pred_01 = np.clip(pred_raw, 0.0, 1.0)

            # Ensure resolution
            if pred_01.shape != (128, 256, 256):
                pred_01 = resize(pred_01, (128, 256, 256))

            pred_vessels = extract_vessels_oct(pred_01, ves_model, device=device)
            v_metrics = compute_vessel_metrics(pred_vessels, gt_cache[k], threshold=threshold)

            eval_results["Vessel_Dice"][k] = v_metrics["Vessel_Dice"]
            eval_results["Vessel_IoU"][k] = v_metrics["Vessel_IoU"]
            eval_results["Vessel_Soft_Dice"][k] = v_metrics["Vessel_Soft_Dice"]
            computed_cases += 1

        dt = time.time() - t0

        # Save updated eval_results.pkl
        with open(pkl_path, "wb") as f:
            pickle.dump(eval_results, f)

        # Update eval_results.csv
        save_summary_table(eval_results, run_dir)

        dices = list(eval_results["Vessel_Dice"].values())
        mean_dice = np.mean(dices) if dices else np.nan
        std_dice = np.std(dices, ddof=1) if len(dices) > 1 else 0.0
        print(f"[{run_idx + 1}/{len(run_data_map)}] {run_rel}: {computed_cases} cases in {dt:.1f}s -> Vessel Dice: {mean_dice:.4f} ± {std_dice:.4f}")
        processed_count += 1

    print(f"\nDone! Successfully updated {processed_count} runs.")

    # Print final summary table
    print("\n=======================================================")
    print("           UPDATED OCT METRICS SUMMARY TABLE           ")
    print("=======================================================")
    df = extract_metrics_from_directory(str(results_base), sort_by="vessel")
    if not df.empty:
        print("\n" + df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(description="Recompute vessel dice for finished OCT runs.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/media/thomas-pinetz/data/data/OCT/results",
        help="Base directory containing OCT evaluation results."
    )
    parser.add_argument(
        "--gt_vol_dir",
        type=str,
        default="/home/thomas-pinetz/data/OCT/retinsight/FundusMIP/flattened_volumes",
        help="Path to flattened OCT ground truth volumes."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/thomas-pinetz/data/OCT/vessel_gt_cache",
        help="Path to cache ground truth vessel maps."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary threshold for vessel segmentation."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run vessel model on."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if Vessel_Dice exists."
    )
    args = parser.parse_args()

    recompute_vessels_for_runs(
        results_base=Path(args.results_dir),
        gt_vol_dir=Path(args.gt_vol_dir),
        cache_dir=Path(args.cache_dir),
        threshold=args.threshold,
        device=args.device,
        force_recompute=args.force
    )


if __name__ == "__main__":
    main()
