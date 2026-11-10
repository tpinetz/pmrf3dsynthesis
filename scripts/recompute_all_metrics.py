import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from pmrf3dsynthesis.eval_model import compute_sdif, compute_volume_lpips, load_sample, save_summary_table
from scripts.extract_metrics import extract_metrics_from_directory


def recompute_modality_metrics(
    modality: str,
    results_root: Path,
    device: str = "cuda",
    batch_size: int = 32
):
    print(f"\n=======================================================")
    print(f"Recomputing SDIF and LPIPS for {modality.upper()} Runs")
    print(f"Results Directory: {results_root}")
    print(f"=======================================================")

    # 1. Find all runs with eval_results.pkl
    pkl_paths = sorted(list(results_root.glob("**/eval_results.pkl")))
    if not pkl_paths:
        print(f"No eval_results.pkl found in {results_root}")
        return

    print(f"Found {len(pkl_paths)} finished runs to re-evaluate.")

    # 2. Collect all unique case keys across all runs
    all_keys = set()
    for pkl_path in pkl_paths:
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            all_keys.update(data.get("PSNR", {}).keys())
        except Exception:
            pass

    sorted_keys = sorted(list(all_keys))
    print(f"Identified {len(sorted_keys)} unique test cases. Pre-loading Ground Truth volumes...")

    # 3. Cache all Ground Truth volumes and precompute GT SDiff
    gt_cache: Dict[str, np.ndarray] = {}
    gt_sdiff_cache: Dict[str, float] = {}

    for key in tqdm(sorted_keys, desc="Loading GT volumes"):
        try:
            _, gt_vol = load_sample(key, modality)
            gt_vol = np.clip(gt_vol.astype(np.float32), 0.0, 1.0)
            gt_cache[key] = gt_vol
            gt_sdiff_cache[key] = float(np.mean(np.abs(np.diff(gt_vol, axis=0))))
        except Exception as e:
            print(f"Warning: Could not load GT for {key}: {e}")

    print(f"Cached {len(gt_cache)} ground truth volumes in RAM.")

    # 4. Initialize LPIPS metric on GPU
    print(f"Initializing LPIPS AlexNet on {device}...")
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    lpips_fn.eval()

    # 5. Recompute metrics for each run
    for run_idx, pkl_path in enumerate(pkl_paths):
        run_dir = pkl_path.parent
        run_name = str(run_dir.relative_to(results_root))
        print(f"\n[{run_idx + 1}/{len(pkl_paths)}] Processing: {run_name}")

        try:
            with open(pkl_path, "rb") as f:
                eval_data = pickle.load(f)

            if "LPIPS" not in eval_data:
                eval_data["LPIPS"] = {}
            if "SDIF" not in eval_data:
                eval_data["SDIF"] = {}

            keys_in_run = list(eval_data.get("PSNR", {}).keys())
            computed_cases = 0

            t0 = time.time()
            for key in tqdm(keys_in_run, desc=f"Evaluating {run_dir.name[:30]}", leave=False):
                pred_path = run_dir / f"{key}.npy"
                if not pred_path.exists():
                    continue

                if key not in gt_cache:
                    continue

                gt_vol = gt_cache[key]
                pred_vol = np.load(pred_path).astype(np.float32)

                # Ensure [0, 1] range
                if pred_vol.min() < 0:
                    pred_vol = np.clip(pred_vol * 0.5 + 0.5, 0.0, 1.0)
                else:
                    pred_vol = np.clip(pred_vol, 0.0, 1.0)

                # 1. SDIF (Slice Difference gap)
                pred_sdiff = float(np.mean(np.abs(np.diff(pred_vol, axis=0))))
                sdif_gap = float(np.abs(pred_sdiff - gt_sdiff_cache[key]))
                eval_data["SDIF"][key] = sdif_gap

                # 2. LPIPS perceptual similarity
                lpips_score = compute_volume_lpips(pred_vol, gt_vol, lpips_fn, device=device, batch_size=batch_size)
                eval_data["LPIPS"][key] = lpips_score

                computed_cases += 1

            # Save updated pickle and CSV
            with open(pkl_path, "wb") as f:
                pickle.dump(eval_data, f)

            save_summary_table(eval_data, run_dir)

            elapsed = time.time() - t0
            lpips_vals = list(eval_data["LPIPS"].values())
            sdif_vals = list(eval_data["SDIF"].values())
            print(
                f"  -> Done in {elapsed:.1f}s ({computed_cases} cases). "
                f"LPIPS: {np.mean(lpips_vals):.4f} ± {np.std(lpips_vals, ddof=1):.4f} | "
                f"SDIF: {np.mean(sdif_vals):.6f} ± {np.std(sdif_vals, ddof=1):.6f}"
            )

        except Exception as e:
            print(f"Error processing {run_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Recompute SDIF and LPIPS for all finished OCT and CT runs.")
    parser.add_argument("--modality", choices=["oct", "ct", "both"], default="both", help="Which modality to process.")
    parser.add_argument("--oct_dir", type=str, default="/media/thomas-pinetz/data/data/OCT/results")
    parser.add_argument("--ct_dir", type=str, default="/media/thomas-pinetz/data/data/CT/results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.modality in ["oct", "both"]:
        recompute_modality_metrics("oct", Path(args.oct_dir), device=args.device, batch_size=args.batch_size)
        print("\n--- Summary Table for OCT Runs (Sorted by LPIPS) ---")
        df_oct = extract_metrics_from_directory(args.oct_dir, sort_by="lpips")
        print(df_oct.to_markdown(index=False))

    if args.modality in ["ct", "both"]:
        recompute_modality_metrics("ct", Path(args.ct_dir), device=args.device, batch_size=args.batch_size)
        print("\n--- Summary Table for CT Runs (Sorted by LPIPS) ---")
        df_ct = extract_metrics_from_directory(args.ct_dir, sort_by="lpips")
        print(df_ct.to_markdown(index=False))


if __name__ == "__main__":
    main()
