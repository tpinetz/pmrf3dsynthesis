import argparse
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def extract_metrics_from_directory(
    results_dir: str,
    output_csv: Optional[str] = None,
    sort_by: str = "name"
) -> pd.DataFrame:
    """
    Loops through subdirectories of results_dir, loads eval_results.pkl or eval_results.csv,
    and summarizes PSNR, SSIM, LPIPS, SDIF, Vessel Dice, FID, and case count N.
    """
    base_path = Path(results_dir).expanduser()
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    rows = []
    # Search for eval_results.pkl in all subdirectories
    pkl_files = sorted(list(base_path.glob("**/eval_results.pkl")))

    for pkl_path in pkl_files:
        run_name = str(pkl_path.parent.relative_to(base_path))
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            psnr_dict = data.get("PSNR", {})
            ssim_dict = data.get("SSIM", {})
            lpips_dict = data.get("LPIPS", {})
            sdif_dict = data.get("SDIF", {})
            fid_list = data.get("FID", [])
            vessel_dice_dict = data.get("Vessel_Dice", {})

            psnr_vals = [float(v) for v in psnr_dict.values() if np.isfinite(v)]
            ssim_vals = [float(v) for v in ssim_dict.values() if np.isfinite(v)]
            lpips_vals = [float(v) for v in lpips_dict.values() if np.isfinite(v)]
            sdif_vals = [float(v) for v in sdif_dict.values() if np.isfinite(v)]
            vessel_vals = [float(v) for v in vessel_dice_dict.values() if np.isfinite(v)]

            psnr_mean = np.mean(psnr_vals) if psnr_vals else np.nan
            psnr_std = np.std(psnr_vals, ddof=1) if len(psnr_vals) > 1 else 0.0

            ssim_mean = np.mean(ssim_vals) if ssim_vals else np.nan
            ssim_std = np.std(ssim_vals, ddof=1) if len(ssim_vals) > 1 else 0.0

            lpips_mean = np.mean(lpips_vals) if lpips_vals else np.nan
            lpips_std = np.std(lpips_vals, ddof=1) if len(lpips_vals) > 1 else 0.0

            sdif_mean = np.mean(sdif_vals) if sdif_vals else np.nan
            sdif_std = np.std(sdif_vals, ddof=1) if len(sdif_vals) > 1 else 0.0

            vessel_mean = np.mean(vessel_vals) if vessel_vals else np.nan
            vessel_std = np.std(vessel_vals, ddof=1) if len(vessel_vals) > 1 else 0.0

            fid_val = np.nan
            if isinstance(fid_list, list) and len(fid_list) > 0:
                fid_val = float(fid_list[0])
            elif isinstance(fid_list, (int, float)):
                fid_val = float(fid_list)

            rows.append({
                "Run / Model": run_name,
                "PSNR_mean": psnr_mean,
                "PSNR_std": psnr_std,
                "SSIM_mean": ssim_mean,
                "SSIM_std": ssim_std,
                "LPIPS_mean": lpips_mean,
                "LPIPS_std": lpips_std,
                "SDIF_mean": sdif_mean,
                "SDIF_std": sdif_std,
                "Vessel_Dice_mean": vessel_mean,
                "Vessel_Dice_std": vessel_std,
                "FID": fid_val,
                "N": len(psnr_vals)
            })
        except Exception as e:
            print(f"Warning: Could not process {pkl_path}: {e}")

    df_raw = pd.DataFrame(rows)
    if df_raw.empty:
        print("No evaluation runs found.")
        return df_raw

    # Sort
    if sort_by == "psnr":
        df_raw = df_raw.sort_values(by="PSNR_mean", ascending=False)
    elif sort_by == "ssim":
        df_raw = df_raw.sort_values(by="SSIM_mean", ascending=False)
    elif sort_by == "lpips":
        df_raw = df_raw.sort_values(by="LPIPS_mean", ascending=True)
    elif sort_by == "sdif":
        df_raw = df_raw.sort_values(by="SDIF_mean", ascending=True)
    elif sort_by == "vessel":
        df_raw = df_raw.sort_values(by="Vessel_Dice_mean", ascending=False)
    elif sort_by == "fid":
        df_raw = df_raw.sort_values(by="FID", ascending=True)
    else:
        df_raw = df_raw.sort_values(by="Run / Model")

    # Formatted display table
    df_display = pd.DataFrame()
    df_display["Run / Model"] = df_raw["Run / Model"]
    df_display["PSNR (dB)"] = df_raw.apply(
        lambda r: f"{r['PSNR_mean']:.2f} ± {r['PSNR_std']:.2f}" if pd.notnull(r["PSNR_mean"]) else "N/A", axis=1
    )
    df_display["SSIM"] = df_raw.apply(
        lambda r: f"{r['SSIM_mean']:.4f} ± {r['SSIM_std']:.4f}" if pd.notnull(r["SSIM_mean"]) else "N/A", axis=1
    )
    if "LPIPS_mean" in df_raw.columns and df_raw["LPIPS_mean"].notnull().any():
        df_display["LPIPS ↓"] = df_raw.apply(
            lambda r: f"{r['LPIPS_mean']:.4f} ± {r['LPIPS_std']:.4f}" if pd.notnull(r["LPIPS_mean"]) else "N/A", axis=1
        )
    if "SDIF_mean" in df_raw.columns and df_raw["SDIF_mean"].notnull().any():
        df_display["SDIF ↓"] = df_raw.apply(
            lambda r: f"{r['SDIF_mean']:.6f} ± {r['SDIF_std']:.6f}" if pd.notnull(r["SDIF_mean"]) else "N/A", axis=1
        )
    if "Vessel_Dice_mean" in df_raw.columns and df_raw["Vessel_Dice_mean"].notnull().any():
        df_display["Vessel Dice"] = df_raw.apply(
            lambda r: f"{r['Vessel_Dice_mean']:.4f} ± {r['Vessel_Dice_std']:.4f}" if pd.notnull(r["Vessel_Dice_mean"]) else "N/A", axis=1
        )
    df_display["FID ↓"] = df_raw["FID"].apply(lambda v: f"{v:.2f}" if pd.notnull(v) else "N/A")
    df_display["N"] = df_raw["N"]

    if output_csv:
        out_path = Path(output_csv).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(out_path, index=False)
        print(f"Saved raw metrics table to: {out_path}")

    return df_display


def main():
    parser = argparse.ArgumentParser(description="Extract evaluation metrics from a results directory.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/media/thomas-pinetz/data/data/CT/results",
        help="Path to results directory containing run subdirectories."
    )
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path to save metrics CSV.")
    parser.add_argument(
        "--sort_by",
        type=str,
        choices=["name", "psnr", "ssim", "lpips", "sdif", "vessel", "fid"],
        default="name",
        help="Column to sort runs by."
    )
    args = parser.parse_args()

    df = extract_metrics_from_directory(args.results_dir, args.output_csv, args.sort_by)
    if not df.empty:
        print("\n" + df.to_markdown(index=False))


if __name__ == "__main__":
    main()
