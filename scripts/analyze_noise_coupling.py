#!/usr/bin/env python3
"""
Statistical analysis and reporting for the controlled noise-coupling ablation experiment.
Aggregates metrics across seeds, computes summary statistics (mean +/- std, median),
performs paired Wilcoxon signed-rank tests with Holm-Bonferroni correction,
calculates 95% bootstrap confidence intervals, and formats comparison tables.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRIC_PRETTY_NAMES = {
    "Adjacent_SSIM_Gap": "Adjacent SSIM Gap (|Δ|) ↓",
    "Log_Spectral_Distance": "Log-Spectral Distance ↓",
    "Orthogonal_SSIM_mean": "Orthogonal SSIM ↑",
    "Orthogonal_LPIPS_mean": "Orthogonal LPIPS ↓",
    "SDIF": "SDIF Gap (|Δ|) ↓",
    "PSNR": "PSNR (dB) ↑",
    "SSIM": "In-Plane SSIM ↑",
    "LPIPS": "In-Plane LPIPS ↓",
    "Vessel_Dice": "Vessel Dice ↑",
    "FID": "FID ↓",
    "Flow_Displacement_L1": "Flow Displacement (L1)",
}

ARM_PRETTY_NAMES = {
    "normal": "A: Independent (Normal)",
    "constant": "B: Shared (Constant)",
    "linear_vm": "C: Variance-Matched Linear",
    "slerp": "D: SLERP",
    "linear": "E: Plain Linear (No VM)",
}


def bootstrap_ci(data: np.ndarray, num_boot: int = 2000, ci: float = 0.95) -> Tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval for the mean."""
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return (np.nan, np.nan)
    n = len(data)
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(num_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return (low, high)


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = p * (m - rank)
        adj_p = min(max(adj_p, running_max), 1.0)
        running_max = adj_p
        adjusted[orig_idx] = adj_p

    return adjusted


def load_and_aggregate_results(raw_path: Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    """
    Loads raw ablation results dictionary: arm -> seed -> metric -> key -> value.
    Averages metrics across seeds per case.
    Returns:
      per_arm_dfs: dict mapping arm -> DataFrame of per-case averaged metrics
      per_arm_fids: dict mapping arm -> mean FID score across seeds
    """
    with open(raw_path, "rb") as f:
        data = pickle.load(f)

    arms = list(data.keys())
    per_arm_dfs = {}
    per_arm_fids = {}

    for arm in arms:
        seeds = list(data[arm].keys())
        # First gather all metric names with data
        metric_names = [m for m in data[arm][seeds[0]].keys() if m != "FID" and len(data[arm][seeds[0]][m]) > 0]

        # Collect cases across available metrics
        case_keys = set()
        for m in metric_names:
            case_keys.update(data[arm][seeds[0]][m].keys())
        case_keys = sorted(list(case_keys))
        case_rows = []

        for key in case_keys:
            row = {"Case": key}
            for m in metric_names:
                vals = []
                for s in seeds:
                    v = data[arm][s].get(m, {}).get(key, np.nan)
                    if np.isfinite(v):
                        vals.append(v)
                row[m] = float(np.mean(vals)) if vals else np.nan
            case_rows.append(row)

        df_arm = pd.DataFrame(case_rows)
        per_arm_dfs[arm] = df_arm

        # Aggregate FID across seeds
        fid_vals = [data[arm][s].get("FID", [np.nan])[0] for s in seeds if "FID" in data[arm][s]]
        per_arm_fids[arm] = float(np.mean(fid_vals)) if fid_vals else np.nan

    return per_arm_dfs, per_arm_fids


def generate_summary_table(
    per_arm_dfs: Dict[str, pd.DataFrame],
    per_arm_fids: Dict[str, float],
    metrics_to_show: List[str]
) -> pd.DataFrame:
    """Generates formatted Mean +/- STD summary table for all arms."""
    rows = []
    for arm, df in per_arm_dfs.items():
        row = {"Arm": ARM_PRETTY_NAMES.get(arm, arm)}
        for m in metrics_to_show:
            if m == "FID":
                row[METRIC_PRETTY_NAMES.get(m, m)] = f"{per_arm_fids.get(arm, np.nan):.2f}"
            elif m in df.columns:
                vals = df[m].dropna().values
                mean_val = np.mean(vals)
                std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row[METRIC_PRETTY_NAMES.get(m, m)] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                row[METRIC_PRETTY_NAMES.get(m, m)] = "N/A"
        row["N"] = len(df)
        rows.append(row)

    return pd.DataFrame(rows)


def run_paired_hypothesis_tests(
    per_arm_dfs: Dict[str, pd.DataFrame],
    test_metrics: List[str]
) -> pd.DataFrame:
    """
    Runs paired Wilcoxon signed-rank tests for all pairs of arms on selected metrics.
    Applies Holm-Bonferroni correction.
    """
    arms = list(per_arm_dfs.keys())
    test_rows = []

    for metric in test_metrics:
        # Check metric availability in all arms
        if not all(metric in df.columns for df in per_arm_dfs.values()):
            continue

        raw_p_values = []
        pair_data = []

        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                arm1, arm2 = arms[i], arms[j]
                df1 = per_arm_dfs[arm1].set_index("Case")
                df2 = per_arm_dfs[arm2].set_index("Case")
                common_cases = df1.index.intersection(df2.index)

                v1 = df1.loc[common_cases, metric].values
                v2 = df2.loc[common_cases, metric].values
                valid_mask = np.isfinite(v1) & np.isfinite(v2)
                v1, v2 = v1[valid_mask], v2[valid_mask]

                diff = v1 - v2
                med_diff = float(np.median(diff)) if len(diff) > 0 else np.nan
                ci_low, ci_high = bootstrap_ci(diff)

                if len(diff) >= 5 and np.any(diff != 0):
                    try:
                        w_stat, p_val = wilcoxon(v1, v2, alternative='two-sided')
                    except Exception:
                        p_val = 1.0
                else:
                    p_val = 1.0

                raw_p_values.append(p_val)
                pair_data.append((arm1, arm2, med_diff, ci_low, ci_high, len(diff)))

        # Adjust p-values for this metric
        adj_p_values = holm_bonferroni(raw_p_values)

        for (arm1, arm2, med_diff, ci_low, ci_high, n), p_raw, p_adj in zip(pair_data, raw_p_values, adj_p_values):
            sig_star = ""
            if p_adj < 0.001:
                sig_star = "***"
            elif p_adj < 0.01:
                sig_star = "**"
            elif p_adj < 0.05:
                sig_star = "*"

            test_rows.append({
                "Metric": METRIC_PRETTY_NAMES.get(metric, metric),
                "Arm 1": ARM_PRETTY_NAMES.get(arm1, arm1),
                "Arm 2": ARM_PRETTY_NAMES.get(arm2, arm2),
                "Median Diff (1 - 2)": f"{med_diff:+.4f}",
                "95% CI": f"[{ci_low:+.4f}, {ci_high:+.4f}]",
                "Raw p": f"{p_raw:.4e}" if p_raw < 0.001 else f"{p_raw:.4f}",
                "Holm p": f"{p_adj:.4e}{sig_star}" if p_adj < 0.001 else f"{p_adj:.4f}{sig_star}",
                "N": n
            })

    return pd.DataFrame(test_rows)


def df_to_markdown_str(df: pd.DataFrame) -> str:
    """Formats DataFrame as a GitHub-flavored Markdown table without requiring tabulate."""
    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def analyze(raw_pkl_path: str, out_dir: Optional[str] = None, modality: str = "auto"):
    raw_path = Path(raw_pkl_path).expanduser()
    if not raw_path.exists():
        raise FileNotFoundError(f"Ablation file not found: {raw_path}")

    save_dir = Path(out_dir).expanduser() if out_dir else raw_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ablation results from: {raw_path}")
    per_arm_dfs, per_arm_fids = load_and_aggregate_results(raw_path)

    # Detect modality if auto
    has_vessels = any(
        "Vessel_Dice" in df.columns and df["Vessel_Dice"].dropna().count() > 0
        for df in per_arm_dfs.values()
    )
    if modality == "auto":
        is_ct = ("ct" in str(raw_path).lower()) or (not has_vessels)
        modality_str = "CT" if is_ct else "OCT"
    else:
        modality_str = modality.upper()
        is_ct = (modality_str == "CT")

    summary_metrics = [
        "Adjacent_SSIM_Gap",
        "Log_Spectral_Distance",
        "Orthogonal_SSIM_mean",
        "Orthogonal_LPIPS_mean",
        "SDIF",
        "PSNR",
        "SSIM",
        "LPIPS",
        "FID",
        "Flow_Displacement_L1"
    ]
    if not is_ct and has_vessels:
        summary_metrics.insert(8, "Vessel_Dice")

    available_cols = set()
    for df in per_arm_dfs.values():
        for col in df.columns:
            if col != "Case" and df[col].dropna().count() > 0:
                available_cols.add(col)

    summary_metrics = [m for m in summary_metrics if (m in available_cols or m == "FID")]
    hypothesis_metrics = [m for m in hypothesis_metrics if m in available_cols]

    summary_df = generate_summary_table(per_arm_dfs, per_arm_fids, summary_metrics)
    summary_csv = save_dir / "ablation_summary_table.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\n" + "=" * 90)
    print(f"NOISE COUPLING ABLATION SUMMARY ({modality_str}: 2D Flow + MMSE)")
    print("=" * 90)
    print(df_to_markdown_str(summary_df))

    tests_df = run_paired_hypothesis_tests(per_arm_dfs, hypothesis_metrics)
    tests_csv = save_dir / "paired_statistical_tests.csv"
    tests_df.to_csv(tests_csv, index=False)
    print("\n" + "=" * 90)
    print("PAIRED STATISTICAL TESTS (Wilcoxon Signed-Rank with Holm-Bonferroni Correction)")
    print("=" * 90)
    print(df_to_markdown_str(tests_df))

    # Export clean markdown report
    report_md = save_dir / "noise_coupling_report.md"
    with open(report_md, "w") as f:
        f.write(f"# Noise Coupling Ablation: Through-Plane Structural Comparison ({modality_str})\n\n")
        f.write("### Model Configuration\n")
        f.write(f"- **Modality**: {modality_str}\n")
        if is_ct:
            f.write("- **Initial MMSE Estimator**: `2026-06-26_12-03-29_ms_xray_to_ct`\n")
            f.write("- **2D Rectified Flow**: `2026-06-30_11-24-06_ms_rflow_ct_float32` (SongUNet, $\\sigma_{start} = 0.01$, $T = 20$)\n")
        else:
            f.write("- **Initial MMSE Estimator**: `2026-01-23_11-33-50_ms_mmse_added_augs_transupnet`\n")
            f.write("- **2D Rectified Flow**: `2026-02-20_09-09-18_ms__rflow_cleaned_song_sigma_01` (SongUNet, $\\sigma_{start} = 0.1$, $T = 20$)\n")
        f.write("- **Randomness**: Paired Randomness / Common Random Numbers (matched endpoints $z_0, z_1$ across arms)\n\n")
        f.write("### Summary Table\n\n")
        f.write(df_to_markdown_str(summary_df))
        f.write("\n\n### Paired Statistical Tests\n\n")
        f.write(df_to_markdown_str(tests_df))
        f.write("\n\n*Significance: \\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001 (Holm-corrected)*\n")

    print(f"\nSaved analysis summary table to: {summary_csv}")
    print(f"Saved paired statistical tests to: {tests_csv}")
    print(f"Saved markdown report to: {report_md}")


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of noise coupling ablation.")
    parser.add_argument(
        "--raw_pkl",
        type=str,
        default="/media/thomas-pinetz/data/data/OCT/results/noise_coupling_ablation_2026-02-20_09-09-18/ablation_results_raw.pkl",
        help="Path to ablation_results_raw.pkl."
    )
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save output tables and reports.")
    parser.add_argument("--modality", type=str, choices=["oct", "ct", "auto"], default="auto", help="Modality (oct, ct, or auto).")
    args = parser.parse_args()
    analyze(args.raw_pkl, args.out_dir, args.modality)


if __name__ == "__main__":
    main()
