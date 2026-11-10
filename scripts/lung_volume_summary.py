"""Summarise the CSVs written by lung_volume_eval.py.

For every (segmenter, structure, method): volume MAE, mean signed error,
Pearson and Spearman correlation with the ground-truth volume, and mean Dice.
Adds a trivial `mean_volume_baseline` that predicts the leave-one-out mean
ground-truth volume for every case.

  python scripts/lung_volume_summary.py out_dir/*.csv --out summary.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def stats(vol, gt, dice=None):
    err = vol - gt
    ok = np.std(vol) > 0 and np.std(gt) > 0 and len(vol) > 2
    return {
        "n": len(vol),
        "gt_mean_ml": gt.mean(),
        "pred_mean_ml": vol.mean(),
        "mae_ml": np.abs(err).mean(),
        "bias_ml": err.mean(),
        "mape_pct": 100 * np.mean(np.abs(err) / np.maximum(gt, 1e-6)),
        "pearson_r": pearsonr(vol, gt)[0] if ok else np.nan,
        "spearman_r": spearmanr(vol, gt)[0] if ok else np.nan,
        "dice_mean": np.nanmean(dice) if dice is not None else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--complete_only", action="store_true",
                    help="Only keep cases present for every method of a segmenter.")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True)
    df = df[df.structure != "_done"].drop_duplicates(["key", "method", "segmenter", "structure"])
    rows = []
    for (seg, struct), g in df.groupby(["segmenter", "structure"]):
        if args.complete_only:
            n_methods = g.method.nunique()
            counts = g.groupby("key").method.nunique()
            g = g[g.key.isin(counts[counts == n_methods].index)]
        gt = g[g.method == "gt"].set_index("key").volume_ml
        if len(gt) > 2:
            loo = (gt.sum() - gt) / (len(gt) - 1)
            base = stats(loo.values, gt.values)
            base["pearson_r"] = base["spearman_r"] = np.nan  # constant predictor, undefined
            rows.append({"segmenter": seg, "structure": struct, "method": "mean_volume_baseline", **base})
        for method, m in g[g.method != "gt"].groupby("method"):
            rows.append({"segmenter": seg, "structure": struct, "method": method,
                         **stats(m.volume_ml.values, m.gt_volume_ml.values, m.dice_vs_gt.values)})
    out = pd.DataFrame(rows).sort_values(["segmenter", "structure", "mae_ml"])
    out.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
