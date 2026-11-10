import unittest
import tempfile
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from pmrf3dsynthesis.eval_model import compute_vessel_metrics, save_summary_table, evaluate_model
from pmrf3dsynthesis.data import constants as C
from scripts.extract_metrics import extract_metrics_from_directory


class TestEvalVessels(unittest.TestCase):

    def test_compute_vessel_metrics_identical(self):
        pred = np.ones((64, 64), dtype=np.float32)
        gt = np.ones((64, 64), dtype=np.float32)
        metrics = compute_vessel_metrics(pred, gt, threshold=0.5)

        self.assertAlmostEqual(metrics["Vessel_Dice"], 1.0, places=4)
        self.assertAlmostEqual(metrics["Vessel_IoU"], 1.0, places=4)
        self.assertAlmostEqual(metrics["Vessel_Soft_Dice"], 1.0, places=4)

    def test_compute_vessel_metrics_disjoint(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        pred[:32, :] = 0.9
        gt = np.zeros((64, 64), dtype=np.float32)
        gt[32:, :] = 0.9
        metrics = compute_vessel_metrics(pred, gt, threshold=0.5)

        self.assertAlmostEqual(metrics["Vessel_Dice"], 0.0, places=4)
        self.assertAlmostEqual(metrics["Vessel_IoU"], 0.0, places=4)
        self.assertAlmostEqual(metrics["Vessel_Soft_Dice"], 0.0, places=4)

    def test_compute_vessel_metrics_partial_overlap(self):
        pred = np.zeros((10, 10), dtype=np.float32)
        gt = np.zeros((10, 10), dtype=np.float32)
        # Overlap of 2 elements, pred has 4 elements, gt has 4 elements
        pred[:2, :2] = 0.8  # 4 elements
        gt[1:3, :2] = 0.8   # 4 elements (overlaps at row 1, cols 0 and 1 -> 2 elements)

        metrics = compute_vessel_metrics(pred, gt, threshold=0.5)
        # Dice = 2 * 2 / (4 + 4) = 4 / 8 = 0.5
        self.assertAlmostEqual(metrics["Vessel_Dice"], 0.5, places=4)
        # IoU = 2 / (4 + 4 - 2) = 2 / 6 = 1/3
        self.assertAlmostEqual(metrics["Vessel_IoU"], 1.0 / 3.0, places=4)

    def test_save_summary_table_with_vessels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res_path = Path(tmpdir)
            eval_results = {
                "PSNR": {"case1": 25.0, "case2": 27.0},
                "SSIM": {"case1": 0.85, "case2": 0.90},
                "FID": [12.5],
                "Vessel_Dice": {"case1": 0.65, "case2": 0.75},
                "Vessel_IoU": {"case1": 0.50, "case2": 0.60},
                "Vessel_Soft_Dice": {"case1": 0.70, "case2": 0.80}
            }

            save_summary_table(eval_results, res_path)
            csv_path = res_path / "eval_results.csv"
            self.assertTrue(csv_path.exists())

            df = pd.read_csv(csv_path)
            self.assertIn("Vessel_Dice", df.columns)
            self.assertIn("Vessel_IoU", df.columns)
            self.assertIn("Vessel_Soft_Dice", df.columns)

            mean_row = df[df["Key"] == "MEAN"].iloc[0]
            self.assertAlmostEqual(mean_row["Vessel_Dice"], 0.70, places=4)
            self.assertAlmostEqual(mean_row["Vessel_IoU"], 0.55, places=4)
            self.assertAlmostEqual(mean_row["Vessel_Soft_Dice"], 0.75, places=4)

    def test_extract_metrics_script_with_vessels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_oct_test"
            run_dir.mkdir(parents=True)

            eval_results = {
                "PSNR": {"case1": 22.0, "case2": 24.0},
                "SSIM": {"case1": 0.75, "case2": 0.80},
                "FID": [15.0],
                "Vessel_Dice": {"case1": 0.55, "case2": 0.65}
            }
            with open(run_dir / "eval_results.pkl", "wb") as f:
                pickle.dump(eval_results, f)

            df = extract_metrics_from_directory(str(tmpdir))
            self.assertIn("Vessel Dice", df.columns)
            self.assertEqual(len(df), 1)
            self.assertIn("0.6000", df["Vessel Dice"].iloc[0])


if __name__ == "__main__":
    unittest.main()
