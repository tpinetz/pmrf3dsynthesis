import tempfile
import pickle
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from pmrf3dsynthesis.eval_model import compute_sdif, compute_volume_lpips, save_summary_table
from scripts.extract_metrics import extract_metrics_from_directory


class TestEvalMetrics(unittest.TestCase):

    def test_compute_sdif_identical_volumes(self):
        vol = np.random.rand(10, 32, 32).astype(np.float32)
        res = compute_sdif(vol, vol)
        self.assertIn("SDIF", res)
        self.assertIn("SDIF_pred", res)
        self.assertIn("SDIF_gt", res)
        self.assertAlmostEqual(res["SDIF"], 0.0, places=6)
        self.assertAlmostEqual(res["SDIF_pred"], res["SDIF_gt"], places=6)

    def test_compute_sdif_smooth_vs_oscillating(self):
        # Smooth volume along slice axis
        smooth_vol = np.zeros((10, 16, 16), dtype=np.float32)
        for i in range(10):
            smooth_vol[i] = i * 0.1

        # Oscillating volume
        noisy_vol = np.zeros((10, 16, 16), dtype=np.float32)
        for i in range(10):
            noisy_vol[i] = 1.0 if (i % 2 == 0) else 0.0

        res = compute_sdif(noisy_vol, smooth_vol)
        self.assertGreater(res["SDIF_pred"], res["SDIF_gt"])
        self.assertGreater(res["SDIF"], 0.0)

    def test_compute_volume_lpips(self):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
        lpips_fn.eval()

        vol1 = np.ones((8, 64, 64), dtype=np.float32) * 0.5
        vol2 = np.ones((8, 64, 64), dtype=np.float32) * 0.5

        # Identical volumes should give LPIPS near 0
        score_identical = compute_volume_lpips(vol1, vol2, lpips_fn, device=device)
        self.assertAlmostEqual(score_identical, 0.0, places=3)

        # Perturbed volume should give higher LPIPS
        vol_noise = np.random.rand(8, 64, 64).astype(np.float32)
        score_noisy = compute_volume_lpips(vol_noise, vol1, lpips_fn, device=device)
        self.assertGreater(score_noisy, score_identical)

    def test_save_summary_table_with_lpips_and_sdif(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            eval_results = {
                "PSNR": {"case_1": 20.0, "case_2": 25.0},
                "SSIM": {"case_1": 0.8, "case_2": 0.9},
                "LPIPS": {"case_1": 0.15, "case_2": 0.10},
                "SDIF": {"case_1": 0.005, "case_2": 0.003},
                "Vessel_Dice": {"case_1": 0.7, "case_2": 0.8},
                "FID": [15.2]
            }

            save_summary_table(eval_results, tmp_path)
            csv_file = tmp_path / "eval_results.csv"
            self.assertTrue(csv_file.exists())

            df = pd.read_csv(csv_file)
            self.assertIn("LPIPS", df.columns)
            self.assertIn("SDIF", df.columns)
            self.assertIn("Vessel_Dice", df.columns)

            mean_row = df[df["Key"] == "MEAN"].iloc[0]
            self.assertAlmostEqual(mean_row["PSNR"], 22.5)
            self.assertAlmostEqual(mean_row["LPIPS"], 0.125)
            self.assertAlmostEqual(mean_row["SDIF"], 0.004)

    def test_extract_metrics_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_dir = tmp_path / "test_run"
            run_dir.mkdir(parents=True)

            eval_results = {
                "PSNR": {"c1": 20.0, "c2": 22.0},
                "SSIM": {"c1": 0.7, "c2": 0.8},
                "LPIPS": {"c1": 0.12, "c2": 0.14},
                "SDIF": {"c1": 0.002, "c2": 0.004},
                "FID": [12.5]
            }
            with open(run_dir / "eval_results.pkl", "wb") as f:
                pickle.dump(eval_results, f)

            df = extract_metrics_from_directory(str(tmp_path), sort_by="lpips")
            self.assertEqual(len(df), 1)
            self.assertIn("LPIPS ↓", df.columns)
            self.assertIn("SDIF ↓", df.columns)


if __name__ == '__main__':
    unittest.main()
