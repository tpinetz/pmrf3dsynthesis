import unittest
import numpy as np
import torch

from pmrf3dsynthesis.evaluation.interslice import (
    compute_lag_ssim_profile,
    compute_lag_ssim_comparison,
    compute_through_plane_psd,
    compute_spectral_metrics,
    compute_orthogonal_reslice_metrics,
    compute_flow_displacement,
    evaluate_interslice_structure
)


class TestIntersliceMetrics(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Smooth volume
        self.D, self.H, self.W = 32, 64, 64
        z = np.linspace(0, 1, self.D)[:, None, None]
        base_slice = np.random.rand(1, self.H, self.W).astype(np.float32)
        self.smooth_vol = np.clip(base_slice + 0.1 * z + 0.01 * np.random.randn(self.D, self.H, self.W).astype(np.float32), 0.0, 1.0)
        # Flickering / noisy volume
        self.noisy_vol = np.clip(self.smooth_vol + 0.2 * np.random.randn(self.D, self.H, self.W).astype(np.float32), 0.0, 1.0)

    def test_lag_ssim_comparison(self):
        res = compute_lag_ssim_comparison(self.noisy_vol, self.smooth_vol, lags=(1, 2, 4))
        self.assertIn("LagSSIM_pred_k1", res)
        self.assertIn("LagSSIM_gt_k1", res)
        self.assertIn("Adjacent_SSIM_Gap", res)
        # Smooth volume must have higher adjacent SSIM than noisy volume
        self.assertGreater(res["LagSSIM_gt_k1"], res["LagSSIM_pred_k1"])
        self.assertGreater(res["Adjacent_SSIM_Gap"], 0.0)

    def test_spectral_metrics(self):
        res = compute_spectral_metrics(self.noisy_vol, self.smooth_vol)
        self.assertIn("Log_Spectral_Distance", res)
        self.assertIn("HFR_pred", res)
        self.assertIn("HFR_gt", res)
        # Noisy volume should have higher high-frequency ratio (HFR)
        self.assertGreater(res["HFR_pred"], res["HFR_gt"])
        self.assertGreater(res["Log_Spectral_Distance"], 0.0)

    def test_orthogonal_reslices(self):
        res = compute_orthogonal_reslice_metrics(self.noisy_vol, self.smooth_vol, num_samples=4)
        self.assertIn("Orthogonal_SSIM_coronal", res)
        self.assertIn("Orthogonal_SSIM_sagittal", res)
        self.assertIn("Orthogonal_SSIM_mean", res)
        self.assertTrue(0.0 <= res["Orthogonal_SSIM_mean"] <= 1.0)

    def test_flow_displacement(self):
        res = compute_flow_displacement(self.noisy_vol, self.smooth_vol)
        self.assertIn("Flow_Displacement_L1", res)
        self.assertIn("Flow_Displacement_L2", res)
        self.assertGreater(res["Flow_Displacement_L1"], 0.0)

    def test_evaluate_interslice_structure_wrapper(self):
        res = evaluate_interslice_structure(
            pred=self.noisy_vol,
            gt=self.smooth_vol,
            mmse_pred=self.smooth_vol,
            lags=(1, 2)
        )
        self.assertIn("Adjacent_SSIM_Gap", res)
        self.assertIn("Log_Spectral_Distance", res)
        self.assertIn("Orthogonal_SSIM_mean", res)
        self.assertIn("Flow_Displacement_L1", res)


if __name__ == "__main__":
    unittest.main()
