import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from pmrf3dsynthesis.data import constants as C
from pmrf3dsynthesis.visualize_eval import load_evaluation_cases, EvaluationViewer


class TestVisualizeEval(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_evaluation_cases_from_csv(self):
        csv_file = self.root / "eval_results.csv"
        df = pd.DataFrame([
            {"Key": "case_1", "PSNR": 28.5, "SSIM": 0.89},
            {"Key": "case_2", "PSNR": 24.1, "SSIM": 0.82},
            {"Key": "MEAN", "PSNR": 26.3, "SSIM": 0.855},
            {"Key": "FID", "PSNR": 14.5, "SSIM": 14.5}
        ])
        df.to_csv(csv_file, index=False)

        keys, psnr_dict, ssim_dict, fid = load_evaluation_cases(self.root)
        self.assertEqual(keys, ["case_1", "case_2"])
        self.assertEqual(psnr_dict["case_1"], 28.5)
        self.assertEqual(ssim_dict["case_2"], 0.82)

    def test_load_evaluation_cases_from_pkl(self):
        pkl_file = self.root / "eval_results.pkl"
        data = {
            "PSNR": {"case_A": 30.1, "case_B": 22.0},
            "SSIM": {"case_A": 0.92, "case_B": 0.78},
            "FID": [12.8]
        }
        with open(pkl_file, "wb") as f:
            pickle.dump(data, f)

        keys, psnr_dict, ssim_dict, fid = load_evaluation_cases(self.root)
        self.assertEqual(keys, ["case_A", "case_B"])
        self.assertEqual(psnr_dict["case_A"], 30.1)
        self.assertEqual(fid, 12.8)

    def test_evaluation_viewer_sorting(self):
        res_dir = self.root / "run_1"
        res_dir.mkdir()
        data = {
            "PSNR": {"case_low": 20.0, "case_high": 35.0, "case_mid": 27.0},
            "SSIM": {"case_low": 0.70, "case_high": 0.95, "case_mid": 0.85},
            "FID": [10.0]
        }
        with open(res_dir / "eval_results.pkl", "wb") as f:
            pickle.dump(data, f)

        # Create dummy prediction .npy files
        for k in data["PSNR"].keys():
            np.save(res_dir / f"{k}.npy", np.zeros((16, 32, 32), dtype=np.float32))

        # Test PSNR ascending
        viewer_asc = EvaluationViewer(res_path=res_dir, sort_by="psnr_asc")
        self.assertEqual(viewer_asc.keys, ["case_low", "case_mid", "case_high"])

        # Test PSNR descending
        viewer_desc = EvaluationViewer(res_path=res_dir, sort_by="psnr_desc")
        self.assertEqual(viewer_desc.keys, ["case_high", "case_mid", "case_low"])

        # Test slice extraction across planes
        test_vol = np.arange(24).reshape(2, 3, 4)
        viewer_asc.slice_axis = 0
        self.assertEqual(viewer_asc._extract_slice(test_vol, 0).shape, (3, 4))
        viewer_asc.slice_axis = 1
        self.assertEqual(viewer_asc._extract_slice(test_vol, 0).shape, (2, 4))
        viewer_asc.slice_axis = 2
        self.assertEqual(viewer_asc._extract_slice(test_vol, 0).shape, (2, 3))

    def test_case_jumping(self):
        res_dir = self.root / "run_jump"
        res_dir.mkdir()
        keys = [f"case_{i:03d}" for i in range(50)]
        for k in keys:
            np.save(res_dir / f"{k}.npy", np.zeros((8, 16, 16), dtype=np.float32))

        viewer = EvaluationViewer(res_path=res_dir, keys=keys)
        self.assertEqual(viewer.current_case_idx, 0)
        self.assertEqual(viewer.current_key, "case_000")

        # Jump to index 25
        viewer.current_case_idx = 25
        viewer._load_current_case_data()
        self.assertEqual(viewer.current_key, "case_025")

        # Jump past end
        viewer.current_case_idx = min(len(keys) - 1, viewer.current_case_idx + 100)
        viewer._load_current_case_data()
        self.assertEqual(viewer.current_key, "case_049")


if __name__ == '__main__':
    unittest.main()
