import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None
import pandas as pd
import imageio.v3 as iio

from pmrf3dsynthesis.data import constants as C
from pmrf3dsynthesis.eval_model import load_dataset, load_sample, slice_volume_ssim, save_summary_table


class TestEvalModelCT(unittest.TestCase):

    def setUp(self):
        if not HAS_H5PY:
            self.skipTest("h5py is not installed")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_dataset_csv_and_txt(self):
        # Test .csv with 'Keys'
        csv_path = self.root / "split.csv"
        df = pd.DataFrame({"Keys": ["case_001", "case_002", "case_003"]})
        df.to_csv(csv_path, index=False)
        keys_csv = load_dataset(csv_path)
        self.assertEqual(keys_csv, ["case_001", "case_002", "case_003"])

        # Test .csv with 'patient_id'
        csv_path2 = self.root / "split2.csv"
        df2 = pd.DataFrame({"patient_id": ["pat_A", "pat_B"]})
        df2.to_csv(csv_path2, index=False)
        keys_csv2 = load_dataset(csv_path2)
        self.assertEqual(keys_csv2, ["pat_A", "pat_B"])

        # Test .txt
        txt_path = self.root / "split.txt"
        with open(txt_path, "w") as f:
            f.write("case_101\ncase_102\n\ncase_103\n")
        keys_txt = load_dataset(txt_path)
        self.assertEqual(keys_txt, ["case_101", "case_102", "case_103"])

    def test_load_sample_ct_h5(self):
        case_id = "LIDC_0001"
        case_dir = self.root / case_id
        case_dir.mkdir()
        h5_path = case_dir / "ct_xray_data.h5"

        # Create dummy H5 with CT volume (32, 64, 64) and X-ray (64, 64)
        dummy_ct = np.random.uniform(0, 5800, size=(32, 64, 64)).astype(np.float32)
        dummy_xray = np.random.uniform(0, 1200, size=(64, 64)).astype(np.float32)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("ct", data=dummy_ct)
            f.create_dataset("xray1", data=dummy_xray)

        input_img, volume = load_sample(
            key=case_id,
            modality=C.MODALITY_CT,
            data_dir=self.root,
            data_format=C.FORMAT_H5,
            ct_scale=5800.0,
            xray_scale=1200.0
        )

        self.assertEqual(input_img.shape, (64, 64))
        self.assertEqual(volume.shape, (32, 64, 64))
        self.assertTrue(0.0 <= input_img.min() and input_img.max() <= 1.0)
        self.assertTrue(0.0 <= volume.min() and volume.max() <= 1.0)

    def test_load_sample_ct_npy(self):
        case_id = "case_npy_01"
        vol_dir = self.root / "ct_vols"
        img_dir = self.root / "xray_imgs"
        vol_dir.mkdir()
        img_dir.mkdir()

        dummy_vol = np.random.uniform(0.0, 1.0, size=(32, 64, 64)).astype(np.float32)
        dummy_xray = (np.random.uniform(0, 255, size=(64, 64))).astype(np.uint8)

        np.save(vol_dir / f"{case_id}.npy", dummy_vol)
        iio.imwrite(img_dir / f"{case_id}.png", dummy_xray)

        input_img, volume = load_sample(
            key=case_id,
            modality=C.MODALITY_CT,
            vol_path=vol_dir,
            img_path=img_dir,
            data_format=C.FORMAT_NPY
        )

        self.assertEqual(input_img.shape, (64, 64))
        self.assertEqual(volume.shape, (32, 64, 64))
        self.assertTrue(0.0 <= input_img.min() and input_img.max() <= 1.0)
        self.assertTrue(0.0 <= volume.min() and volume.max() <= 1.0)

    def test_slice_volume_ssim(self):
        y_hat = torch.ones((1, 1, 16, 32, 32))
        gt = torch.ones((1, 1, 16, 32, 32))
        score = slice_volume_ssim(y_hat[0], gt[0])
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_save_summary_table(self):
        eval_results = {
            "PSNR": {"case_1": 25.5, "case_2": 26.5},
            "SSIM": {"case_1": 0.85, "case_2": 0.87},
            "FID": [15.2]
        }
        save_summary_table(eval_results, self.root)
        csv_file = self.root / "eval_results.csv"
        self.assertTrue(csv_file.exists())
        df = pd.read_csv(csv_file)
        self.assertIn("PSNR", df.columns)
        self.assertIn("SSIM", df.columns)
        self.assertEqual(len(df), 5)  # 2 cases + MEAN + STD + FID


if __name__ == '__main__':
    unittest.main()
