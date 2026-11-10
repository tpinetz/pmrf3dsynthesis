import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import imageio.v3 as iio
import torch

from pmrf3dsynthesis.eval_model_only_fundus import (
    load_fundus_image,
    crop_fundus_fovea,
    prepare_model_input,
    collect_fundus_inputs,
    resolve_checkpoint_path,
    run_fundus_synthesis,
)


class MockFovModel:
    def __init__(self, fovea_x=500.0, fovea_y=400.0):
        self.fovea_x = fovea_x
        self.fovea_y = fovea_y

    def predict(self, img):
        return [self.fovea_x, self.fovea_y, 0.0, 0.0]


class MockErrorFovModel:
    def predict(self, img):
        raise RuntimeError("Detection failed")


class MockSynthesisModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        # x is (1, 3, 128, 256)
        # Returns (1, 1, 128, 256, 256) in [-1, 1] range
        batch_size = x.shape[0]
        return torch.zeros((batch_size, 1, 128, 256, 256), dtype=torch.float32)


class TestEvalModelOnlyFundus(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_fundus_image_rgb(self):
        # Test RGB image loading
        img_arr = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        img_file = self.tmp_path / "test_rgb.png"
        iio.imwrite(str(img_file), img_arr)

        loaded = load_fundus_image(img_file)
        self.assertEqual(loaded.shape, (200, 300, 3))
        self.assertEqual(loaded.dtype, np.uint8)
        np.testing.assert_array_equal(loaded, img_arr)

    def test_load_fundus_image_grayscale(self):
        # Test 2D grayscale expansion to 3-channel
        gray_arr = np.random.randint(0, 255, (150, 250), dtype=np.uint8)
        img_file = self.tmp_path / "test_gray.png"
        iio.imwrite(str(img_file), gray_arr)

        loaded = load_fundus_image(img_file)
        self.assertEqual(loaded.shape, (150, 250, 3))
        self.assertEqual(loaded.dtype, np.uint8)
        np.testing.assert_array_equal(loaded[:, :, 0], gray_arr)
        np.testing.assert_array_equal(loaded[:, :, 1], gray_arr)

    def test_load_fundus_image_rgba(self):
        # Test RGBA channel stripping
        rgba_arr = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        img_file = self.tmp_path / "test_rgba.png"
        iio.imwrite(str(img_file), rgba_arr)

        loaded = load_fundus_image(img_file)
        self.assertEqual(loaded.shape, (100, 100, 3))
        np.testing.assert_array_equal(loaded, rgba_arr[:, :, :3])

    def test_crop_fundus_fovea_normal(self):
        # 1000x1000 fundus image with known fovea at (500, 500)
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        img[490:510, 490:510, :] = 255  # Distinct center marker

        crop, (fx, fy) = crop_fundus_fovea(
            img,
            fovea_coords=(500, 500),
            crop_radius=100,
            crop_size=200
        )
        self.assertEqual(crop.shape, (200, 200, 3))
        self.assertEqual(fx, 500.0)
        self.assertEqual(fy, 500.0)

    def test_crop_fundus_fovea_boundary_padding(self):
        # Fovea near upper-left border: x=30, y=20 with radius 100
        img = np.random.randint(50, 200, (500, 500, 3), dtype=np.uint8)
        crop, (fx, fy) = crop_fundus_fovea(
            img,
            fovea_coords=(30, 20),
            crop_radius=100,
            crop_size=200
        )
        # Should be padded seamlessly to 200x200
        self.assertEqual(crop.shape, (200, 200, 3))
        self.assertEqual(fx, 30.0)
        self.assertEqual(fy, 20.0)

    def test_crop_fundus_fovea_with_fov_model(self):
        img = np.zeros((800, 800, 3), dtype=np.uint8)
        mock_model = MockFovModel(fovea_x=450.0, fovea_y=350.0)

        crop, (fx, fy) = crop_fundus_fovea(
            img,
            fov_model=mock_model,
            crop_radius=150,
            crop_size=300
        )
        self.assertEqual(crop.shape, (300, 300, 3))
        self.assertEqual(fx, 450.0)
        self.assertEqual(fy, 350.0)

    def test_crop_fundus_fovea_fallback_on_error(self):
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        mock_error = MockErrorFovModel()

        crop, (fx, fy) = crop_fundus_fovea(
            img,
            fov_model=mock_error,
            crop_radius=100,
            crop_size=200
        )
        self.assertEqual(crop.shape, (200, 200, 3))
        # Image center fallback: W/2 = 400, H/2 = 300
        self.assertEqual(fx, 400.0)
        self.assertEqual(fy, 300.0)

    def test_prepare_model_input(self):
        crop = np.random.randint(0, 255, (850, 850, 3), dtype=np.uint8)
        tensor = prepare_model_input(crop, target_shape=(128, 256))
        self.assertEqual(tensor.shape, (1, 3, 128, 256))
        self.assertGreaterEqual(tensor.min().item(), -1.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_collect_fundus_inputs(self):
        # Single image
        img1 = self.tmp_path / "case1.png"
        iio.imwrite(str(img1), np.zeros((100, 100, 3), dtype=np.uint8))

        res1 = collect_fundus_inputs(img1)
        self.assertEqual(len(res1), 1)
        self.assertEqual(res1[0][0], "case1")

        # Directory with multiple images
        sub_dir = self.tmp_path / "images"
        sub_dir.mkdir()
        img2 = sub_dir / "case2.jpg"
        iio.imwrite(str(img2), np.zeros((100, 100, 3), dtype=np.uint8))

        res2 = collect_fundus_inputs(sub_dir)
        self.assertEqual(len(res2), 1)
        self.assertEqual(res2[0][0], "case2")

        # CSV list
        csv_file = self.tmp_path / "cases.csv"
        df = pd.DataFrame({"key": ["case2"]})
        df.to_csv(csv_file, index=False)

        res3 = collect_fundus_inputs(csv_file, fundus_dir=sub_dir)
        self.assertEqual(len(res3), 1)
        self.assertEqual(res3[0][0], "case2")

    def test_run_fundus_synthesis_crop_only(self):
        # Create test fundus image
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        img_path = self.tmp_path / "test_patient.png"
        iio.imwrite(str(img_path), img)

        res_path = self.tmp_path / "out_crops"
        fundus_inputs = [("test_patient", img_path)]

        summary = run_fundus_synthesis(
            fundus_inputs=fundus_inputs,
            res_path=res_path,
            crop_only=True,
            crop_radius=100,
            crop_size=200,
            detect_fovea=False
        )

        self.assertEqual(summary["num_processed"], 1)
        expected_crop = res_path / "test_patient.png"
        self.assertTrue(expected_crop.exists())
        saved_img = iio.imread(str(expected_crop))
        self.assertEqual(saved_img.shape, (200, 200, 3))

        # Check metadata CSV
        meta_csv = res_path / "fundus_crops_metadata.csv"
        self.assertTrue(meta_csv.exists())
        meta_df = pd.read_csv(meta_csv)
        self.assertEqual(len(meta_df), 1)
        self.assertEqual(meta_df.iloc[0]["key"], "test_patient")

    def test_run_fundus_synthesis_with_mock_model(self):
        # Create test fundus image
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        img_path = self.tmp_path / "synth_patient.png"
        iio.imwrite(str(img_path), img)

        res_path = self.tmp_path / "out_synth"
        fundus_inputs = [("synth_patient", img_path)]

        # Patch load_synthesis_model to return mock
        mock_model = MockSynthesisModel()
        from unittest.mock import patch
        with patch("pmrf3dsynthesis.eval_model_only_fundus.load_synthesis_model", return_value=mock_model):
            summary = run_fundus_synthesis(
                fundus_inputs=fundus_inputs,
                res_path=res_path,
                ckpt_path="dummy.ckpt",
                crop_only=False,
                crop_radius=100,
                crop_size=200,
                save_crops=True,
                save_mip=True,
                detect_fovea=False
            )

        self.assertEqual(summary["num_processed"], 1)
        vol_file = res_path / "synth_patient.npy"
        self.assertTrue(vol_file.exists())
        vol = np.load(vol_file)
        self.assertEqual(vol.shape, (128, 256, 256))

        # Check MIP preview
        mip_file = res_path / "synth_patient_mip.png"
        self.assertTrue(mip_file.exists())

        # Check saved crop
        crop_file = res_path / "crops" / "synth_patient.png"
        self.assertTrue(crop_file.exists())


if __name__ == "__main__":
    unittest.main()
