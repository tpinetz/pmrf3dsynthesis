import unittest
import torch
import numpy as np

from pmrf3dsynthesis.data import constants as C
from pmrf3dsynthesis.models.slice_noise import make_slice_noise, draw_slice_endpoints


class TestSliceNoise(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.shape = (1, 1, 33, 64, 64)  # D=33 so midpoint is exact index 16

    def test_endpoint_drawing_and_shape(self):
        g = torch.Generator().manual_seed(42)
        z0, z1 = draw_slice_endpoints(self.shape, generator=g)
        self.assertEqual(z0.shape, (1, 1, 64, 64))
        self.assertEqual(z1.shape, (1, 1, 64, 64))

    def test_endpoint_identity_across_arms(self):
        """Endpoints slice 0 and slice D-1 must be bit-identical across normal, linear, linear_vm, and slerp."""
        z0, z1 = draw_slice_endpoints(self.shape, generator=torch.Generator().manual_seed(123))

        arms = [
            C.NOISE_TYPE_NORMAL,
            C.NOISE_TYPE_LINEAR,
            C.NOISE_TYPE_LINEAR_VM,
            C.NOISE_TYPE_SLERP,
        ]

        noises = {}
        for arm in arms:
            n = make_slice_noise(arm, self.shape, z0=z0, z1=z1)
            self.assertEqual(n.shape, self.shape)
            noises[arm] = n
            # Check slice 0 == z0 and slice -1 == z1
            torch.testing.assert_close(n[:, :, 0], z0, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(n[:, :, -1], z1, rtol=1e-5, atol=1e-6)

        # Check all arms match each other at endpoints
        for arm in [C.NOISE_TYPE_LINEAR, C.NOISE_TYPE_LINEAR_VM, C.NOISE_TYPE_SLERP]:
            torch.testing.assert_close(noises[arm][:, :, 0], noises[C.NOISE_TYPE_NORMAL][:, :, 0])
            torch.testing.assert_close(noises[arm][:, :, -1], noises[C.NOISE_TYPE_NORMAL][:, :, -1])

    def test_constant_noise_identical_slices(self):
        """Constant noise must have identical slices throughout the volume."""
        z0, z1 = draw_slice_endpoints(self.shape, generator=torch.Generator().manual_seed(99))
        n_const = make_slice_noise(C.NOISE_TYPE_CONSTANT, self.shape, z0=z0, z1=z1)
        for s in range(self.shape[2]):
            torch.testing.assert_close(n_const[:, :, s], z0)

    def test_marginal_variance_profile(self):
        """
        Verify statistical properties across slices over many samples:
        - normal, constant, linear_vm, slerp have marginal variance ~ 1.0 across all slices.
        - plain linear drops to ~0.5 at the midpoint (alpha=0.5).
        """
        num_samples = 1000
        H, W = 32, 32
        D = 17
        mid_idx = D // 2
        large_shape = (num_samples, 1, D, H, W)

        g = torch.Generator().manual_seed(2026)
        z0, z1 = draw_slice_endpoints(large_shape, generator=g)

        n_norm = make_slice_noise(C.NOISE_TYPE_NORMAL, large_shape, generator=g, z0=z0, z1=z1)
        n_lin = make_slice_noise(C.NOISE_TYPE_LINEAR, large_shape, z0=z0, z1=z1)
        n_vm = make_slice_noise(C.NOISE_TYPE_LINEAR_VM, large_shape, z0=z0, z1=z1)
        n_slerp = make_slice_noise(C.NOISE_TYPE_SLERP, large_shape, z0=z0, z1=z1)

        # Midpoint variances
        var_norm_mid = float(n_norm[:, 0, mid_idx].var())
        var_vm_mid = float(n_vm[:, 0, mid_idx].var())
        var_slerp_mid = float(n_slerp[:, 0, mid_idx].var())
        var_lin_mid = float(n_lin[:, 0, mid_idx].var())

        # normal, linear_vm, and slerp must maintain variance near 1.0
        self.assertAlmostEqual(var_norm_mid, 1.0, delta=0.08)
        self.assertAlmostEqual(var_vm_mid, 1.0, delta=0.08)
        self.assertAlmostEqual(var_slerp_mid, 1.0, delta=0.08)

        # plain linear variance drops to (0.5)^2 + (0.5)^2 = 0.5
        self.assertAlmostEqual(var_lin_mid, 0.5, delta=0.08)

    def test_reproducibility(self):
        g1 = torch.Generator().manual_seed(777)
        g2 = torch.Generator().manual_seed(777)

        for arm in [
            C.NOISE_TYPE_NORMAL,
            C.NOISE_TYPE_CONSTANT,
            C.NOISE_TYPE_LINEAR,
            C.NOISE_TYPE_LINEAR_VM,
            C.NOISE_TYPE_SLERP
        ]:
            n1 = make_slice_noise(arm, self.shape, generator=g1)
            n2 = make_slice_noise(arm, self.shape, generator=g2)
            torch.testing.assert_close(n1, n2)


if __name__ == "__main__":
    unittest.main()
