"""Frequency-split slice noise: unit marginals, shared low band, independent high band."""
import argparse

import pytest
import torch

from pmrf3dsynthesis.data import constants as C
from pmrf3dsynthesis.models.slice_noise import (
    _apply_gain,
    draw_slice_endpoints,
    freqsplit_gains,
    make_slice_noise,
    noise_type_arg,
    parse_freqsplit,
)


def test_parse_names():
    assert parse_freqsplit("freqsplit") == (C.NOISE_TYPE_CONSTANT, C.FREQSPLIT_DEFAULT_CUTOFF)
    assert parse_freqsplit("freqsplit_fc0.2") == (C.NOISE_TYPE_CONSTANT, 0.2)
    assert parse_freqsplit("freqsplit_slerp") == (C.NOISE_TYPE_SLERP, C.FREQSPLIT_DEFAULT_CUTOFF)
    assert parse_freqsplit("freqsplit_slerp_fc.05") == (C.NOISE_TYPE_SLERP, 0.05)
    assert parse_freqsplit("slerp") is None
    assert parse_freqsplit("freqsplit_fc") is None
    assert noise_type_arg("freqsplit_slerp_fc0.3") == "freqsplit_slerp_fc0.3"
    assert noise_type_arg("slerp") == "slerp"
    with pytest.raises(argparse.ArgumentTypeError):
        noise_type_arg("freqsplit_foo")


def test_gains_are_power_complementary():
    low, high = freqsplit_gains(64, 48, 0.1)
    assert low.shape == (64, 25)
    torch.testing.assert_close(low ** 2 + high ** 2, torch.ones_like(low))
    assert low[0, 0] == 1.0 and high[0, 0] == 0.0


def test_gain_filter_is_identity_split():
    z = torch.randn(3, 1, 64, 64)
    low, high = freqsplit_gains(64, 64, 0.1)
    # Amplitudes do not add to one, but power does: filtering twice with each and summing gives z.
    recon = _apply_gain(_apply_gain(z, low), low) + _apply_gain(_apply_gain(z, high), high)
    torch.testing.assert_close(recon, z, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("arm", ["freqsplit", "freqsplit_fc0.2", "freqsplit_slerp_fc0.2"])
def test_shape_and_unit_marginal_variance(arm):
    g = torch.Generator().manual_seed(0)
    shape = (400, 1, 9, 32, 32)
    n = make_slice_noise(arm, shape, generator=g)
    assert n.shape == shape and n.dtype == torch.float32 and torch.isfinite(n).all()
    var = n.var(dim=0).mean(dim=(-2, -1))[0]  # per-slice variance over samples
    torch.testing.assert_close(var, torch.ones_like(var), atol=0.05, rtol=0)


def _spec(x):
    return torch.fft.rfft2(x, dim=(-2, -1))


def _residual_power(residual):
    """Per-bin power of a (B, C, H, W) field, averaged over samples, normalised so white noise = 1."""
    H, W = residual.shape[-2:]
    return (_spec(residual).abs() ** 2).mean(dim=(0, 1)) / (H * W)


def test_freqsplit_slices_share_only_the_low_band():
    shape = (300, 1, 3, 32, 32)
    cutoff = 0.15
    g = torch.Generator().manual_seed(1)
    z0, z1 = draw_slice_endpoints(shape, generator=g)
    n = make_slice_noise(f"freqsplit_fc{cutoff}", shape, generator=g, z0=z0, z1=z1)
    low, high = freqsplit_gains(32, 32, cutoff)
    for s in range(shape[2]):
        # What is left after removing the shared low band is independent noise with power Hi^2.
        power = _residual_power(n[:, :, s] - _apply_gain(z0, low))
        torch.testing.assert_close(power, high ** 2, atol=0.25, rtol=0)
    # Slice differences carry no DC and lose the low band: power 2 * Hi^2.
    power = _residual_power(n[:, :, 0] - n[:, :, 2])
    torch.testing.assert_close(power, 2 * high ** 2, atol=0.4, rtol=0)
    assert power[0, 0] < 1e-6
    # Adjacent slices correlate by the shared power fraction, about pi * fc^2 (~7%).
    a, b = n[:, 0, 0].flatten(), n[:, 0, 1].flatten()
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1]
    assert 0.05 < corr < 0.09


def test_freqsplit_slerp_low_band_follows_slerp():
    shape = (300, 1, 5, 32, 32)
    g = torch.Generator().manual_seed(2)
    z0, z1 = draw_slice_endpoints(shape, generator=g)
    n = make_slice_noise("freqsplit_slerp_fc0.2", shape, generator=g, z0=z0, z1=z1)
    slerp = make_slice_noise(C.NOISE_TYPE_SLERP, shape, z0=z0, z1=z1)
    low, high = freqsplit_gains(32, 32, 0.2)
    for s in (0, 2, 4):
        power = _residual_power(n[:, :, s] - _apply_gain(slerp[:, :, s], low))
        torch.testing.assert_close(power, high ** 2, atol=0.25, rtol=0)


def test_reproducible_with_seed():
    shape = (1, 1, 4, 32, 32)
    a = make_slice_noise("freqsplit_slerp", shape, generator=torch.Generator().manual_seed(5))
    b = make_slice_noise("freqsplit_slerp", shape, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(a, b)
