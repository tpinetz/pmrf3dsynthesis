"""TransUpNet full-resolution fundus stem (fundus_size) and the eval helper fundus_input_size."""
import pytest
import torch
import torchvision

from pmrf3dsynthesis.models import transunet
from pmrf3dsynthesis.models.transupnet import FundusStem, TransUpNet, fundus_input_size


@pytest.fixture(autouse=True)
def no_pretrained_download(monkeypatch):
    # The ViT neck loads ImageNet weights; random weights are enough for shape tests.
    real_vit = torchvision.models.vit_b_16
    monkeypatch.setattr(transunet.torchvision.models, "vit_b_16", lambda weights=None: real_vit(weights=None))


# Small grid so the 3D decoder fits on CPU: en-face 16x32 (2x4 tokens), 64 slices.
ENFACE = (16, 32)
SLICES = 64


def test_default_model_has_no_stem_and_unchanged_keys():
    model = TransUpNet(in_channels=3, slides=SLICES)
    assert model.stem is None and model.fundus_size is None
    assert not any(k.startswith("stem.") for k in model.state_dict())
    out = model(torch.randn(1, 3, *ENFACE))[-1]
    assert out.shape == (1, 1, ENFACE[0], SLICES, ENFACE[1])


def test_full_resolution_input_maps_to_enface_grid():
    model = TransUpNet(in_channels=3, slides=SLICES, fundus_size=(64, 64), enface_size=ENFACE, stem_channels=16)
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    out = model(x)[-1]
    # Same output grid as the default model fed a 16x32 fundus.
    assert out.shape == (1, 1, ENFACE[0], SLICES, ENFACE[1])
    out.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.stem.net[0].weight.grad is not None


def test_wrong_input_size_raises():
    model = TransUpNet(in_channels=3, slides=SLICES, fundus_size=(64, 64), enface_size=ENFACE, stem_channels=16)
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, *ENFACE))


@pytest.mark.parametrize("fundus_size", [(53, 53), (48, 64)])
def test_non_power_of_two_sizes_map_to_enface_grid(fundus_size):
    model = TransUpNet(in_channels=3, slides=SLICES, fundus_size=fundus_size, enface_size=ENFACE, stem_channels=16)
    out = model(torch.randn(1, 3, *fundus_size))[-1]
    assert out.shape == (1, 1, ENFACE[0], SLICES, ENFACE[1])


def test_stem_on_850_crop():
    # The saved registered crops are 850x850: strided stages to 213x425, then pooled to 128x256.
    stem = FundusStem(3, 8, (850, 850), (128, 256))
    assert isinstance(stem.net[-1], torch.nn.AdaptiveAvgPool2d)
    assert stem(torch.randn(1, 3, 850, 850)).shape == (1, 8, 128, 256)


def test_power_of_two_crop_needs_no_pooling():
    stem = FundusStem(3, 8, (512, 512), (128, 256))
    assert not any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in stem.net)
    assert stem(torch.randn(1, 3, 512, 512)).shape == (1, 8, 128, 256)


def test_fundus_smaller_than_grid_raises():
    with pytest.raises(ValueError):
        TransUpNet(in_channels=3, slides=SLICES, fundus_size=(8, 64), enface_size=ENFACE)


def test_fundus_input_size_helper():
    class Wrapper(torch.nn.Module):
        def __init__(self, network, mmse_model=None):
            super().__init__()
            self.network = network
            self.mmse_model = mmse_model

    legacy = Wrapper(TransUpNet(in_channels=3, slides=SLICES))
    full = Wrapper(TransUpNet(in_channels=3, slides=SLICES, fundus_size=(64, 64), enface_size=ENFACE))
    assert fundus_input_size(legacy) == (128, 256)
    assert fundus_input_size(full) == (64, 64)
    # A flow model asks its MMSE model; without one it falls back to the en-face grid.
    assert fundus_input_size(Wrapper(torch.nn.Identity(), mmse_model=full)) == (64, 64)
    assert fundus_input_size(Wrapper(torch.nn.Identity())) == (128, 256)
    assert fundus_input_size(None) == (128, 256)
