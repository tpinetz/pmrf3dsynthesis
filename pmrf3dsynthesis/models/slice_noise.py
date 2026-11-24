import argparse
import re
from typing import Optional, Tuple, Union
import torch
from pmrf3dsynthesis.data import constants as C


def draw_slice_endpoints(
    shape: Union[Tuple[int, ...], torch.Size],
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Draws a pair of standard normal endpoint tensors (z0, z1) of shape (B, C, H, W).
    These endpoints can be shared across all noise coupling arms to enable paired randomness
    and common random numbers.
    """
    if len(shape) == 5:
        B, C_ch, _, H, W = shape
        end_shape = (B, C_ch, H, W)
    elif len(shape) == 4:
        B, C_ch, H, W = shape
        end_shape = (B, C_ch, H, W)
    else:
        raise ValueError(f"Expected 4D or 5D shape, got {shape}")

    z0 = torch.randn(end_shape, generator=generator, device=device, dtype=dtype)
    z1 = torch.randn(end_shape, generator=generator, device=device, dtype=dtype)
    return z0, z1


_FREQSPLIT_RE = re.compile(r"^(freqsplit(?:_slerp)?)(?:_fc([0-9]*\.?[0-9]+))?$")


def parse_freqsplit(noise_type: str) -> Optional[Tuple[str, float]]:
    """Returns (low-band source, cutoff) for a frequency-split noise name, else None.

    'freqsplit' -> ('constant', 0.1); 'freqsplit_slerp_fc0.2' -> ('slerp', 0.2).
    """
    m = _FREQSPLIT_RE.match(noise_type)
    if m is None:
        return None
    cutoff = float(m.group(2)) if m.group(2) else C.FREQSPLIT_DEFAULT_CUTOFF
    if not 0.0 < cutoff:
        raise ValueError(f"freqsplit cutoff must be positive, got {cutoff}")
    source = C.NOISE_TYPE_SLERP if m.group(1) == C.NOISE_TYPE_FREQSPLIT_SLERP else C.NOISE_TYPE_CONSTANT
    return source, cutoff


def is_valid_noise_type(noise_type: str) -> bool:
    return noise_type in (
        C.NOISE_TYPE_NORMAL, C.NOISE_TYPE_CONSTANT, C.NOISE_TYPE_LINEAR,
        C.NOISE_TYPE_LINEAR_VM, C.NOISE_TYPE_SLERP,
    ) or parse_freqsplit(noise_type) is not None


def noise_type_arg(value: str) -> str:
    """argparse type for --noise_type: the fixed names plus freqsplit[_slerp][_fc<cutoff>]."""
    if not is_valid_noise_type(value):
        raise argparse.ArgumentTypeError(
            f"unknown noise type '{value}' (normal, constant, linear, linear_vm, slerp, "
            "freqsplit[_fc<cutoff>], freqsplit_slerp[_fc<cutoff>])"
        )
    return value


def freqsplit_gains(H: int, W: int, cutoff: float, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Power-complementary low/high gains on the rfft2 grid (H, W // 2 + 1), float32."""
    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.rfftfreq(W, device=device)
    r2 = fy[:, None] ** 2 + fx[None, :] ** 2
    low = torch.exp(-r2 / (2.0 * cutoff ** 2))
    high = torch.sqrt(torch.clamp(1.0 - low ** 2, min=0.0))
    return low, high


def _apply_gain(z: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """Filters the last two axes of a real tensor with a real, radially symmetric gain."""
    H, W = z.shape[-2:]
    spec = torch.fft.rfft2(z.float(), dim=(-2, -1)) * gain
    return torch.fft.irfft2(spec, s=(H, W), dim=(-2, -1)).to(z.dtype)


def _slerp_volume(z0: torch.Tensor, z1: torch.Tensor, D: int, eps: float) -> torch.Tensor:
    """Slerp between per-batch endpoints z0, z1 (B, C, H, W) over D slices -> (B, C, D, H, W)."""
    alphas = torch.linspace(0.0, 1.0, D, device=z0.device, dtype=z0.dtype).view(1, 1, D, 1, 1)
    noise_batches = []
    for b in range(z0.shape[0]):
        z0_b = z0[b]  # (C, H, W)
        z1_b = z1[b]  # (C, H, W)
        z0_flat = z0_b.reshape(-1)
        z1_flat = z1_b.reshape(-1)
        norm_0 = z0_flat.norm()
        norm_1 = z1_flat.norm()

        cos_omega = torch.dot(z0_flat, z1_flat) / (norm_0 * norm_1 + eps)
        cos_omega = torch.clamp(cos_omega, -1.0 + 1e-7, 1.0 - 1e-7)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega)

        # Slerp weights across depth: shape (1, D, 1, 1)
        alpha_d = alphas[0]  # (1, D, 1, 1)
        w1 = torch.sin((1.0 - alpha_d) * omega) / (sin_omega + eps)
        w2 = torch.sin(alpha_d * omega) / (sin_omega + eps)

        # (1, D, 1, 1) * (C, 1, H, W) -> (C, D, H, W)
        noise_b = w1 * z0_b.unsqueeze(1) + w2 * z1_b.unsqueeze(1)
        noise_batches.append(noise_b)

    noise = torch.stack(noise_batches, dim=0)  # (B, C, D, H, W)
    return noise


def make_slice_noise(
    noise_type: str,
    shape: Union[Tuple[int, ...], torch.Size],
    generator: Optional[torch.Generator] = None,
    z0: Optional[torch.Tensor] = None,
    z1: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Generates a 5D noise volume of shape (B, C, D, H, W) with different through-plane slice correlations:
    
    1. 'normal' (independent):
       Each slice s has independent N(0, I) noise. If z0 and z1 are supplied, slice 0 is z0
       and slice D-1 is z1, providing paired endpoint alignment.
       
    2. 'constant' (shared):
       Every slice s has the identical noise field z0 across the entire volume depth.
       
    3. 'linear' (plain linear mixing):
       epsilon_s = (1 - alpha_s) * z0 + alpha_s * z1.
       Endpoints have variance 1, but mid-volume variance shrinks to 0.5.
       
    4. 'linear_vm' (variance-matched linear mixing):
       epsilon_s = ((1 - alpha_s) * z0 + alpha_s * z1) / sqrt((1 - alpha_s)^2 + alpha_s^2).
       Preserves unit marginal variance Var(epsilon_s) = 1 at all slices.
       
    5. 'slerp' (Spherical Linear Interpolation):
       Interpolates along the great-circle arc between z0 and z1 on the high-dimensional sphere.
       Preserves unit norm/variance across slices.

    6. 'freqsplit' / 'freqsplit_slerp' (frequency-split coupling):
       Splits every slice into a low and a high in-plane (B-scan) frequency band with
       power-complementary Gaussian gains, L(f) = exp(-|f|^2 / (2 fc^2)) and
       Hi(f) = sqrt(1 - L(f)^2). The low band is shared by all slices (L * z0 for
       'freqsplit') or follows the slerp path (L * slerp(z0, z1) for 'freqsplit_slerp');
       the high band is fresh N(0, I) noise per slice. Because L^2 + Hi^2 = 1 the
       independent-band sum is again white, so each 'freqsplit' slice is exactly N(0, I)
       (and 'freqsplit_slerp' matches slerp's marginal variance). Append '_fc<cutoff>'
       (cycles/pixel, default C.FREQSPLIT_DEFAULT_CUTOFF) to pick the cutoff. The low band
       holds about pi * fc^2 of the power (fc=0.1: 3%, fc=0.2: 13%, fc=0.3: 28%).
    """
    freqsplit = parse_freqsplit(noise_type)
    if len(shape) != 5:
        raise ValueError(f"make_slice_noise expects shape (B, C, D, H, W), got {shape}")

    B, C_ch, D, H, W = shape

    if device is None and z0 is not None:
        device = z0.device
    if dtype is None and z0 is not None:
        dtype = z0.dtype

    # Draw endpoints if not provided
    if z0 is None or z1 is None:
        rand_z0, rand_z1 = draw_slice_endpoints(shape, generator=generator, device=device, dtype=dtype)
        if z0 is None:
            z0 = rand_z0
        if z1 is None:
            z1 = rand_z1

    # Ensure shape (B, C, H, W)
    if z0.ndim == 3:
        z0 = z0.unsqueeze(1)
    if z1.ndim == 3:
        z1 = z1.unsqueeze(1)

    if noise_type == C.NOISE_TYPE_NORMAL:
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        # Pin endpoints to z0 and z1 for paired seed comparison
        noise[:, :, 0, :, :] = z0
        noise[:, :, -1, :, :] = z1
        return noise

    elif noise_type == C.NOISE_TYPE_CONSTANT:
        return z0.unsqueeze(2).expand(B, C_ch, D, H, W).clone()

    elif noise_type == C.NOISE_TYPE_LINEAR:
        # alphas in [0, 1] across depth axis
        alphas = torch.linspace(0.0, 1.0, D, device=device, dtype=dtype).view(1, 1, D, 1, 1)
        z0_5d = z0.unsqueeze(2)  # (B, C, 1, H, W)
        z1_5d = z1.unsqueeze(2)  # (B, C, 1, H, W)
        noise = (1.0 - alphas) * z0_5d + alphas * z1_5d
        return noise

    elif noise_type == C.NOISE_TYPE_LINEAR_VM:
        alphas = torch.linspace(0.0, 1.0, D, device=device, dtype=dtype).view(1, 1, D, 1, 1)
        norm_factor = torch.sqrt((1.0 - alphas) ** 2 + alphas ** 2)
        z0_5d = z0.unsqueeze(2)
        z1_5d = z1.unsqueeze(2)
        noise = ((1.0 - alphas) * z0_5d + alphas * z1_5d) / norm_factor
        return noise

    elif noise_type == C.NOISE_TYPE_SLERP:
        return _slerp_volume(z0, z1, D, eps)

    elif freqsplit is not None:
        low_source, cutoff = freqsplit
        low_gain, high_gain = freqsplit_gains(H, W, cutoff, device=z0.device)
        if low_source == C.NOISE_TYPE_SLERP:
            low = _apply_gain(_slerp_volume(z0, z1, D, eps), low_gain)
        else:
            low = _apply_gain(z0, low_gain).unsqueeze(2)
        independent = torch.randn(shape, generator=generator, device=z0.device, dtype=z0.dtype)
        return low + _apply_gain(independent, high_gain)

    else:
        valid_choices = [
            C.NOISE_TYPE_NORMAL,
            C.NOISE_TYPE_CONSTANT,
            C.NOISE_TYPE_LINEAR,
            C.NOISE_TYPE_LINEAR_VM,
            C.NOISE_TYPE_SLERP,
            C.NOISE_TYPE_FREQSPLIT,
            C.NOISE_TYPE_FREQSPLIT_SLERP,
        ]
        raise ValueError(f"Unknown noise_type '{noise_type}'. Choices: {valid_choices}")
