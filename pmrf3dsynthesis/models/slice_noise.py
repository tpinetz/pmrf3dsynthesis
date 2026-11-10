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
    """
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
        alphas = torch.linspace(0.0, 1.0, D, device=device, dtype=dtype).view(1, 1, D, 1, 1)
        noise_batches = []
        for b in range(B):
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

    else:
        valid_choices = [
            C.NOISE_TYPE_NORMAL,
            C.NOISE_TYPE_CONSTANT,
            C.NOISE_TYPE_LINEAR,
            C.NOISE_TYPE_LINEAR_VM,
            C.NOISE_TYPE_SLERP,
        ]
        raise ValueError(f"Unknown noise_type '{noise_type}'. Choices: {valid_choices}")
