from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure


def _to_3d_numpy(vol: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Converts 3D/4D/5D volume to 3D numpy array of shape (D, H, W) in [0, 1]."""
    if isinstance(vol, torch.Tensor):
        vol = vol.detach().cpu().numpy()
    while vol.ndim > 3:
        vol = vol[0]
    return np.clip(vol.astype(np.float32), 0.0, 1.0)


def compute_lag_ssim_profile(
    volume: Union[np.ndarray, torch.Tensor],
    lags: Tuple[int, ...] = (1, 2, 4, 8, 16),
    axis: int = 0
) -> Dict[int, float]:
    """
    Computes through-plane slice-to-slice SSIM profile at multiple slice lags:
    LagSSIM_k(V) = mean_{s} SSIM(V[s], V[s+k]).
    
    A high LagSSIM indicates smooth through-plane continuity.
    Comparing LagSSIM(pred) vs LagSSIM(gt) reveals whether predictions
    under-smooth (independent noise flicker) or over-smooth (shared noise streaking).
    """
    vol = _to_3d_numpy(volume)
    if axis != 0:
        vol = np.moveaxis(vol, axis, 0)

    D, H, W = vol.shape
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    results = {}
    for k in lags:
        if k >= D:
            continue
        scores = []
        for s in range(0, D - k):
            sl_a = torch.from_numpy(vol[s:s + 1, None])      # (1, 1, H, W)
            sl_b = torch.from_numpy(vol[s + k:s + k + 1, None])  # (1, 1, H, W)
            score = float(ssim_metric(sl_a, sl_b).item())
            scores.append(score)
        results[k] = float(np.mean(scores)) if scores else float('nan')

    return results


def compute_lag_ssim_comparison(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
    lags: Tuple[int, ...] = (1, 2, 4, 8, 16),
    axis: int = 0
) -> Dict[str, float]:
    """
    Computes Lag-k SSIM for prediction, ground truth, and the absolute gap:
    LagSSIM_Gap_k = |LagSSIM_k(pred) - LagSSIM_k(gt)|.
    """
    pred_profile = compute_lag_ssim_profile(pred, lags=lags, axis=axis)
    gt_profile = compute_lag_ssim_profile(gt, lags=lags, axis=axis)

    metrics = {}
    for k in lags:
        if k in pred_profile and k in gt_profile:
            p_val = pred_profile[k]
            g_val = gt_profile[k]
            metrics[f"LagSSIM_pred_k{k}"] = p_val
            metrics[f"LagSSIM_gt_k{k}"] = g_val
            metrics[f"LagSSIM_Gap_k{k}"] = float(abs(p_val - g_val))

    # Primary adjacent-slice scalar (k=1)
    if 1 in pred_profile and 1 in gt_profile:
        metrics["Adjacent_SSIM_Gap"] = metrics["LagSSIM_Gap_k1"]
        metrics["Adjacent_SSIM_pred"] = metrics["LagSSIM_pred_k1"]
        metrics["Adjacent_SSIM_gt"] = metrics["LagSSIM_gt_k1"]

    return metrics


def compute_through_plane_psd(
    volume: Union[np.ndarray, torch.Tensor],
    axis: int = 0
) -> np.ndarray:
    """
    Computes 1D power spectral density (PSD) along the through-plane slice axis,
    averaged over all in-plane pixels (H, W).
    """
    vol = _to_3d_numpy(volume)
    if axis != 0:
        vol = np.moveaxis(vol, axis, 0)

    # vol is (D, H, W). RFFT along axis 0
    fft_vals = np.fft.rfft(vol, axis=0)  # shape (D//2 + 1, H, W)
    power = np.abs(fft_vals) ** 2
    avg_psd = np.mean(power, axis=(1, 2))  # shape (D//2 + 1,)
    return avg_psd


def compute_spectral_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
    axis: int = 0,
    eps: float = 1e-10
) -> Dict[str, float]:
    """
    Computes Log-Spectral Distance (LSD) and High-Frequency Energy Ratio (HFR)
    along the through-plane slice axis.
    """
    psd_pred = compute_through_plane_psd(pred, axis=axis)
    psd_gt = compute_through_plane_psd(gt, axis=axis)

    log_pred = np.log10(psd_pred + eps)
    log_gt = np.log10(psd_gt + eps)

    # Root mean squared log-spectral distance
    lsd = float(np.sqrt(np.mean((log_pred - log_gt) ** 2)))

    # High frequency fraction (upper half of spectrum)
    num_bins = len(psd_pred)
    cutoff = num_bins // 2
    hfr_pred = float(np.sum(psd_pred[cutoff:]) / (np.sum(psd_pred) + eps))
    hfr_gt = float(np.sum(psd_gt[cutoff:]) / (np.sum(psd_gt) + eps))
    hfr_gap = float(abs(hfr_pred - hfr_gt))

    return {
        "Log_Spectral_Distance": lsd,
        "HFR_pred": hfr_pred,
        "HFR_gt": hfr_gt,
        "HFR_gap": hfr_gap,
    }


def compute_orthogonal_reslice_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
    lpips_fn: Optional[torch.nn.Module] = None,
    device: str = 'cpu',
    num_samples: int = 16
) -> Dict[str, float]:
    """
    Evaluates 2D image quality (SSIM and optional LPIPS) on orthogonal reslices:
    - Coronal reslices: plane through (D, W) sampled along H
    - Sagittal reslices: plane through (D, H) sampled along W
    
    Through-plane artefacts (such as slice flicker or vertical streaking) appear
    as severe horizontal stripes in these orthogonal views.
    """
    p_vol = _to_3d_numpy(pred)
    g_vol = _to_3d_numpy(gt)

    D, H, W = p_vol.shape
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Subsample slices to keep evaluation fast and representative
    h_indices = np.linspace(0, H - 1, min(num_samples, H), dtype=int)
    w_indices = np.linspace(0, W - 1, min(num_samples, W), dtype=int)

    coronal_ssims = []
    sagittal_ssims = []
    coronal_lpips = []
    sagittal_lpips = []

    # Coronal: plane along (D, W)
    for h in h_indices:
        p_slice = torch.from_numpy(p_vol[:, h, :][None, None]).to(device)  # (1, 1, D, W)
        g_slice = torch.from_numpy(g_vol[:, h, :][None, None]).to(device)
        coronal_ssims.append(float(ssim_metric(p_slice, g_slice).item()))

        if lpips_fn is not None:
            p_3ch = p_slice.repeat(1, 3, 1, 1)
            g_3ch = g_slice.repeat(1, 3, 1, 1)
            with torch.no_grad():
                val = lpips_fn(p_3ch, g_3ch).item()
                coronal_lpips.append(val)

    # Sagittal: plane along (D, H)
    for w in w_indices:
        p_slice = torch.from_numpy(p_vol[:, :, w][None, None]).to(device)  # (1, 1, D, H)
        g_slice = torch.from_numpy(g_vol[:, :, w][None, None]).to(device)
        sagittal_ssims.append(float(ssim_metric(p_slice, g_slice).item()))

        if lpips_fn is not None:
            p_3ch = p_slice.repeat(1, 3, 1, 1)
            g_3ch = g_slice.repeat(1, 3, 1, 1)
            with torch.no_grad():
                val = lpips_fn(p_3ch, g_3ch).item()
                sagittal_lpips.append(val)

    c_ssim = float(np.mean(coronal_ssims))
    s_ssim = float(np.mean(sagittal_ssims))
    metrics = {
        "Orthogonal_SSIM_coronal": c_ssim,
        "Orthogonal_SSIM_sagittal": s_ssim,
        "Orthogonal_SSIM_mean": float(0.5 * (c_ssim + s_ssim))
    }

    if lpips_fn is not None and coronal_lpips and sagittal_lpips:
        c_lpips = float(np.mean(coronal_lpips))
        s_lpips = float(np.mean(sagittal_lpips))
        metrics["Orthogonal_LPIPS_coronal"] = c_lpips
        metrics["Orthogonal_LPIPS_sagittal"] = s_lpips
        metrics["Orthogonal_LPIPS_mean"] = float(0.5 * (c_lpips + s_lpips))

    return metrics


def compute_flow_displacement(
    pred: Union[np.ndarray, torch.Tensor],
    mmse_pred: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    Measures the magnitude of the 2D flow displacement relative to the initial MMSE estimator:
    ||y_hat - x_MMSE||.
    """
    p = _to_3d_numpy(pred)
    m = _to_3d_numpy(mmse_pred)

    diff = p - m
    l1_disp = float(np.mean(np.abs(diff)))
    l2_disp = float(np.sqrt(np.mean(diff ** 2)))

    return {
        "Flow_Displacement_L1": l1_disp,
        "Flow_Displacement_L2": l2_disp,
    }


def evaluate_interslice_structure(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
    mmse_pred: Optional[Union[np.ndarray, torch.Tensor]] = None,
    lpips_fn: Optional[torch.nn.Module] = None,
    device: str = 'cpu',
    lags: Tuple[int, ...] = (1, 2, 4, 8, 16)
) -> Dict[str, float]:
    """
    Computes all direct inter-slice structural metrics in one call:
    1. Lag-k SSIM profile and gap to ground truth
    2. Log-Spectral Distance (LSD) and High-Frequency Energy Ratio
    3. Orthogonal reslice SSIM and LPIPS
    4. Flow displacement from MMSE (if mmse_pred provided)
    """
    results = {}
    # 1. Lag SSIM
    results.update(compute_lag_ssim_comparison(pred, gt, lags=lags, axis=0))
    # 2. Spectral metrics
    results.update(compute_spectral_metrics(pred, gt, axis=0))
    # 3. Orthogonal reslice metrics
    results.update(compute_orthogonal_reslice_metrics(pred, gt, lpips_fn=lpips_fn, device=device))
    # 4. Flow displacement
    if mmse_pred is not None:
        results.update(compute_flow_displacement(pred, mmse_pred))

    return results
