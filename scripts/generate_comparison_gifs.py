import argparse
import io
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from pmrf3dsynthesis.eval_model import load_sample



def generate_synthesis_gif(
    case_key: str,
    modality: str,
    gt_vol: np.ndarray,
    input_img: np.ndarray,
    mmse_vol: np.ndarray,
    noslerp_vol: np.ndarray,
    slerp_vol: np.ndarray,
    output_path: str,
    steps: int,
    metrics_str: str = "",
    fps: float = 8.0,
    duration: Optional[int] = 120,
    ping_pong: bool = True,
    show_slice_line: Optional[bool] = None,
    target_size: Optional[Tuple[int, int]] = None,
    panels: Optional[list] = None
):
    """
    Generates an animated GIF comparing synthesis panels.
    Default panels: [2D Input] | [Ground Truth] | [MMSE Estimate] | [Without Slerp Noise] | [With Slerp Noise (Ours)]
    """
    if panels is None:
        panels = ["input", "gt", "mmse", "noslerp", "slerp"]

    if target_size is None:
        target_size = (352 * len(panels), 416)

    gt_vol = np.clip(gt_vol, 0.0, 1.0)
    mmse_vol = np.clip(mmse_vol, 0.0, 1.0)
    noslerp_vol = np.clip(noslerp_vol, 0.0, 1.0)
    slerp_vol = np.clip(slerp_vol, 0.0, 1.0)

    num_slices = gt_vol.shape[0]
    frames = []

    plane_label = "B-scan" if modality.lower() == "oct" else "Axial Slice"
    input_label = "2D Input Fundus" if modality.lower() == "oct" else "2D Input X-ray (DRR)"
    mmse_label = "MMSE Estimate\n(TransUpNet)" if modality.lower() == "oct" else "MMSE Estimate\n(1-step Expectation)"

    if show_slice_line is None:
        show_slice_line = (modality.lower() == "oct")

    display_key = case_key if len(case_key) <= 20 else f"{case_key[:8]}-..."
    frame_duration = duration if duration is not None else int(1000 / fps)

    print(f"Rendering {num_slices} frames for {case_key} ({modality.upper()}, {steps} steps, panels={panels})...")

    panel_defs = {
        "input": {
            "title": input_label,
            "title_color": "white",
        },
        "gt": {
            "title": f"Ground Truth\n(3D {modality.upper()})",
            "title_color": "white",
            "vol": gt_vol,
        },
        "mmse": {
            "title": mmse_label,
            "title_color": "#93c5fd",
            "vol": mmse_vol,
        },
        "noslerp": {
            "title": f"Without Slerp Noise\n({steps} steps, Gaussian)",
            "title_color": "#fcd34d",
            "vol": noslerp_vol,
        },
        "slerp": {
            "title": f"With Slerp Noise (Ours)\n({steps} steps)",
            "title_color": "#4ade80",
            "vol": slerp_vol,
        },
    }

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.4 * n_panels, 5.2), dpi=120)
    if n_panels == 1:
        axes = [axes]
    else:
        axes = list(axes)
    fig.patch.set_facecolor('#0b1120')
    plt.subplots_adjust(top=0.68, bottom=0.06, left=0.02, right=0.98, wspace=0.14)

    # Main header
    fig.text(0.5, 0.93, f"{modality.upper()} 3D Synthesis: Case {display_key}", 
             color='white', fontsize=16, fontweight='bold', ha='center')

    # Subtitle
    sub_artist = fig.text(0.5, 0.81, "", color='#94a3b8', fontsize=12, ha='center')

    slice_line = None
    im_artists = {}

    for i, p_key in enumerate(panels):
        p_info = panel_defs[p_key]
        ax = axes[i]
        if p_key == "input":
            if input_img.ndim == 2:
                ax.imshow(input_img, cmap='gray', aspect='equal')
            else:
                ax.imshow(input_img, aspect='auto')
            if show_slice_line:
                slice_line = ax.axhline(y=0, color='white', linewidth=2.0)
        else:
            vol = p_info["vol"]
            im = ax.imshow(vol[0], cmap='gray', vmin=0, vmax=1, aspect='equal')
            im_artists[p_key] = im

        ax.set_title(p_info["title"], color=p_info["title_color"], fontsize=12, fontweight='bold', pad=12)
        ax.axis('off')

    for sl in range(num_slices):
        sub_text = f"{plane_label} {sl+1}/{num_slices}"
        if metrics_str:
            sub_text += f"  |  {metrics_str}"
        sub_artist.set_text(sub_text)

        if slice_line is not None:
            slice_line.set_ydata([sl, sl])

        for p_key, im in im_artists.items():
            im.set_data(panel_defs[p_key]["vol"][sl])

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img_pil = Image.fromarray(rgba).convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
        frames.append(img_pil)

    plt.close(fig)

    if ping_pong:
        all_frames = frames + frames[-2::-1]
    else:
        all_frames = frames

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    all_frames[0].save(
        out_file,
        save_all=True,
        append_images=all_frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True
    )
    print(f"Successfully saved animation to: {out_file} ({len(all_frames)} frames at duration={frame_duration}ms, size={target_size})")


def get_oct_metrics(case_key: str, slerp_dir: Optional[Path] = None) -> str:
    """Retrieves evaluation metrics for an OCT test case from eval_results.pkl."""
    if slerp_dir is None:
        oct_base = Path("/media/thomas-pinetz/data/data/OCT/results")
        slerp_dir = oct_base / "2026-02-16_12-07-23_ms_cleaned_song_sigma_01_rflow_with_20_steps_slerp_noise"
    
    pkl_file = slerp_dir / "eval_results.pkl"
    if not pkl_file.exists():
        return ""

    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        psnr = data.get("PSNR", {}).get(case_key)
        ssim = data.get("SSIM", {}).get(case_key)
        vessel = data.get("Vessel_Dice", {}).get(case_key)

        parts = []
        if psnr is not None and np.isfinite(psnr):
            parts.append(f"PSNR {psnr:.2f} dB")
        if ssim is not None and np.isfinite(ssim):
            parts.append(f"SSIM {ssim:.4f}")
        if vessel is not None and np.isfinite(vessel):
            parts.append(f"Vessel Dice {vessel:.4f}")

        if parts:
            return f"Slerp (20 steps): " + ", ".join(parts)
    except Exception as e:
        print(f"Warning: Could not read metrics for {case_key}: {e}")

    return ""


def get_top_oct_cases(n: int = 5, rank_by: str = "vessel") -> List[str]:
    """Returns top N OCT case keys ranked by Vessel_Dice or PSNR."""
    oct_base = Path("/media/thomas-pinetz/data/data/OCT/results")
    slerp_dir = oct_base / "2026-02-16_12-07-23_ms_cleaned_song_sigma_01_rflow_with_20_steps_slerp_noise"
    pkl_file = slerp_dir / "eval_results.pkl"
    if not pkl_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {pkl_file}")

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    metric_name = "Vessel_Dice" if rank_by.lower() == "vessel" else "PSNR"
    metric_dict = data.get(metric_name, {})
    sorted_cases = sorted(
        metric_dict.keys(),
        key=lambda k: metric_dict[k] if np.isfinite(metric_dict[k]) else -1e9,
        reverse=True
    )
    return sorted_cases[:n]


def generate_oct_case_gif(
    case_key: str,
    output_path: str,
    panels: Optional[list] = None,
    target_size: Optional[Tuple[int, int]] = (1760, 416),
    fps: float = 8.0,
    duration: int = 120,
    show_slice_line: bool = True
):
    """Generates an animated comparison GIF for an OCT test case."""
    in_oct, gt_oct = load_sample(case_key, "oct")

    oct_base = Path("/media/thomas-pinetz/data/data/OCT/results")
    slerp_dir = oct_base / "2026-02-16_12-07-23_ms_cleaned_song_sigma_01_rflow_with_20_steps_slerp_noise"
    noslerp_dir = oct_base / "2026-02-16_12-07-23_ms_cleaned_song_sigma_01_rflow_with_20_steps"
    mmse_dir = oct_base / "mmse/2026-01-23_11-33-50_ms_mmse_added_augs_transupnet"

    slerp_file = slerp_dir / f"{case_key}.npy"
    noslerp_file = noslerp_dir / f"{case_key}.npy"
    mmse_file = mmse_dir / f"{case_key}.npy"

    if not slerp_file.exists():
        raise FileNotFoundError(f"Slerp prediction file not found: {slerp_file}")
    if not noslerp_file.exists():
        raise FileNotFoundError(f"No-slerp prediction file not found: {noslerp_file}")
    if not mmse_file.exists():
        raise FileNotFoundError(f"MMSE prediction file not found: {mmse_file}")

    slerp_vol = np.load(slerp_file)
    noslerp_vol = np.load(noslerp_file)
    mmse_vol = np.load(mmse_file)

    if slerp_vol.min() < 0:
        slerp_vol = slerp_vol * 0.5 + 0.5
    if noslerp_vol.min() < 0:
        noslerp_vol = noslerp_vol * 0.5 + 0.5
    if mmse_vol.min() < 0:
        mmse_vol = mmse_vol * 0.5 + 0.5

    metrics_str = get_oct_metrics(case_key, slerp_dir)
    generate_synthesis_gif(
        case_key=case_key,
        modality="oct",
        gt_vol=gt_oct,
        input_img=in_oct,
        mmse_vol=mmse_vol,
        noslerp_vol=noslerp_vol,
        slerp_vol=slerp_vol,
        output_path=output_path,
        steps=20,
        metrics_str=metrics_str,
        fps=fps,
        duration=duration,
        show_slice_line=show_slice_line,
        target_size=target_size,
        panels=panels
    )


def generate_best_oct_gif(
    output_path: str = "assets/test_case_oct.gif",
    panels: Optional[list] = None,
    target_size: Optional[Tuple[int, int]] = (1760, 416)
):
    case_key = "58049930-25-GIAPKAIDVOKAXAGZMAFILIRCK+BSIMVTVVRKFLXIBHVBLXGTPUICTMNDWJCZYZT+INUBYN"
    return generate_oct_case_gif(
        case_key=case_key,
        output_path=output_path,
        panels=panels,
        target_size=target_size
    )


def generate_best_oct_3panel_gif(
    output_path: str = "assets/test_case_oct_gt_mmse_ours.gif",
    target_size: Optional[Tuple[int, int]] = None
):
    return generate_best_oct_gif(
        output_path=output_path,
        panels=["gt", "mmse", "slerp"],
        target_size=target_size
    )


def generate_best_ct_gif(
    output_path: str = "assets/test_case_ct.gif",
    panels: Optional[list] = None,
    target_size: Optional[Tuple[int, int]] = (1760, 416)
):
    case_key = "LIDC-IDRI-0177"
    in_ct, gt_ct = load_sample(case_key, "ct")

    ct_base = Path("/media/thomas-pinetz/data/data/CT/results")
    slerp_file = ct_base / "2026-06-30_11-24-06_ms_rflow_ct_float32_with_10_steps_slerm_noise" / f"{case_key}.npy"
    noslerp_file = ct_base / "2026-06-30_11-24-06_ms_rflow_ct_float32_with_10_steps" / f"{case_key}.npy"
    mmse_file = ct_base / "2026-06-30_11-24-06_ms_rflow_ct_float32_with_1_steps" / f"{case_key}.npy"

    slerp_vol = np.load(slerp_file)
    noslerp_vol = np.load(noslerp_file)
    mmse_vol = np.load(mmse_file)

    if slerp_vol.min() < 0:
        slerp_vol = slerp_vol * 0.5 + 0.5
    if noslerp_vol.min() < 0:
        noslerp_vol = noslerp_vol * 0.5 + 0.5
    if mmse_vol.min() < 0:
        mmse_vol = mmse_vol * 0.5 + 0.5

    metrics_str = "Slerp (10 steps): PSNR 19.81 dB, SSIM 0.6933  vs  No Slerp: PSNR 19.79 dB, SSIM 0.6829"
    generate_synthesis_gif(
        case_key=case_key,
        modality="ct",
        gt_vol=gt_ct,
        input_img=in_ct,
        mmse_vol=mmse_vol,
        noslerp_vol=noslerp_vol,
        slerp_vol=slerp_vol,
        output_path=output_path,
        steps=10,
        metrics_str=metrics_str,
        fps=8,
        duration=120,
        show_slice_line=False,
        target_size=target_size,
        panels=panels
    )


def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side animated GIFs for synthesis evaluation.")
    parser.add_argument("--modality", choices=["oct", "ct", "both"], default="both", help="Which modality GIF to generate.")
    parser.add_argument("--oct_out", type=str, default="assets/test_case_oct.gif")
    parser.add_argument("--ct_out", type=str, default="assets/test_case_ct.gif")
    parser.add_argument("--case_key", type=str, default=None, help="Specific case key to render.")
    parser.add_argument("--cases", nargs="+", default=None, help="List of specific case keys to render.")
    parser.add_argument("--top_n", type=int, default=None, help="Generate GIFs for top N cases ranked by metrics.")
    parser.add_argument("--rank_by", choices=["vessel", "psnr"], default="vessel", help="Metric to rank cases when using --top_n.")
    parser.add_argument("--output_dir", type=str, default="assets", help="Directory where case GIFs are stored when using --cases or --top_n.")
    parser.add_argument("--preset", choices=["full", "3panel", "both"], default="full", help="Panel preset.")
    parser.add_argument("--panels", nargs="+", choices=["input", "gt", "mmse", "noslerp", "slerp"], default=None, help="Custom panels list.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    panels = ["gt", "mmse", "slerp"] if args.preset == "3panel" else args.panels

    # Handle batch or specific case generation for OCT
    if args.case_key:
        generate_oct_case_gif(
            case_key=args.case_key,
            output_path=args.oct_out if args.oct_out != "assets/test_case_oct.gif" else str(out_dir / f"test_case_oct_{args.case_key[:8]}.gif"),
            panels=panels
        )
        return

    if args.cases:
        for idx, c_key in enumerate(args.cases, start=1):
            out_name = out_dir / f"test_case_oct_case{idx}.gif"
            generate_oct_case_gif(
                case_key=c_key,
                output_path=str(out_name),
                panels=panels
            )
        return

    if args.top_n:
        top_cases = get_top_oct_cases(args.top_n, rank_by=args.rank_by)
        print(f"Generating GIFs for top {len(top_cases)} OCT cases ranked by {args.rank_by}...")
        for idx, c_key in enumerate(top_cases, start=1):
            out_name = out_dir / f"test_case_oct_case{idx}.gif"
            generate_oct_case_gif(
                case_key=c_key,
                output_path=str(out_name),
                panels=panels
            )
        return

    # Standard presets
    if args.preset == "3panel":
        if args.modality in ["oct", "both"]:
            generate_best_oct_3panel_gif(args.oct_out if args.oct_out != "assets/test_case_oct.gif" else "assets/test_case_oct_gt_mmse_ours.gif")
    elif args.preset == "both":
        if args.modality in ["oct", "both"]:
            generate_best_oct_gif(args.oct_out, panels=args.panels)
            generate_best_oct_3panel_gif("assets/test_case_oct_gt_mmse_ours.gif")
        if args.modality in ["ct", "both"]:
            generate_best_ct_gif(args.ct_out, panels=args.panels)
    else:
        if args.modality in ["oct", "both"]:
            generate_best_oct_gif(args.oct_out, panels=args.panels)
        if args.modality in ["ct", "both"]:
            generate_best_ct_gif(args.ct_out, panels=args.panels)


if __name__ == "__main__":
    main()

