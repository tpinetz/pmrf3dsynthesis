import argparse
import os
import sys
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from skimage.transform import resize
from tqdm import tqdm

# Add repo roots to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
EYE_SCREEN_ROOT = os.environ.get("EYE_SCREEN_ROOT", "/home/thomas-pinetz/repos/eye-screen")
sys.path.insert(0, EYE_SCREEN_ROOT)

from eyescreen.models.stable_edm_model import Stable_EDM_model
from causalvideovae.model import ModelRegistry
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pmrf3dsynthesis.eval_model import (
    load_sample,
    compute_sdif,
    compute_volume_lpips,
    load_vessel_model,
    extract_vessels_oct,
    compute_vessel_metrics,
    setup_fid_metric,
    save_summary_table
)

def edm_sampler_fn(net, latents, adjacent, num_steps=20, sigma_min=0.002, sigma_max=80.0, rho=7):
    """2nd order Heun EDM sampler matching the training configuration."""
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    adj = net.init_kernel(adjacent)[:, None].transpose(2, 3)
    
    # Pad adj to match latent temporal dimension if needed
    if adj.shape[2] != latents.shape[2]:
        pad_t = latents.shape[2] - adj.shape[2]
        adj = F.pad(adj, (0, 0, 0, 0, 0, pad_t), mode='replicate')

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        # Euler step
        inp = torch.concat([x_cur.float(), adj], dim=1)
        denoised = net(inp, t_cur).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        
        # 2nd order correction
        if i < num_steps - 1:
            inp2 = torch.concat([x_next.float(), adj], dim=1)
            denoised2 = net(inp2, t_next).to(torch.float64)
            d_prime = (x_next - denoised2) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

RUN_NAME = "2026-09-04_11-48-25_ms__stable_edm_fundus_oct_pretrained_vae"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the Stable EDM (WF-VAE latent) baseline on the OCT test set.")
    p.add_argument("--num-steps", type=int, default=20, help="Heun sampler steps (paper: 20 and 100).")
    p.add_argument("--ckpt", default=f"/home/thomas-pinetz/data/results/iscreen/gen/{RUN_NAME}/version_0/checkpoints/last.ckpt")
    p.add_argument("--stats", default="/home/thomas-pinetz/data/OCT/retinsight/FundusMIP/volumes_wf_vae/dataset_mean_std.pkl")
    p.add_argument("--fundus-dir", default="/home/thomas-pinetz/data/OCT/retinsight/FundusMIP/Fundus_cropped_color")
    p.add_argument("--test-csv", default=os.path.join(EYE_SCREEN_ROOT, "data_splits/fund_mip_251211/test_keys.csv"))
    p.add_argument("--seed", type=int, default=None,
                   help="Seed case i with seed + i. The 100-step paper run used --seed 42; the 20-step run was unseeded.")
    p.add_argument("--out-dir", default=None,
                   help="Defaults to .../OCT/results/<run>, with _with_<N>_steps appended when --num-steps != 20.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Paths
    ckpt_path = args.ckpt
    stats_path = args.stats
    fundus_dir = Path(args.fundus_dir)
    if args.out_dir is None:
        suffix = "" if args.num_steps == 20 else f"_with_{args.num_steps}_steps"
        out_dir = Path(f"/media/thomas-pinetz/data/data/OCT/results/{RUN_NAME}{suffix}")
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Stable EDM model
    print(f"Loading checkpoint from: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = Stable_EDM_model(**ckpt["hyper_parameters"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device).eval()
    print("Stable_EDM_model loaded successfully.", flush=True)

    # 2. Load WF-VAE
    print("Loading WF-VAE ('chestnutlzj/WF-VAE-L-16Chn')...", flush=True)
    vae = ModelRegistry.get_model("WFVAE").from_pretrained("chestnutlzj/WF-VAE-L-16Chn")
    vae = vae.to(device, dtype=torch.bfloat16).eval()
    vae.enable_tiling()
    print("WF-VAE loaded successfully.", flush=True)

    # 3. Load stats
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    mean_th = torch.from_numpy(stats["mean"]).to(device=device, dtype=torch.float32)[None, :, None, None, None]
    std_th = torch.from_numpy(stats["std"]).to(device=device, dtype=torch.float32)[None, :, None, None, None]

    # 4. Get 81 test keys
    test_csv = args.test_csv
    keys = pd.read_csv(test_csv)["Keys"].tolist()
    print(f"Total test cases: {len(keys)}", flush=True)

    # 5. Initialize metric calculators
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    lpips_fn.eval()
    ves_model = load_vessel_model().to(device).eval()
    fid_metric = setup_fid_metric("oct", "auto", None, device)

    eval_results = {
        "PSNR": {},
        "SSIM": {},
        "LPIPS": {},
        "SDIF": {},
        "Vessel_Dice": {},
        "Vessel_IoU": {},
        "Vessel_Soft_Dice": {},
        "FID": []
    }

    # 6. Evaluation loop
    pbar = tqdm(keys, desc="Evaluating Stable EDM OCT")
    for idx, key in enumerate(pbar):
        # Load Ground Truth (shape (128, 256, 256) in [0, 1])
        in_img, gt_vol = load_sample(key, "oct")
        gt_01 = np.clip(gt_vol.astype(np.float32), 0.0, 1.0)
        
        # Load Fundus input
        fundus_file = fundus_dir / f"{key}.png"
        if not fundus_file.exists():
            fundus_file = fundus_file.with_suffix(".jpg")
        fundus_img = imageio.imread(fundus_file)
        if fundus_img.dtype == np.uint8:
            fundus_img = fundus_img / 255.0
        fundus_img = resize(fundus_img, (128, 256))
        fundus_t = torch.from_numpy(fundus_img.transpose(2, 0, 1)[None].astype(np.float32)).to(device)

        pred_file = out_dir / f"{key}.npy"
        if pred_file.exists():
            pred_01 = np.load(pred_file).astype(np.float32)
            if pred_01.shape != (128, 256, 256):
                if pred_01.shape[1] > 256:
                    pred_01 = pred_01[:128, 100:356, :]
                pred_01 = resize(pred_01, (128, 256, 256)).astype(np.float32)
                np.save(pred_file, pred_01)
        else:
            with torch.no_grad():
                if args.seed is not None:
                    torch.manual_seed(args.seed + idx)
                init_latents = torch.randn((1, 16, 33, 64, 64), device=device)
                sample = edm_sampler_fn(model, init_latents, fundus_t, num_steps=args.num_steps)
                unnorm = sample.float() * (2.0 * std_th) + mean_th
                dec = vae.decode(unnorm.to(dtype=torch.bfloat16)).sample
                pred_vol = dec[0, 0, :128].float().cpu().numpy()
                
                # Rescale VAE output [-1, 1] to [0, 1]
                if pred_vol.min() < -0.1:
                    pred_01 = np.clip(pred_vol * 0.5 + 0.5, 0.0, 1.0)
                else:
                    pred_01 = np.clip(pred_vol, 0.0, 1.0)

                # Crop and resize from (128, 512, 512) to (128, 256, 256)
                if pred_01.shape[1] > 256:
                    pred_01 = pred_01[:, 100:356, :]
                pred_01 = resize(pred_01, (128, 256, 256)).astype(np.float32)
                np.save(pred_file, pred_01)

        p_t = torch.from_numpy(pred_01[None, None]).to(device)
        g_t = torch.from_numpy(gt_01[None, None]).to(device)

        # Compute metrics
        eval_results["PSNR"][key] = psnr_fn(p_t, g_t).item()
        eval_results["SSIM"][key] = ssim_fn(p_t, g_t).item()
        eval_results["LPIPS"][key] = compute_volume_lpips(pred_01, gt_01, lpips_fn, device=device)
        eval_results["SDIF"][key] = compute_sdif(pred_01, gt_01, axis=0)["SDIF"]

        # Vessel segmentation metrics
        pred_vessels = extract_vessels_oct(pred_01, ves_model, device=device)
        gt_vessels = extract_vessels_oct(gt_01, ves_model, device=device)
        v_metrics = compute_vessel_metrics(pred_vessels, gt_vessels)
        eval_results["Vessel_Dice"][key] = v_metrics["Vessel_Dice"]
        eval_results["Vessel_IoU"][key] = v_metrics["Vessel_IoU"]
        eval_results["Vessel_Soft_Dice"][key] = v_metrics["Vessel_Soft_Dice"]

        # FID
        p_sl = p_t[0].transpose(1, 0)
        g_sl = g_t[0].transpose(1, 0)
        fid_metric.update(p_sl, real=False)
        fid_metric.update(g_sl, real=True)

        pbar.set_postfix({
            "PSNR": f"{np.mean(list(eval_results['PSNR'].values())):.2f}dB",
            "SSIM": f"{np.mean(list(eval_results['SSIM'].values())):.4f}",
            "LPIPS": f"{np.mean(list(eval_results['LPIPS'].values())):.4f}",
            "VDice": f"{np.mean(list(eval_results['Vessel_Dice'].values())):.4f}"
        })

    eval_results["FID"] = [fid_metric.compute().item()]

    # Save pickle and CSV
    with open(out_dir / "eval_results.pkl", "wb") as f:
        pickle.dump(eval_results, f)

    save_summary_table(eval_results, out_dir)
    print(f"\nSaved eval_results.pkl and eval_results.csv to {out_dir}")

    # Summary
    psnrs = list(eval_results["PSNR"].values())
    ssims = list(eval_results["SSIM"].values())
    lpips_vals = list(eval_results["LPIPS"].values())
    sdif_vals = list(eval_results["SDIF"].values())
    vdice_vals = list(eval_results["Vessel_Dice"].values())

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS: {RUN_NAME}, {args.num_steps} steps")
    print("=" * 70)
    print(f"  PSNR:        {np.mean(psnrs):.2f} ± {np.std(psnrs, ddof=1):.2f} dB")
    print(f"  SSIM:        {np.mean(ssims):.4f} ± {np.std(ssims, ddof=1):.4f}")
    print(f"  LPIPS:       {np.mean(lpips_vals):.4f} ± {np.std(lpips_vals, ddof=1):.4f}")
    print(f"  SDIF:        {np.mean(sdif_vals):.6f} ± {np.std(sdif_vals, ddof=1):.6f}")
    print(f"  Vessel Dice: {np.mean(vdice_vals):.4f} ± {np.std(vdice_vals, ddof=1):.4f}")
    print(f"  FID:         {eval_results['FID'][0]:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
