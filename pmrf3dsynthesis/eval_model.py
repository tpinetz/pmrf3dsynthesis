import argparse
from pathlib import Path
import pandas as pd
import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm
from skimage.transform import resize
from pmrf3dsynthesis.data import constants as C
from pmrf3dsynthesis.evaluation.ssim import SSIM3D
from pmrf3dsynthesis.evaluation.retfound_fid import RETFoundFeatureExtractor
from pmrf3dsynthesis.models.flow_model import FlowModel
from pmrf3dsynthesis.models.mmse_model import MMSEModel
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

import pickle
import logging
import os
import nibabel as nib

def load_model(model_name, ckpt_path, device):
    if model_name == C.MMSE_MODEL:
        model = MMSEModel.load_from_checkpoint(ckpt_path, map_location=device)
    elif model_name == C.FLOW_MODEL:
        model = FlowModel.load_from_checkpoint(ckpt_path, map_location=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def load_dataset(df_path):
    keys = list(pd.read_csv(df_path).Keys)
    
    return keys

def slice_volume_ssim(y_hat, gt):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    scores = []
    for i in range(y_hat.shape[2]):
        scores.append(ssim(y_hat[:,:,i], gt[:,:,i]).item())
    return np.mean(scores)

@torch.no_grad()
def evaluate_model(model, T, keys, res_path, modality, noise_type, device):
    #TODO: make this not fixed paths
    if modality == 'oct':
        vol_path = Path(os.path.expandvars("$HOME/data/OCT/retinsight/FundusMIP/flattened_volumes/"))
        img_path = Path(os.path.expandvars("$HOME/data/OCT/retinsight/FundusMIP/Fundus_cropped_color/"))
    else:
        vol_path = Path(os.path.expandvars("$HOME/data/CT/LIDC-IDRI/CT/"))
        img_path = Path(os.path.expandvars("$HOME/data/CT/LIDC-IDRI/synth_xray/"))
        
    res_path.mkdir(exist_ok=True)

    metrics = {
        "PSNR": PeakSignalNoiseRatio(data_range=1.0).to(device),
        "SSIM": SSIM3D().to(device),
        "FID":  FrechetInceptionDistance(feature=RETFoundFeatureExtractor(), normalize=True).to(device) if modality == 'oct' else FrechetInceptionDistance(normalize=True).to(device)
    }
    output = {m: {} for m in metrics.keys()}
    
    for key in tqdm(keys, desc="Evaluating model"):
        # Load data
        if modality == 'oct':
            volume = resize(np.load(vol_path / f"{key}.npy")[:, 100:356, :], (128, 256, 256))  # Shape: (slices, H, W)
            input_img = resize(iio.imread(img_path / f"{key}.png"), (128, 256))  # Shape: (1, H, W)
        elif modality == 'ct':
            volume = nib.load(vol_path / f"{key}.nii.gz").get_fdata()  # Shape: (slices, H, W)
            volume =  (np.clip(volume.transpose((2,0,1)), -1350, 150) + 1350) / 1500
            input_img = iio.imread(img_path / f"{key}.png")[:,:,None] / 255.  # Shape: (1, H, W) 
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Perform evaluation
        x = torch.from_numpy(input_img.astype(np.float32).transpose((2,0,1))[None] * 2 - 1).to(device)

        if type(model) == FlowModel:
            y_hat = model(x, T, noise_type=noise_type)
        else:
            y_hat = model(x)
        
        if isinstance(y_hat, list):
            y_hat = y_hat[-1]

        gt = torch.from_numpy(volume[None, None]).float().to(device)
        output["PSNR"][key] = metrics["PSNR"](y_hat * 0.5 + 0.5, gt).item()
        output["SSIM"][key] = metrics["SSIM"](y_hat * 0.5 + 0.5, gt).item()

        # FID expects images in (C, H, W) format and values in [0, 1]
        if modality == 'oct':
            metrics["FID"].update((y_hat[0].transpose(1, 0) * 0.5 + 0.5), real=False)
            metrics["FID"].update(gt[0].transpose(1, 0), real=True)
        else:
            metrics["FID"].update(y_hat[0].transpose(1, 0).expand(-1, 3, -1, -1) * 0.5 + 0.5, real=False)
            metrics["FID"].update(gt[0].transpose(1, 0).expand(-1, 3, -1, -1), real=True)
        logging.info(f"Evaluated {key}: PSNR={output['PSNR'][key]}, SSIM={output['SSIM'][key]}")

        np.save(res_path / f"{key}.npy", y_hat.detach().cpu().numpy()[0,0])
        # break  # Remove this break to evaluate on all keys

    output["FID"] = [metrics["FID"].compute().item()]
    logging.info(f"Computed FID: {output['FID'][0]}")

    return output

def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--model_name', type=str, choices=[C.MMSE_MODEL, C.FLOW_MODEL], required=True, help='Name of the model to evaluate.')
    parser.add_argument('--df', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--res_path', type=str, required=True, help='Path to save evaluation results.')
    parser.add_argument('--mmse_path', type=str, default='', help='Path to the MMSE model file (if applicable).')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the evaluation on.')
    parser.add_argument('--T', type=int, default=20, help='Number of flow steps (if applicable).')
    parser.add_argument('--modality', type=str, choices=['oct', 'ct'], default='oct', help='Modality to evaluate on.')
    parser.add_argument('--noise_type', type=str, choices=[C.NOISE_TYPE_NORMAL, C.NOISE_TYPE_CONSTANT, C.NOISE_TYPE_SLERM], default=C.NOISE_TYPE_NORMAL, help='Type of noise to apply.')
    args = parser.parse_args()

    model = load_model(args.model_name, args.ckpt_path, args.device)

    if args.device != 'cpu':
        torch.set_float32_matmul_precision('high')

    if args.mmse_path:
        mmse_model = load_model(C.MMSE_MODEL, args.mmse_path, device=args.device)
        model.set_mmse_model(mmse_model)

    model = model.to(args.device)

    keys = list(pd.read_csv(args.df).Keys)

    eval_name = Path(args.ckpt_path).parent.parent.parent.name
    if args.model_name == C.FLOW_MODEL:
        eval_name += f"_with_{args.T}_steps"

    if args.noise_type != C.NOISE_TYPE_NORMAL:
        eval_name += f"_{args.noise_type}_noise"

    logging.info(f"Starting evaluation: {eval_name}")

    res_path = Path(args.res_path) / eval_name
    res_path.mkdir(parents=True, exist_ok=True)

    eval_results = evaluate_model(model.eval(), args.T, keys, res_path, args.modality, args.noise_type, device=args.device)
    with open(res_path / "eval_results.pkl", "wb") as f:
        pickle.dump(eval_results, f)
    print(f"Evaluation Results: {eval_results}")


if __name__ == '__main__':
    run_cli()
