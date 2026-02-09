import torch
import argparse
from pathlib import Path
import imageio
import numpy as np
from tqdm import tqdm
from skimage.transform import resize


def to_contrast(img):
    return ((img * 0.5 + 0.5) * 255).astype(np.uint8)

def spread_contrast(img):
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

def main(res_path: Path, fl_oct_path: Path, model: torch.nn.Module = None):
    res_mmse = res_path / "mmse"
    res_mmse.mkdir(exist_ok=True)
    res_oct = res_path / "oct"
    res_oct.mkdir(exist_ok=True)

    for pat in tqdm(fl_oct_path.iterdir()):
        test_oct = np.load(pat)
        orig_oct = resize(test_oct[:, 100:356, :], (64, 256, 256))
        mip_img = orig_oct.mean(axis=1)
        x = torch.from_numpy(np.repeat(mip_img[:,None,:], 256, axis=1)[None, None] * 2 - 1)
        with torch.no_grad():
            y_hat = model(x.to('cuda')).detach().cpu().numpy()
            if isinstance(y_hat, list):
                y_hat = y_hat[-1]

        for fold, scan in zip([res_mmse, res_oct], [y_hat[0,0], orig_oct * 2 - 1]):
            for i, bscan in enumerate(scan):
                imageio.imsave(fold / (pat.name.split(".")[0] + f"_{i:03d}.png"), to_contrast(bscan))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()()
    parser.add_argument('res_path', type=Path, help='Path to save results')
    parser.add_argument('fl_oct_path', type=Path, help='Path to save results')
    parser.add_argument('--model_path', type=Path, required=True, help='Path to the trained model')
    args = parser.parse_args()
    main(Path(args.res_path), Path(args.fl_oct_path))
