from pathlib import Path
import argparse
import SimpleITK as sitk
import imageio
import numpy as np
from tqdm import tqdm
import logging
from skimage.transform import resize

from pmrf3dsynthesis.registration.registration import register_three_stages

logging.basicConfig(level=logging.INFO)

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def process_img(fundus_img: np.ndarray, oct_img:np.ndarray, common_size: int = -1) -> np.ndarray:

    if common_size > 0:
        fixed_image = resize(oct_img.mean(axis=1), (common_size, common_size))  # in case the resolution is known
    else:
        fixed_image = resize(oct_img.mean(axis=1), (oct_img.shape[2], oct_img.shape[2]))    # in case the fundus image resolution is not known

    fixed = sitk.GetImageFromArray((fixed_image - fixed_image.min()) / (fixed_image.max() - fixed_image.min()) * 255.)
    moving = sitk.GetImageFromArray((fundus_img - fundus_img.min()) / (fundus_img.max() - fundus_img.min()) * 255.)
    out = register_three_stages(fixed, moving)

    return sitk.GetArrayFromImage(out)



def main(fundus_path: Path, oct_path: Path, res_path: Path, common_size: int = -1):
    res_path.mkdir(exist_ok=True)
    if fundus_path.is_dir():
        fundus_imgs = list(fundus_path.iterdir())
    else:
        fundus_imgs = [fundus_path]
    
    logging.info(f"Found {len(fundus_imgs)} fundus images to process.")
    for fundus_img_path in tqdm(fundus_imgs):
        try:
            oct_img_path = oct_path / (fundus_img_path.name.split(".")[0] + ".npy")
            fundus_img = rgb2gray(imageio.imread(fundus_img_path))
            oct_img = np.load(oct_img_path)
            registered_mip = process_img(fundus_img, oct_img, common_size=common_size)
            save_path = res_path / fundus_img_path.name
            imageio.imwrite(save_path, registered_mip.astype(np.uint8))
        except FileNotFoundError as e:
            logging.warning(f"Skipping {fundus_img_path.name}: {e}")
            continue
        except Exception as e:
            logging.error(f"Error processing {fundus_img_path.name}: {e}")
            continue


def run_cli():
    """Wrapper function for the console_script entry point"""
    parser = argparse.ArgumentParser(description="Run 3D Synthesis Registration")
    parser.add_argument('fundus_path', type=Path, help='Path to fundus images')
    parser.add_argument('oct_path', type=Path, help='Path to OCT images')
    parser.add_argument('res_path', type=Path, help='Path to save results')
    parser.add_argument('--common_size', type=int, default=-1, help='Size of the fundus image in oct pixels (if known)')
    
    args = parser.parse_args()
    main(Path(args.fundus_path), Path(args.oct_path), Path(args.res_path), common_size=args.common_size)


if __name__ == '__main__':
    run_cli()