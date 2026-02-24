from pathlib import Path
import argparse
import SimpleITK as sitk
import cv2
import imageio
import numpy as np
from sklearn import ensemble
from tqdm import tqdm
import logging
from skimage.transform import resize
import fundus_image_toolbox as fit

from pmrf3dsynthesis.registration.registration import adapt_large_to_small
from pmrf3dsynthesis.vessel_segmentation.utils import load_model, extract_vessels_oct

logging.basicConfig(level=logging.INFO)

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def resample_rgb(image, reference, transform):
    # 1. Split RGB into 3 separate scalar images
    components = [sitk.VectorIndexSelectionCast(image, i) for i in range(3)]
    
    # 2. Resample each channel individually using standard settings
    resampled_components = [
        sitk.Resample(comp, reference, transform, sitk.sitkLinear, 0.0, comp.GetPixelID())
        for comp in components
    ]
    
    # 3. Merge them back into one RGB image
    return sitk.Compose(resampled_components)


def main(fundus_path: Path, oct_path: Path, res_path: Path, common_size: int = -1):
    res_path.mkdir(exist_ok=True)
    if fundus_path.is_dir():
        fundus_imgs = [file for file in list(fundus_path.iterdir()) if not (res_path / file.name).exists()]
    else:
        fundus_imgs = [fundus_path]

    print(f'Found {len(fundus_imgs)} files: ', fundus_imgs)

    print("Loading model")
    device = "cuda"
    checkpoint_dir = "default"
    fov_model, _ = fit.load_fovea_od_model(checkpoint_dir, device=device, return_test_dataloader=False)
    ensemble = fit.load_segmentation_ensemble(device=device)
    ves_model = load_model().to(device)
    
    logging.info(f"Found {len(fundus_imgs)} fundus images to process.")
    for fundus_img_path in tqdm(fundus_imgs):
        try:
            oct_img_path = oct_path / (fundus_img_path.name.split(".")[0] + ".npy")
            fundus_img = imageio.imread(fundus_img_path)
            fovea_x, fovea_y, _, _ = fov_model.predict(str(fundus_img_path))
            fovea_x = int(np.round(fovea_x))
            fovea_y = int(np.round(fovea_y))


            oct_img = np.load(oct_img_path)


            moving_vessels = fit.ensemble_predict_segmentation(ensemble, [str(fundus_img_path)], threshold=0.5, size=(512, 512)) 
            fixed_vessels = extract_vessels_oct(oct_img, ves_model, device=device)


            if common_size < 0:
                new_size = max(fixed_vessels.shape)  # default size if not provided
            else:
                new_size = common_size

            fixed_vessels_uint = (resize(fixed_vessels, (new_size, new_size)) * 255).astype(np.uint8)
            moving_vessels_uint = (moving_vessels * 255).astype(np.uint8)

            rad = 700
            _, M = adapt_large_to_small(fixed_vessels_uint,
                                        moving_vessels_uint[max(fovea_y-rad, 0):fovea_y+rad, max(0, fovea_x-rad):fovea_x+rad])
            cropped_fundus = cv2.warpAffine(fundus_img[max(fovea_y-rad, 0):fovea_y+rad, max(0, fovea_x-rad):fovea_x+rad], M, (new_size, new_size))
            save_path = res_path / fundus_img_path.name

            imageio.imwrite(save_path, cropped_fundus.astype(np.uint8))
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