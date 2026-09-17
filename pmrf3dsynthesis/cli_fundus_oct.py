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
import pydicom

from pmrf3dsynthesis.registration.registration import adapt_large_to_small
from pmrf3dsynthesis.vessel_segmentation.utils import load_model, extract_vessels_oct

logging.basicConfig(level=logging.INFO)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * b + 0.1140 * b
    return gray

def resample_rgb(image, reference, transform):
    components = [sitk.VectorIndexSelectionCast(image, i) for i in range(3)]
    resampled_components = [
        sitk.Resample(comp, reference, transform, sitk.sitkLinear, 0.0, comp.GetPixelID())
        for comp in components
    ]
    return sitk.Compose(resampled_components)

def load_fundus_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == '.dcm':
        dcm = pydicom.dcmread(str(path))
        img = dcm.pixel_array
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img
    return imageio.imread(str(path))

def load_oct_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() == '.dcm':
        dcm = pydicom.dcmread(str(path))
        oct_vol = dcm.pixel_array
    elif path.suffix.lower() == '.npy':
        oct_vol = np.load(str(path))
    else:
        oct_vol = np.load(str(path))
    
    if oct_vol.dtype == np.uint8:
        oct_vol = oct_vol.astype(np.float32) / 255.0
    elif oct_vol.dtype != np.float32 and oct_vol.dtype != np.float64:
        oct_vol = oct_vol.astype(np.float32)
        if oct_vol.max() > 1.0:
            oct_vol = oct_vol / oct_vol.max()
    return oct_vol

def build_oct_lookup(oct_path: Path):
    lookup = {}
    if not oct_path.is_dir():
        return lookup
    for oct_file in oct_path.rglob('*'):
        if oct_file.suffix.lower() in ['.dcm', '.npy']:
            name_lower = oct_file.name.lower()
            parts = name_lower.split('_')
            person_id = parts[0]
            lat = 'l' if '_l_' in name_lower or name_lower.endswith('_l.dcm') or name_lower.endswith('_l.npy') else ('r' if '_r_' in name_lower or name_lower.endswith('_r.dcm') or name_lower.endswith('_r.npy') else None)
            is_macula = 'mac' in name_lower
            
            key = (person_id, lat)
            if key not in lookup:
                lookup[key] = []
            lookup[key].append((is_macula, oct_file))
    return lookup

def find_oct_for_fundus(fundus_img_path: Path, oct_path: Path, oct_lookup: dict = None) -> Path:
    # 1. Direct name match (e.g. 01l.png -> 01l.npy or 01l.dcm)
    stem = fundus_img_path.name.split(".")[0]
    for ext in ['.npy', '.dcm', '.png']:
        candidate = oct_path / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # 2. Lookup by person_id and laterality
    if oct_lookup:
        name_lower = fundus_img_path.name.lower()
        parts = name_lower.split('_')
        person_id = parts[0]
        lat = 'l' if '_l_' in name_lower else ('r' if '_r_' in name_lower else None)
        
        candidates = oct_lookup.get((person_id, lat), [])
        if candidates:
            # Prioritize macula OCT
            macula_cands = [c[1] for c in candidates if c[0]]
            if macula_cands:
                return macula_cands[0]
            return candidates[0][1]
            
    # 3. Check within subject folder
    sub_dir = oct_path / fundus_img_path.parent.name
    if sub_dir.is_dir():
        for ext in ['.npy', '.dcm']:
            candidate = sub_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"No corresponding OCT file found for {fundus_img_path.name}")

def main(fundus_path: Path, oct_path: Path, res_path: Path, common_size: int = -1):
    res_path.mkdir(parents=True, exist_ok=True)
    if fundus_path.is_dir():
        # Find all image and DICOM files recursively or flat
        all_fundus = sorted([f for f in fundus_path.rglob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.dcm']])
        fundus_imgs = [
            f for f in all_fundus
            if not (res_path / f"{f.stem}.png").exists() and not (res_path / f.name).exists()
        ]
    else:
        fundus_imgs = [fundus_path]

    print(f'Found {len(fundus_imgs)} fundus files to process.')

    print("Building OCT lookup index...")
    oct_lookup = build_oct_lookup(oct_path) if oct_path.is_dir() else {}

    print("Loading models...")
    device = "cuda"
    checkpoint_dir = "default"
    fov_model, _ = fit.load_fovea_od_model(checkpoint_dir, device=device, return_test_dataloader=False)
    ensemble = fit.load_segmentation_ensemble(device=device)
    ves_model = load_model().to(device)
    
    logging.info(f"Processing {len(fundus_imgs)} fundus images...")
    processed_count = 0
    for fundus_img_path in tqdm(fundus_imgs):
        try:
            oct_img_path = find_oct_for_fundus(fundus_img_path, oct_path, oct_lookup)
            fundus_img = load_fundus_image(fundus_img_path)
            
            fovea_x, fovea_y, _, _ = fov_model.predict(fundus_img)
            fovea_x = int(np.round(fovea_x))
            fovea_y = int(np.round(fovea_y))

            oct_img = load_oct_volume(oct_img_path)

            moving_vessels = fit.ensemble_predict_segmentation(ensemble, [fundus_img], threshold=0.5, size=(512, 512)) 
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
            
            save_name = f"{fundus_img_path.stem}.png" if fundus_img_path.suffix.lower() == '.dcm' else fundus_img_path.name
            save_path = res_path / save_name

            imageio.imwrite(str(save_path), cropped_fundus.astype(np.uint8))
            processed_count += 1
        except FileNotFoundError as e:
            logging.warning(f"Skipping {fundus_img_path.name}: {e}")
            continue
        except Exception as e:
            logging.error(f"Error processing {fundus_img_path.name}: {e}")
            continue

    logging.info(f"Finished processing. Successfully saved {processed_count} images to {res_path}.")

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