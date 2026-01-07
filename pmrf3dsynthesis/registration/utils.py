import SimpleITK as sitk
import numpy as np
from skimage.filters import frangi
from skimage import exposure

def extract_vessels(sitk_image):
    """
    Takes a SimpleITK image, converts to numpy, enhances contrast,
    and applies Frangi vesselness filter. Returns a SimpleITK image.
    """
    # 1. Convert to Numpy
    img_arr = sitk.GetArrayFromImage(sitk_image) * 1.0
    
    # 2. Normalize to 0-1 range for processing
    img_arr = 1. - (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())

    # 3. Adaptive Histogram Equalization (CLAHE) to boost local contrast
    # This helps faint vessels in the OCT appear clearer
    img_arr = exposure.equalize_adapthist(img_arr, clip_limit=0.03)

    # 4. Frangi Vesselness Filter
    # sigmas: range of vessel scales (1 to 6 pixels is typical for retina)
    vessel_map = frangi(img_arr, sigmas=range(5, 10), black_ridges=False)
    
    # 5. Convert back to SimpleITK
    vessel_image = sitk.GetImageFromArray(vessel_map)
    vessel_image.CopyInformation(sitk_image) # Keep original spacing/origin
    
    return vessel_image

def convert_rigid_to_affine(rigid_tx):
    """
    Helper to promote a Euler2DTransform (Rigid) to an AffineTransform.
    We copy the Matrix, Translation, and Center.
    """
    affine = sitk.AffineTransform(2)
    affine.SetMatrix(rigid_tx.GetMatrix())
    affine.SetTranslation(rigid_tx.GetTranslation())
    affine.SetCenter(rigid_tx.GetCenter())
    return affine