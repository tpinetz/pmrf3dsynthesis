from pmrf3dsynthesis.registration.utils import extract_vessels, convert_rigid_to_affine
import SimpleITK as sitk
import numpy as np
import cv2

import logging
logging.basicConfig(level=logging.INFO)


def initialize_fundus_to_oct(fixed_oct, moving_fundus, fundus_fovea_index):
    """
    fixed_oct: The OCT image (Target) - centered on fovea
    moving_fundus: The Fundus image (Source) - moving to match OCT
    fundus_fovea_index: The (x, y) pixel coordinates of the fovea on the Fundus
    """
    
    # 1. Identify the Target Point (Center of OCT)
    oct_size = fixed_oct.GetSize()
    oct_center_index = [sz / 2.0 for sz in oct_size]
    
    # Convert to physical space (mm)
    p_fixed_oct = fixed_oct.TransformContinuousIndexToPhysicalPoint(oct_center_index)
    
    # 2. Identify the Source Point (Fovea on Fundus)
    p_moving_fundus = moving_fundus.TransformContinuousIndexToPhysicalPoint(fundus_fovea_index)
    
    # 3. Create the Rigid Transform
    init_transform = sitk.Euler2DTransform()
    
    # CRITICAL: Set the center of rotation to the FUNDUS Fovea.
    # We want to pin the Fundus at this point.
    init_transform.SetCenter(p_moving_fundus)
    
    # 4. Calculate Translation (Target - Source)
    # This moves the Fundus fovea until it sits on top of the OCT center.
    translation_vector = [Pf - Pm for Pf, Pm in zip(p_fixed_oct, p_moving_fundus)]
    init_transform.SetTranslation(translation_vector)
    
    print(f"Anchoring Moving Fundus Fovea {p_moving_fundus} -> Fixed OCT Center {p_fixed_oct}")
    print(f"Computed Translation: {translation_vector}")
    
    return init_transform

def register_three_stages(fixed, moving, fovea_point):
    # ==========================================
    # STAGE 1: RAW IMAGES + RIGID (EULER)
    # ==========================================
    print("\n--- STAGE 1: Rigid Registration (Raw Images) ---")
    
    R1 = sitk.ImageRegistrationMethod()

    initial_tx = initialize_fundus_to_oct(fixed, moving, fovea_point).GetInverse()
    
    R1.SetInitialTransform(initial_tx, inPlace=False)
    
    # Metric: Mutual Information (Different Modalities)
    R1.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R1.SetMetricSamplingStrategy(R1.RANDOM)
    R1.SetMetricSamplingPercentage(0.05)
    
    # Optimizer
    R1.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=500)
    R1.SetOptimizerScalesFromPhysicalShift()
    
    try:
        rigid_result = R1.Execute(fixed, moving)
        print(f"Stage 1 Metric: {R1.GetMetricValue():.4f}")
    except Exception as e:
        print(f"Stage 1 Failed: {e}")
        return

    # ==========================================
    # STAGE 2: VESSEL MAPS + AFFINE
    # ==========================================
    print("\n--- STAGE 2: Affine Registration (Vessel Maps) ---")
    
    # 1. Generate Vessel Maps
    print("Extracting vessels...")
    fixed_vessels = extract_vessels(fixed, 5, 8)
    moving_vessels = extract_vessels(moving, 5, 8)
    
    # 2. Convert Rigid Result to Affine for Initialization
    # This "promotes" the transform so we can add shearing/scaling optimization
    initial_affine = convert_rigid_to_affine(rigid_result.GetNthTransform(0))
    
    R2 = sitk.ImageRegistrationMethod()
    R2.SetInitialTransform(initial_affine, inPlace=False)
    
    # Metric: Mean Squares (Same Modality: Vessel vs Vessel)
    R2.SetMetricAsMeanSquares()
    # R2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R2.SetMetricSamplingStrategy(R2.RANDOM)
    R2.SetMetricSamplingPercentage(0.10)
    
    # Optimizer: Affine is harder, so we use more iterations
    R2.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=1000)
    R2.SetOptimizerScalesFromPhysicalShift()
    
    try:
        final_affine = R2.Execute(fixed_vessels, moving_vessels)
        print(f"Stage 2 Metric: {R2.GetMetricValue():.4f}")
    except Exception as e:
        print(f"Stage 2 Failed: {e}")
        return

    # ==========================================
    # APPLY & SAVE
    # ==========================================
    print("\n--- Applying Final Transform ---")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_affine)

    out_vessels = resampler.Execute(moving_vessels)

    transform_domain_mesh_size = [16] * fixed_vessels.GetDimension()
    
    initial_transform = sitk.BSplineTransformInitializer(
        image1=fixed_vessels,
        transformDomainMeshSize=transform_domain_mesh_size,
        order=3  # Standard cubic B-spline
    )
    
    print(f"Initial Number of Parameters: {initial_transform.GetNumberOfParameters()}")
    
    # 2. Set up the Registration Method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Initialize with the B-spline transform
    registration_method.SetInitialTransform(initial_transform)
    
    # 3. Choose a Metric (Mean Squares or Mutual Information)
    # Mean Squares is good for same-modality; Mutual Information (Mattes) for multi-modality.
    registration_method.SetMetricAsMeanSquares()
    
    # 4. Choose an Optimizer
    # LBFGSB is often preferred for B-splines as it handles high-dimensional parameter spaces well.
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e+7
    )
    
    # 5. Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # 6. Execute Registration
    print("--- STAGE 3: Starting B-spline registration ---")
    final_transform = registration_method.Execute(fixed_vessels, out_vessels)
    
    print(f"Final Metric Value: {registration_method.GetMetricValue()}")
    print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")

    final_composite = sitk.CompositeTransform(final_affine)
    final_composite.AddTransform(final_transform)

    return final_composite



def adapt_large_to_small(img_small, img_large, 
                         angle_range=(-10, 10), 
                         scale_range=(0.9, 1.1), 
                         steps=15):
    """
    Finds the small mask inside the large one, then Warps/Crops the 
    LARGE image to match the geometry of the SMALL image.
    """
    
    # --- PHASE 1: GRID SEARCH (Find where small fits in large) ---
    # We create a list of scales and angles to test
    scales = np.linspace(scale_range[0], scale_range[1], steps)
    angles = np.linspace(angle_range[0], angle_range[1], steps)
    
    best_score = -1
    best_params = None # Will store (scale, angle, loc_x, loc_y)
    
    # Blur large image once for smoother matching
    large_smooth = cv2.GaussianBlur(img_large, (5,5), 2)
    
    h_small, w_small = img_small.shape[:2]
    
    print(f"Searching {len(scales)*len(angles)} combinations...")

    for scale in scales:
        # Resize template
        curr_w = int(w_small * scale)
        curr_h = int(h_small * scale)
        resized = cv2.resize(img_small, (curr_w, curr_h), interpolation=cv2.INTER_LINEAR)
        
        # Center for rotation
        center = (curr_w // 2, curr_h // 2)
        
        for angle in angles:
            # Rotate template
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(resized, M_rot, (curr_w, curr_h))
            
            # Blur template
            rotated_smooth = cv2.GaussianBlur(rotated, (5,5), 2)

            # Match
            res = cv2.matchTemplate(large_smooth, rotated_smooth, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_score:
                best_score = max_val
                # max_loc is the top-left corner in the large image
                best_params = (scale, angle, max_loc)

    scale, angle, (loc_x, loc_y) = best_params
    print(f"Match: Scale={scale:.3f}, Angle={angle:.2f}°, Loc={loc_x},{loc_y}")

    # --- PHASE 2: INVERSE WARP (Large -> Small) ---
    
    # We need to map pixels from the Large image BACK to the Small image.
    # The robust way is to define 3 points on the 'Small' image and calculate 
    # where they land on the 'Large' image.
    
    # 1. Define 3 points on the Original Small Image (Triangle)
    #    (Top-Left, Top-Right, Bottom-Left)
    pts_small = np.float32([ [0,0], [w_small, 0], [0, h_small] ])
    
    # 2. Calculate where these points are in the Rotated/Scaled state
    #    (This mimics the transformations we did in the loop)
    
    # A. Scale
    pts_scaled = pts_small * scale
    
    # B. Rotate
    # Rotation was around the center of the SCALED image
    scaled_center = (w_small * scale / 2, h_small * scale / 2)
    M_rot = cv2.getRotationMatrix2D(scaled_center, angle, 1.0)
    
    # OpenCV's transform expects an array of shape (N, 1, 2)
    pts_scaled_reshaped = np.array([pts_scaled], dtype=np.float32).transpose(1,0,2)
    pts_rotated = cv2.transform(pts_scaled_reshaped, M_rot)[:,0,:]
    
    # C. Translate
    # The 'loc_x, loc_y' from matchTemplate is the top-left of the rotated image canvas.
    # We simply add this offset to our points.
    pts_on_large = pts_rotated + [loc_x, loc_y]

    pts_on_large = pts_on_large.astype(np.float32)
    pts_small = pts_small.astype(np.float32)
    
    # 3. Compute the Affine Matrix: Large -> Small
    # We want a matrix that takes 'pts_on_large' and maps them to 'pts_small'
    M_final = cv2.getAffineTransform(pts_on_large, pts_small)
    
    # 4. Warp the Large Image
    # Result size is exactly (w_small, h_small)
    matched_fundus = cv2.warpAffine(img_large, M_final, (w_small, h_small))
    
    return matched_fundus, M_final