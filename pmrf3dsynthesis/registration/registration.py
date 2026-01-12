from pmrf3dsynthesis.registration.utils import extract_vessels, convert_rigid_to_affine
import SimpleITK as sitk
import numpy as np

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
