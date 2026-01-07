from pmrf3dsynthesis.registration.utils import extract_vessels, convert_rigid_to_affine
import SimpleITK as sitk
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)


def register_three_stages(fixed: sitk.Image, moving: sitk.Image, sampling_percentage: float = 0.1) -> sitk.Image:
    """Register two images using a three-stage approach.
       Expects the two images to be grayscale.
    Args:
        fixed (sitk.Image): The fixed image (e.g., OCT MIP).
        moving (sitk.Image): The moving image (e.g., Fundus).

    Returns:
        sitk.Image: The registered moving image.
    """

    # ==========================================
    # STAGE 1: RAW IMAGES + RIGID (EULER)
    # ==========================================
    logging.info("\n--- STAGE 1: Rigid Registration (Raw Images) ---")
    
    R1 = sitk.ImageRegistrationMethod()
    
    # Initialize Geometry (Centers the images)
    initial_tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler2DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    R1.SetInitialTransform(initial_tx, inPlace=False)
    
    # Metric: Mutual Information (Different Modalities)
    R1.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R1.SetMetricSamplingStrategy(R1.RANDOM)
    R1.SetMetricSamplingPercentage(sampling_percentage)
    
    # Optimizer
    R1.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    R1.SetOptimizerScalesFromPhysicalShift()
    
    try:
        rigid_result = R1.Execute(fixed, moving)
        logging.info(f"Stage 1 Metric: {R1.GetMetricValue():.4f}")
    except Exception as e:
        if sampling_percentage < 0.5:
            logging.warning(f"Stage 1 Failed: {e} Retrying with larger sampling percentage.")
            return register_three_stages(fixed, moving, sampling_percentage=min(0.5, sampling_percentage * 2))
        else:
            logging.warning(f"Stage 1 Failed: {e}. Aborting registration.")
        raise e

    # ==========================================
    # STAGE 2: VESSEL MAPS + AFFINE
    # ==========================================
    logging.info("\n--- STAGE 2: Affine Registration (Vessel Maps) ---")
    
    # 1. Generate Vessel Maps
    logging.info("Extracting vessels...")
    fixed_vessels = extract_vessels(fixed)
    moving_vessels = extract_vessels(moving)
    
    # 2. Convert Rigid Result to Affine for Initialization
    # This "promotes" the transform so we can add shearing/scaling optimization
    initial_affine = convert_rigid_to_affine(rigid_result.GetNthTransform(0))
    
    R2 = sitk.ImageRegistrationMethod()
    R2.SetInitialTransform(initial_affine, inPlace=False)
    
    # Metric: Mean Squares (Same Modality: Vessel vs Vessel)
    R2.SetMetricAsMeanSquares()
    R2.SetMetricSamplingStrategy(R2.RANDOM)
    R2.SetMetricSamplingPercentage(0.10)
    
    # Optimizer: Affine is harder, so we use more iterations
    R2.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=200)
    R2.SetOptimizerScalesFromPhysicalShift()
    
    try:
        final_affine = R2.Execute(fixed_vessels, moving_vessels)
        logging.info(f"Stage 2 Metric: {R2.GetMetricValue():.4f}")
    except Exception as e:
        logging.warning(f"Stage 2 Failed: {e}")
        return

    logging.info("--- STAGE 3: Starting B-spline registration ---")
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

    logging.info(f"Initial Number of Parameters: {initial_transform.GetNumberOfParameters()}")
    
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
    final_transform = registration_method.Execute(fixed_vessels, out_vessels)

    logging.info(f"Final Metric Value: {registration_method.GetMetricValue()}")
    logging.info(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
    final_composite = sitk.CompositeTransform(final_affine)
    final_composite.AddTransform(final_transform)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_composite)
    
    # Return the final resampled moving image
    return resampler.Execute(moving)

