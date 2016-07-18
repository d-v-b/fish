""" Tools for estimating transformations between images
"""


def estimate_translation_itk(fixed, moving):
    """
    Use SimpleITK to estimate the translation between two images

    Returns a SimpleITK.TranslationTransform object

    Parameters
    ----------
    fixed : numpy array, 2D or 3D
        The reference image.

    moving : numpy array, 2D or 3D (dims must match fixed)
        The image to be transformed. The estimated transformation will take moving --> fixed.
    """
    import SimpleITK as sitk
    fixed_ = sitk.GetImageFromArray(fixed.astype('float32'))
    moving_ = sitk.GetImageFromArray(moving.astype('float32'))
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1,
                                               minStep=1e-5,
                                               numberOfIterations=100,
                                               gradientMagnitudeTolerance=1e-8)
    r.SetMetricSamplingStrategy(r.REGULAR)
    r.SetMetricSamplingPercentage(.25)
    tx = sitk.TranslationTransform(fixed_.GetDimension())
    r.SetInitialTransform(tx)
    r.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(False)
    r.SetShrinkFactorsPerLevel(shrinkFactors = [8, 4, 2])
    r.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1])
    r.SetInterpolator(sitk.sitkLinear)
    out_tx = r.Execute(fixed_, moving_)
    return out_tx


def apply_transform_itk(tx, moving, baseline=100):
    """
     Use SimpleITK to apply a transformation to an image, generating a new image.

    Parameters
    ----------
    tx : SimpleITK Transform object

    moving : numpy array, 2D or 3D
         The image to be transformed.

    baseline : float or int, optional
        The fill value added to edges of the transformed image
    """

    import SimpleITK as sitk
    moving_ = sitk.GetImageFromArray(moving.astype('float32'))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(moving_)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(baseline)
    resampler.SetTransform(tx)
    out = sitk.GetArrayFromImage(resampler.Execute(moving_))
    return out


def proj_reg(fixed, moving, p_fun):
    """
    Use SimpleITK to estimate transformation between two images by estimating transformations between the projections
    of the images

    Parameters
    ----------
    fixed : numpy array, 2D or 3D
        The reference image.

    moving : numpy array, 2D or 3D
        The image to be transformed.

    p_fun : projection function, must take an image and an integer as arguments
        A function that generates a lower-dimensional projection of an image, e.g. numpy.max()
    """
    import SimpleITK as sitk
    from numpy import zeros, where, fliplr, eye, array

    txs = []
    n_dim = len(fixed.shape)

    for dim in range(n_dim):
        txs.append(estimate_translation_itk(p_fun(fixed, dim), p_fun(moving, dim)))

    # generate a new transform that averages the transforms estimated for each axis
    params = array([t.GetParameters() for t in txs])
    inds = where(fliplr(1 - eye(n_dim)))
    full_mat = zeros([n_dim, n_dim])
    full_mat[inds] = params.ravel()
    # take a sum down the columns of full_mat, divide by n_dim - 1
    d_xyz = full_mat.sum(0) / (n_dim-1)

    final_tx = sitk.TranslationTransform(len(fixed.shape))
    final_tx.SetParameters(d_xyz)

    return final_tx


def proj_reg_batch(fixed, moving):
    """
    Convenience function for efficiently estimating a transformation between volumes in a parallel context.

    Uses the maximum projections of the input images to estimate transformations.

    Parameters
    ----------
    fixed : numpy array, 2D or 3D
        The reference image.

    moving : numpy array, 2D or 3D
    The image to be transformed.
    """
    from numpy import array, max    

    tx = proj_reg(fixed, moving, max)
    # Transpose to get array with shape [dimensions, time]
    dxdydz = array(tx.GetParameters()).T

    return dxdydz

def estimate_translation_batch(fixed, moving):
    """
    Convenience function for efficiently estimating a transformation between volumes in a parallel context.

    Parameters
    ----------
    fixed : numpy array, 2D or 3D
        The reference image.

    moving : numpy array, 2D or 3D
    The image to be transformed.
    """
    from numpy import array
    tx = estimate_translation_itk(fixed, moving)
    dxdydz = array(tx.GetParameters()).T

    return dxdydz


def ants_registration(fixed, moving, out_tform, tip='r', restrict=None):
    """
    Image registration between two image files using a command-line call to ANTS. Written by Mika Rubinov with
    modifications by Davis Bennett.

    Parameters
    ----------
    fixed : string
        Path to the reference image.

    moving : string
        Path to the image to be registered.

    out_tform : string
        path + prefix appended to the transform generated as output.

    tip : string
        Indicates which transformations to perform. Defaults to 'r', for rigid

    restrict : int
        Axis along which to restrict estimation of transformations
    """

    from os import system

    ants_reg_path = '/groups/ahrens/home/bennettd/ants/ants-2.1.0-redhat/antsRegistration'

    lin_tform_params = ' '.join([
        '--metric MI[{0}, {1}, 1, 32, Regular, 0.25]'.format(fixed, moving),
        '--convergence [1000x500x250x125]',
        '--shrink-factors 12x8x4x2',
        '--smoothing-sigmas 4x3x2x1vox',
    ])

    syn_tform_params = ' '.join([
        '--metric CC[{0}, {1}, 1, 4]'.format(fixed, moving),
        '--convergence [100x100x70x50x20]',
        '--shrink-factors 10x6x4x2x1',
        '--smoothing-sigmas 5x3x2x1x0vox',
    ])

    antsRegistration_call = ' '.join([
        ants_reg_path,
        '--initial-moving-transform [{0}, {1}, 1]'.format(fixed, moving),
        '--output [{0}]'.format(out_tform),
        '--dimensionality 3',
        '--float 1',
        '--interpolation Linear',
        '--winsorize-image-intensities [0.005,0.995]',
        '--use-histogram-matching 0',
        ('--restrict-deformation ' + restrict if restrict else ''),
        ('--transform Rigid[0.1] '            + lin_tform_params if 'r' in tip else ''),
        ('--transform Similarity[0.1] '       + lin_tform_params if 'i' in tip else ''),
        ('--transform Affine[0.1] '           + lin_tform_params if 'a' in tip else ''),
        ('--transform SyN[0.1,3,0] '          + syn_tform_params if 's' in tip else ''),
        ('--transform BSplineSyN[0.1,26,0,3]' + syn_tform_params if 'b' in tip else '')
    ])

    system(antsRegistration_call)
