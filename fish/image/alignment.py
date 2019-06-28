#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Tools for estimating transformations between images
#
# Davis Bennett
# davis.v.bennett@gmail.com
#
# License: MIT
#


def estimate_translation(
    fixed,
    moving,
    metric_sampling=1.0,
    factors=(4, 2, 1),
    level_iters=(1000, 1000, 1000),
    sigmas=(8, 4, 1),
):
    """
    Estimate translation between 2D or 3D images using dipy.align.

    Parameters
    ----------
    fixed : numpy array, 2D or 3D
        The reference image.

    moving : numpy array, 2D or 3D
        The image to be transformed.

    metric_sampling : float, within the interval (0,  1]
        Fraction of the metric sampling to use for optimization

    factors : iterable
        The image pyramid factors to use

    level_iters : iterable
        Number of iterations per pyramid level

    sigmas : iterable
        Standard deviation of gaussian blurring for each pyramid level

    """
    from dipy.align.transforms import TranslationTransform2D, TranslationTransform3D
    from dipy.align.imaffine import AffineRegistration, MutualInformationMetric

    metric = MutualInformationMetric(32, metric_sampling)
    affreg = AffineRegistration(
        metric=metric,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors,
        verbosity=0,
    )

    if fixed.ndim == 2:
        transform = TranslationTransform2D()
    elif fixed.ndim == 3:
        transform = TranslationTransform3D()

    tx = affreg.optimize(fixed, moving, transform, params0=None)

    return tx


class SYNreg(object):

    """
    Wrap full linear + nonlinear(syn) registration, with the assumption that affine  params and warp field are
    estimated from spatially downsampled data.
    """

    def __init__(
        self,
        level_iters_lin,
        sigmas,
        factors,
        level_iters_syn,
        metric_lin=None,
        metric_syn=None,
        ss_sigma_factor=1.0,
        verbosity=0,
    ):
        from dipy.align.metrics import CCMetric
        from dipy.align.imaffine import MutualInformationMetric

        self.level_iters_lin = level_iters_lin
        self.sigmas = sigmas
        self.factors = factors
        self.level_iters_syn = level_iters_syn
        self.ss_sigma_factor = ss_sigma_factor
        self.verbosity = verbosity
        if metric_lin is None:
            nbins = 32
            self.metric_lin = MutualInformationMetric(nbins, None)
        if metric_syn is None:
            self.metric_syn = CCMetric(3)

        self.affreg = None
        self.sdreg = None
        self.translation_tx = None
        self.rigid_tx = None
        self.affine_tx = None
        self.sdr_tx = None

    def generate_warp_field(self, static, moving, static_axis_units, moving_axis_units):
        from numpy import eye
        from dipy.align.imaffine import AffineRegistration
        from dipy.align.transforms import (
            TranslationTransform3D,
            RigidTransform3D,
            AffineTransform3D,
        )
        from dipy.align.imwarp import SymmetricDiffeomorphicRegistration as SDR

        static_g2w = eye(1 + static.ndim)
        moving_g2w = static_g2w.copy()
        params0 = None

        static_g2w[range(static.ndim), range(static.ndim)] = static_axis_units
        moving_g2w[range(moving.ndim), range(moving.ndim)] = moving_axis_units

        self.affreg = AffineRegistration(
            metric=self.metric_lin,
            level_iters=self.level_iters_lin,
            sigmas=self.sigmas,
            factors=self.factors,
            verbosity=self.verbosity,
            ss_sigma_factor=self.ss_sigma_factor,
        )

        self.sdreg = SDR(
            metric=self.metric_syn,
            level_iters=self.level_iters_syn,
            ss_sigma_factor=self.ss_sigma_factor,
        )

        self.translation_tx = self.affreg.optimize(
            static,
            moving,
            TranslationTransform3D(),
            params0,
            static_g2w,
            moving_g2w,
            starting_affine="mass",
        )

        self.rigid_tx = self.affreg.optimize(
            static,
            moving,
            RigidTransform3D(),
            params0,
            static_g2w,
            moving_g2w,
            starting_affine=self.translation_tx.affine,
        )

        self.affine_tx = self.affreg.optimize(
            static,
            moving,
            AffineTransform3D(),
            params0,
            static_g2w,
            moving_g2w,
            starting_affine=self.rigid_tx.affine,
        )

        self.sdr_tx = self.sdreg.optimize(
            static, moving, static_g2w, moving_g2w, self.affine_tx.affine
        )

    def apply_transform(self, moving, moving_axis_units, desired_transform):
        from numpy import eye
        from numpy.linalg import inv

        moving_g2w = eye(1 + moving.ndim)
        moving_g2w[range(moving.ndim), range(moving.ndim)] = moving_axis_units

        txs = dict(affine=self.affine_tx, sdr=self.sdr_tx)
        tx = txs[desired_transform]

        if desired_transform == "sdr":
            result = tx.transform(
                moving,
                image_world2grid=inv(moving_g2w),
                out_shape=moving.shape,
                out_grid2world=moving_g2w,
            )
        else:
            result = tx.transform(
                moving,
                image_grid2world=moving_g2w,
                sampling_grid_shape=moving.shape,
                sampling_grid2world=moving_g2w,
            )

        return result
