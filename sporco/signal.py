# -*- coding: utf-8 -*-
# Copyright (C) 2015-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Signal and image processing functions."""

from __future__ import absolute_import, division, print_function
from builtins import range

import numpy as np

from sporco.fft import is_complex_dtype, fftn, ifftn, rfftn, irfftn, fftconv
from sporco import array


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def complex_randn(*args):
    """Return a complex array of samples drawn from a standard normal
    distribution.

    Parameters
    ----------
    d0, d1, ..., dn : int
      Dimensions of the random array

    Returns
    -------
    a : ndarray
      Random array of shape (d0, d1, ..., dn)
    """

    return np.random.randn(*args) + 1j*np.random.randn(*args)



def spnoise(s, frc, smn=0.0, smx=1.0):
    """Return image with salt & pepper noise imposed on it.

    Parameters
    ----------
    s : ndarray
      Input image
    frc : float
      Desired fraction of pixels corrupted by noise
    smn : float, optional (default 0.0)
      Lower value for noise (pepper)
    smx : float, optional (default 1.0)
      Upper value for noise (salt)

    Returns
    -------
    sn : ndarray
      Noisy image
    """

    sn = s.copy()
    spm = np.random.uniform(-1.0, 1.0, s.shape)
    sn[spm < frc - 1.0] = smn
    sn[spm > 1.0 - frc] = smx
    return sn



def rndmask(shp, frc, dtype=None):
    r"""Return random mask image with values in :math:`\{0,1\}`.

    Parameters
    ----------
    s : tuple
      Mask array shape
    frc : float
      Desired fraction of zero pixels
    dtype : data-type or None, optional (default None)
      Data type of mask array

    Returns
    -------
    msk : ndarray
      Mask image
    """

    msk = np.asarray(np.random.uniform(-1.0, 1.0, shp), dtype=dtype)
    msk[np.abs(msk) > frc] = 1.0
    msk[np.abs(msk) < frc] = 0.0
    return msk



def rgb2gray(rgb):
    """Convert an RGB image (or images) to grayscale.

    Parameters
    ----------
    rgb : ndarray
      RGB image as Nr x Nc x 3 or Nr x Nc x 3 x K array

    Returns
    -------
    gry : ndarray
      Grayscale image as Nr x Nc or Nr x Nc x K array
    """

    w = np.array([0.299, 0.587, 0.114], dtype=rgb.dtype)[np.newaxis,
                                                         np.newaxis]
    return np.sum(w * rgb, axis=2)



def grad(x, axis, zero_pad=False):
    r"""Compute gradient of `x` along axis `axis`.

    The form of the gradient operator depends on parameter `zero_pad`.
    If it is False, the operator is of the form

    .. math::

      \left(\begin{array}{rrrrr}
      -1 & 1 & 0 & \ldots & 0\\
      0 & -1 & 1 & \ldots & 0\\
      \vdots & \vdots & \ddots & \ddots & \vdots\\
      0 & 0 & \ldots & -1 & 1\\
      0 & 0 & \dots & 0 & 0
      \end{array}\right) \;,

    mapping :math:`\mathbb{R}^N \rightarrow \mathbb{R}^N`. If parameter
    `zero_pad` is True, the operator is of the form

    .. math::

      \left(\begin{array}{rrrrr}
      1 & 0 & 0 & \ldots & 0\\
      -1 & 1 & 0 & \ldots & 0\\
      0 & -1 & 1 & \ldots & 0\\
      \vdots & \vdots & \ddots & \ddots & \vdots\\
      0 & 0 & \ldots & -1 & 1\\
      0 & 0 & \ldots & 0 & -1
      \end{array}\right) \;,

    mapping :math:`\mathbb{R}^N \rightarrow \mathbb{R}^{N + 1}`.

    Parameters
    ----------
    x : array_like
      Input array
    axis : int
      Axis on which gradient is to be computed
    zero_pad : boolean
      Flag selecting type of gradient

    Returns
    -------
    xg : ndarray
      Output array
    """

    if zero_pad:
        xg = np.diff(x, axis=axis, prepend=0, append=0)
    else:
        slc = (slice(None),)*axis + (slice(-1, None),)
        xg = np.roll(x, -1, axis=axis) - x
        xg[slc] = 0.0
    return xg



def gradT(x, axis, zero_pad=False):
    """Compute transpose of gradient of `x` along axis `axis`.

    See :func:`grad` for a description of the dependency of the gradient
    operator on parameter `zero_pad`.

    Parameters
    ----------
    x : array_like
      Input array
    axis : int
      Axis on which gradient transpose is to be computed
    zero_pad : boolean
      Flag selecting type of gradient

    Returns
    -------
    xg : ndarray
      Output array
    """

    if zero_pad:
        xg = -np.diff(x, axis=axis)
    else:
        slc0 = (slice(None),) * axis
        xg = np.roll(x, 1, axis=axis) - x
        xg[slc0 + (slice(0, 1),)] = -x[slc0 + (slice(0, 1),)]
        xg[slc0 + (slice(-1, None),)] = x[slc0 + (slice(-2, -1),)]
    return xg



def gradient_filters(ndim, axes, axshp, dtype=None):
    r"""Construct a set of filters for computing gradients in the
    frequency domain.

    Parameters
    ----------
    ndim : integer
      Total number of dimensions in array in which gradients are to be
      computed
    axes : tuple of integers
      Axes on which gradients are to be computed
    axshp : tuple of integers
      Shape of axes on which gradients are to be computed
    dtype : dtype, optional (default np.float32)
      Data type of output arrays

    Returns
    -------
    Gf : ndarray
      Frequency domain gradient operators :math:`\hat{G}_i`
    GHGf : ndarray
      Sum of products :math:`\sum_i \hat{G}_i^H \hat{G}_i`
    """

    if dtype is None:
        dtype = np.float32
    g = np.zeros([2 if k in axes else 1 for k in range(ndim)] +
                 [len(axes),], dtype)
    for k in axes:
        g[(0,) * k + (slice(None),) + (0,) * (g.ndim - 2 - k) + (k,)] = \
            np.array([1, -1])
    if is_complex_dtype(dtype):
        Gf = fftn(g, axshp, axes=axes)
    else:
        Gf = rfftn(g, axshp, axes=axes)
    GHGf = np.sum(np.conj(Gf) * Gf, axis=-1).real
    return Gf, GHGf



def tikhonov_filter(s, lmbda, npd=16):
    r"""Lowpass filter based on Tikhonov regularization.

    Lowpass filter image(s) and return low and high frequency
    components, consisting of the lowpass filtered image and its
    difference with the input image. The lowpass filter is equivalent to
    Tikhonov regularization with `lmbda` as the regularization parameter
    and a discrete gradient as the operator in the regularization term,
    i.e. the lowpass component is the solution to

    .. math::
      \mathrm{argmin}_\mathbf{x} \; (1/2) \left\|\mathbf{x} - \mathbf{s}
      \right\|_2^2 + (\lambda / 2) \sum_i \| G_i \mathbf{x} \|_2^2 \;\;,

    where :math:`\mathbf{s}` is the input image, :math:`\lambda` is the
    regularization parameter, and :math:`G_i` is an operator that
    computes the discrete gradient along image axis :math:`i`. Once the
    lowpass component :math:`\mathbf{x}` has been computed, the highpass
    component is just :math:`\mathbf{s} - \mathbf{x}`.

    Parameters
    ----------
    s : array_like
      Input image or array of images.
    lmbda : float
      Regularization parameter controlling lowpass filtering.
    npd : int, optional (default=16)
      Number of samples to pad at image boundaries.

    Returns
    -------
    slp : array_like
      Lowpass image or array of images.
    shp : array_like
      Highpass image or array of images.
    """

    if np.isrealobj(s):
        fft = rfftn
        ifft = irfftn
    else:
        fft = fftn
        ifft = ifftn
    grv = np.array([-1.0, 1.0]).reshape([2, 1])
    gcv = np.array([-1.0, 1.0]).reshape([1, 2])
    Gr = fft(grv, (s.shape[0] + 2*npd, s.shape[1] + 2*npd), (0, 1))
    Gc = fft(gcv, (s.shape[0] + 2*npd, s.shape[1] + 2*npd), (0, 1))
    A = 1.0 + lmbda * (np.conj(Gr)*Gr + np.conj(Gc)*Gc).real
    if s.ndim > 2:
        A = A[(slice(None),)*2 + (np.newaxis,)*(s.ndim-2)]
    sp = np.pad(s, ((npd, npd),)*2 + ((0, 0),)*(s.ndim-2), 'symmetric')
    spshp = sp.shape
    sp = fft(sp, axes=(0, 1))
    sp /= A
    sp = ifft(sp, s=spshp[0:2], axes=(0, 1))
    slp = sp[npd:(sp.shape[0] - npd), npd:(sp.shape[1] - npd)]
    shp = s - slp
    return slp.astype(s.dtype), shp.astype(s.dtype)



def gaussian(shape, sd=1.0):
    """Sample a multivariate Gaussian pdf, normalised to have unit sum.

    Parameters
    ----------
    shape : tuple
      Shape of output array.
    sd : float, optional (default 1.0)
      Standard deviation of Gaussian pdf.

    Returns
    -------
    gc : ndarray
      Sampled Gaussian pdf.
    """

    gfn = lambda x, sd: np.exp(-(x**2) / (2.0 * sd**2)) / \
                        (np.sqrt(2.0 * np.pi) *sd)
    gc = 1.0
    if isinstance(shape, int):
        shape = (shape,)
    for k, n in enumerate(shape):
        x = np.linspace(-3.0, 3.0, n).reshape(
            (1,) * k + (n,) + (1,) * (len(shape) - k - 1))
        gc = gc * gfn(x, sd)
    gc /= np.sum(gc)
    return gc



def local_contrast_normalise(s, n=7, c=None):
    """Local contrast normalisation of an image.

    Perform local contrast normalisation :cite:`jarret-2009-what` of
    an image, consisting of subtraction of the local mean and division
    by the local norm. The original image can be reconstructed from the
    contrast normalised image as (`snrm` * `scn`) + `smn`.

    Parameters
    ----------
    s : array_like
      Input image or array of images.
    n : int, optional (default 7)
      The size of the local region used for normalisation is :math:`2n+1`.
    c : float, optional (default None)
      The smallest value that can be used in the divisive normalisation.
      If `None`, this value is set to the mean of the local region norms.

    Returns
    -------
    scn : ndarray
      Contrast normalised image(s)
    smn : ndarray
      Additive normalisation correction
    snrm : ndarray
      Multiplicative normalisation correction
    """

    # Construct region weighting filter
    N = 2 * n + 1
    g = gaussian((N, N), sd=1.0)
    # Compute required image padding
    pd = ((n, n),) * 2
    if s.ndim > 2:
        g = g[..., np.newaxis]
        pd += ((0, 0),)
    sp = np.pad(s, pd, mode='symmetric')
    # Compute local mean and subtract from image
    smn = np.roll(fftconv(g, sp), (-n, -n), axis=(0, 1))
    s1 = sp - smn
    # Compute local norm
    snrm = np.roll(np.sqrt(np.clip(fftconv(g, s1**2), 0.0, np.inf)),
                   (-n, -n), axis=(0, 1))
    # Set c parameter if not specified
    if c is None:
        c = np.mean(snrm, axis=(0, 1), keepdims=True)
    # Divide mean-subtracted image by corrected local norm
    snrm = np.maximum(c, snrm)
    s2 = s1 / snrm
    # Return contrast normalised image and normalisation components
    return s2[n:-n, n:-n], smn[n:-n, n:-n], snrm[n:-n, n:-n]
