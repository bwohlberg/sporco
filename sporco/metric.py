# -*- coding: utf-8 -*-
# Copyright (C) 2016-2025 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Image quality metrics and related functions

Note that the well-known SSIM metric is not implemented here as it is
available in a number of other Python packages, including:

  - `scikit-image <https://github.com/scikit-image/scikit-image>`_
  - `PyMetrikz <https://bitbucket.org/kuraiev/pymetrikz>`_
  - `Video Quality Metrics <https://github.com/aizvorski/video-quality>`_
  - `pyssim <https://github.com/jterrace/pyssim>`_

Some implementations are also available in unpackaged collections of
Python code:

  - `src <https://github.com/helderc/src>`_
  - `python <https://github.com/mubeta06/python>`_

|

"""

from __future__ import division

import numpy as np
from scipy import ndimage
from scipy import signal

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


def mae(vref, vcmp):
    """
    Compute Mean Absolute Error (MAE) between two images.

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image

    Returns
    -------
    x : float
      MAE between `vref` and `vcmp`
    """

    if np.iscomplexobj(vref) or np.iscomplexobj(vcmp):
        dtype = np.complex128
    else:
        dtype = np.float64
    r = np.asarray(vref, dtype=dtype).ravel()
    c = np.asarray(vcmp, dtype=dtype).ravel()
    return np.mean(np.abs(r - c))



def mse(vref, vcmp):
    """
    Compute Mean Squared Error (MSE) between two images.

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image

    Returns
    -------
    x : float
      MSE between `vref` and `vcmp`
    """

    if np.iscomplexobj(vref) or np.iscomplexobj(vcmp):
        dtype = np.complex128
    else:
        dtype = np.float64
    r = np.asarray(vref, dtype=dtype).ravel()
    c = np.asarray(vcmp, dtype=dtype).ravel()
    return np.mean(np.abs(r - c)**2)



def snr(vref, vcmp):
    """
    Compute Signal to Noise Ratio (SNR) of two images.

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image

    Returns
    -------
    x : float
      SNR of `vcmp` with respect to `vref`
    """

    dv = np.var(vref)
    with np.errstate(divide='ignore'):
        rt = dv / mse(vref, vcmp)
    return 10.0 * np.log10(rt)



def psnr(vref, vcmp, rng=None):
    """
    Compute Peak Signal to Noise Ratio (PSNR) of two images. The PSNR
    calculation defaults to using the less common definition in terms
    of the actual range (i.e. max minus min) of the reference signal
    instead of the maximum possible range for the data type
    (i.e. :math:`2^b-1` for a :math:`b` bit representation).

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image
    rng : None or int, optional (default None)
      Signal range, either the value to use (e.g. 255 for 8 bit samples) or
      None, in which case the actual range of the reference signal is used

    Returns
    -------
    x : float
      PSNR of `vcmp` with respect to `vref`
    """

    if rng is None:
        rng = np.abs(vref.max() - vref.min())
    dv = (rng + 0.0)**2
    with np.errstate(divide='ignore'):
        rt = dv / mse(vref, vcmp)
    return 10.0 * np.log10(rt)



def isnr(vref, vdeg, vrst):
    """
    Compute Improvement Signal to Noise Ratio (ISNR) for reference,
    degraded, and restored images.

    Parameters
    ----------
    vref : array_like
      Reference image
    vdeg : array_like
      Degraded image
    vrst : array_like
      Restored image

    Returns
    -------
    x : float
      ISNR of `vrst` with respect to `vref` and `vdeg`
    """

    msedeg = mse(vref, vdeg)
    mserst = mse(vref, vrst)
    with np.errstate(divide='ignore'):
        rt = msedeg / mserst
    return 10.0 * np.log10(rt)



def bsnr(vblr, vnsy):
    """
    Compute Blurred Signal to Noise Ratio (BSNR) for a blurred and noisy
    image.

    Parameters
    ----------
    vblr : array_like
      Blurred noise free image
    vnsy : array_like
      Blurred image with additive noise

    Returns
    -------
    x : float
      BSNR of `vnsy` with respect to `vblr` and `vdeg`
    """

    blrvar = np.var(vblr)
    nsevar = np.var(vnsy - vblr)
    with np.errstate(divide='ignore'):
        rt = blrvar / nsevar
    return 10.0 * np.log10(rt)



def pamse(vref, vcmp, rescale=True):
    """
    Compute Perceptual-fidelity Aware Mean Squared Error (PAMSE) IQA metric
    :cite:`xue-2013-perceptual`. This implementation is a translation of the
    reference Matlab implementation provided by the authors of
    :cite:`xue-2013-perceptual`.

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image
    rescale : bool, optional (default True)
      Rescale inputs so that `vref` has a maximum value of 255, as assumed
      by reference implementation

    Returns
    -------
    score : float
      PAMSE IQA metric
    """

    # Calculate difference, promoting to float if vref and vcmp have integer
    # dtype
    emap = np.asarray(vref, dtype=np.float64) - \
        np.asarray(vcmp, dtype=np.float64)
    # Input images in reference code on which this implementation is
    # based are assumed to be on range [0,...,255].
    if rescale:
        emap *= (255.0 / vref.max())
    sigma = 0.8
    herr = ndimage.gaussian_filter(emap, sigma)
    score = np.mean(herr**2)
    return score



def gmsd(vref, vcmp, rescale=True, returnMap=False):
    """
    Compute Gradient Magnitude Similarity Deviation (GMSD) IQA metric
    :cite:`xue-2014-gradient`. This implementation is a translation of the
    reference Matlab implementation provided by the authors of
    :cite:`xue-2014-gradient`.

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image
    rescale : bool, optional (default True)
      Rescale inputs so that `vref` has a maximum value of 255, as assumed
      by reference implementation
    returnMap : bool, optional (default False)
      Flag indicating whether quality map should be returned in addition to
      scalar score

    Returns
    -------
    score : float
      GMSD IQA metric
    quality_map : ndarray
      Quality map
    """

    # Input images in reference code on which this implementation is
    # based are assumed to be on range [0,...,255].
    if rescale:
        scl = (255.0 / vref.max())
    else:
        scl = np.float32(1.0)

    T = 170.0
    dwn = 2
    dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0
    dy = dx.T

    ukrn = np.ones((2, 2)) / 4.0
    aveY1 = signal.convolve2d(scl * vref, ukrn, mode='same', boundary='symm')
    aveY2 = signal.convolve2d(scl * vcmp, ukrn, mode='same', boundary='symm')
    Y1 = aveY1[0::dwn, 0::dwn]
    Y2 = aveY2[0::dwn, 0::dwn]

    IxY1 = signal.convolve2d(Y1, dx, mode='same', boundary='symm')
    IyY1 = signal.convolve2d(Y1, dy, mode='same', boundary='symm')
    grdMap1 = np.sqrt(IxY1**2 + IyY1**2)

    IxY2 = signal.convolve2d(Y2, dx, mode='same', boundary='symm')
    IyY2 = signal.convolve2d(Y2, dy, mode='same', boundary='symm')
    grdMap2 = np.sqrt(IxY2**2 + IyY2**2)

    quality_map = (2*grdMap1*grdMap2 + T) / (grdMap1**2 + grdMap2**2 + T)
    score = np.std(quality_map)

    if returnMap:
        return (score, quality_map)
    else:
        return score
