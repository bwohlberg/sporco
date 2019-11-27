# -*- coding: utf-8 -*-
# Copyright (C) 2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Interpolation and regression functions."""

from __future__ import absolute_import, division
from builtins import range

import warnings
import numpy as np
import scipy.optimize as sco
from scipy.interpolate import interp2d, griddata

from sporco._util import renamed_function


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


def bilinear_demosaic(img):
    """Demosaicing by bilinear interpolation.

    The input is assumed to be an image formed with a `colour filter
    array <https://en.wikipedia.org/wiki/Color_filter_array>`__ with the
    pattern

    ::

      B G B G ...
      G R G R ...
      B G B G ...
      G R G R ...
      . . . . .
      . . . .   .
      . . . .     .


    Parameters
    ----------
    img : 2d ndarray
      A 2d array representing an image formed with a colour filter array

    Returns
    -------
    imgd : 3d ndarray
      Demosaiced 3d image
    """

    # Interpolate red channel
    x = range(1, img.shape[1], 2)
    y = range(1, img.shape[0], 2)
    fi = interp2d(x, y, img[1::2, 1::2])
    sr = fi(range(0, img.shape[1]), range(0, img.shape[0]))

    # Interpolate green channel. We can't use `interp2d` here because
    # the green channel isn't arranged in a simple grid pattern. Since
    # the locations of the green samples can be represented as the union
    # of two grids, we use `griddata` with an array of coordinates
    # constructed by stacking the coordinates of these two grids
    x0, y0 = np.mgrid[0:img.shape[0]:2, 1:img.shape[1]:2]
    x1, y1 = np.mgrid[1:img.shape[0]:2, 0:img.shape[1]:2]
    xy01 = np.vstack((np.hstack((x0.ravel().T, x1.ravel().T)),
                      np.hstack((y0.ravel().T, y1.ravel().T)))).T
    z = np.hstack((img[0::2, 1::2].ravel(), img[1::2, 0::2].ravel()))
    x2, y2 = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    xy2 = np.vstack((x2.ravel(), y2.ravel())).T
    sg = griddata(xy01, z, xy2, method='linear').reshape(img.shape[0:2])
    if np.isnan(sg[0, 0]):
        sg[0, 0] = (sg[0, 1] + sg[1, 0]) / 2.0
    if np.isnan(sg[0, -1]):
        sg[0, -1] = (sg[0, -2] + sg[1, -1]) / 2.0
    if np.isnan(sg[-1, 0]):
        sg[-1, 0] = (sg[-2, 0] + sg[-1, 1]) / 2.0
    if np.isnan(sg[-1, -1]):
        sg[-1, -1] = (sg[-2, -1] + sg[-1, -2]) / 2.0

    # Interpolate blue channel
    x = range(0, img.shape[1], 2)
    y = range(0, img.shape[0], 2)
    fi = interp2d(x, y, img[0::2, 0::2])
    sb = fi(range(0, img.shape[1]), range(0, img.shape[0]))

    return np.dstack((sr, sg, sb))



@renamed_function(depname='lstabsdev', depmod='sporco.util')
def lstabsdev(A, b):
    r"""Least absolute deviations (LAD) linear regression.

    Solve the linear regression problem

    .. math::
      \mathrm{argmin}_\mathbf{x} \; \left\| A \mathbf{x} - \mathbf{b}
      \right\|_1 \;\;.

    The interface is similar to that of :func:`numpy.linalg.lstsq` in
    that `np.linalg.lstsq(A, b)` solves the same linear regression
    problem, but with a least squares rather than a least absolute
    deviations objective. Unlike :func:`numpy.linalg.lstsq`, `b` is
    required to be a 1-d array. The solution is obtained via `mapping to
    a linear program <https://stats.stackexchange.com/a/12564>`__.

    Parameters
    ----------
    A : (M, N) array_like
      Regression coefficient matrix
    b : (M,) array_like
      Regression ordinate / dependent variable

    Returns
    -------
    x : (N,) ndarray
      Least absolute deviations solution
    """

    M, N = A.shape
    c = np.zeros((M + N,))
    c[0:M] = 1.0
    I = np.identity(M)
    A_ub = np.hstack((np.vstack((-I, -I)), np.vstack((-A, A))))
    b_ub = np.hstack((-b, b))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=sco.OptimizeWarning)
        res = sco.linprog(c, A_ub, b_ub)
    if res.success is False:
        raise ValueError('scipy.optimize.linprog failed with status %d' %
                         res.status)
    return res.x[M:]



@renamed_function(depname='lstmaxdev', depmod='sporco.util')
def lstmaxdev(A, b):
    r"""Least maximum deviation (least maximum error) linear regression.

    Solve the linear regression problem

    .. math::
      \mathrm{argmin}_\mathbf{x} \; \left\| A \mathbf{x} - \mathbf{b}
      \right\|_{\infty} \;\;.

    The interface is similar to that of :func:`numpy.linalg.lstsq` in
    that `np.linalg.lstsq(A, b)` solves the same linear regression
    problem, but with a least squares rather than a least maximum
    error objective. Unlike :func:`numpy.linalg.lstsq`, `b` is required
    to be a 1-d array. The solution is obtained via `mapping to a linear
    program <https://stats.stackexchange.com/a/12564>`__.

    Parameters
    ----------
    A : (M, N) array_like
      Regression coefficient matrix
    b : (M,) array_like
      Regression ordinate / dependent variable

    Returns
    -------
    x : (N,) ndarray
      Least maximum deviation solution
    """

    M, N = A.shape
    c = np.zeros((N + 1,))
    c[0] = 1.0
    one = np.ones((M, 1))
    A_ub = np.hstack((np.vstack((-one, -one)), np.vstack((-A, A))))
    b_ub = np.hstack((-b, b))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=sco.OptimizeWarning)
        res = sco.linprog(c, A_ub, b_ub)
    if res.success is False:
        raise ValueError('scipy.optimize.linprog failed with status %d' %
                         res.status)
    return res.x[1:]

