# -*- coding: utf-8 -*-
# Copyright (C) 2019-2025 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Interpolation and regression functions."""

from __future__ import absolute_import, division
from builtins import range

import warnings
import numpy as np
import scipy
import scipy.optimize as sco
from scipy.interpolate import griddata, RectBivariateSpline



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
    x = range(1, img.shape[0], 2)
    y = range(1, img.shape[1], 2)
    #fi = interp2d(x, y, img[1::2, 1::2])
    fi = RectBivariateSpline(x, y, img[1::2, 1::2], s=0)
    sr = fi(range(0, img.shape[0]), range(0, img.shape[1]))

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
    x = range(0, img.shape[0], 2)
    y = range(0, img.shape[1], 2)
    #fi = interp2d(x, y, img[0::2, 0::2])
    fi = RectBivariateSpline(x, y, img[0::2, 0::2], s=0)
    sb = fi(range(0, img.shape[0]), range(0, img.shape[1]))

    return np.dstack((sr, sg, sb))



# Deal with introduction of new method option for scipy.optimize.linprog
# in SciPy 1.6.0
_spv = scipy.__version__.split('.')
if int(_spv[0]) > 1 or (int(_spv[0]) == 1 and int(_spv[1]) >= 6):
    def _linprog(*args, **kwargs):
        kwargs['method'] = 'highs'
        return sco.linprog(*args, **kwargs)
else:
    def _linprog(*args, **kwargs):
        return sco.linprog(*args, **kwargs)



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
    a linear program <https://stats.stackexchange.com/a/12564>`__. The
    current implementation is only suitable for small-scale problems.

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
        res = _linprog(c, A_ub, b_ub, bounds=(None, None))
    if res.success is False:
        raise ValueError('scipy.optimize.linprog failed with status %d' %
                         res.status)
    return res.x[M:]



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
    program <https://stats.stackexchange.com/a/12564>`__. The
    current implementation is only suitable for small-scale problems.

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
        res = _linprog(c, A_ub, b_ub, bounds=(None, None))
    if res.success is False:
        raise ValueError('scipy.optimize.linprog failed with status %d' %
                         res.status)
    return res.x[1:]



def lanczos_kernel(x, a=3):
    r"""Lanczos interpolation kernel.

    Compute the `Lanczos interpolation kernel
    <https://en.wikipedia.org/wiki/Lanczos_resampling>`__

    .. math::
      L(x) = \left\{ \begin{array}{ll} \mathrm{sinc}(x)\,
      \mathrm{sinc}(x/a) & \;\text{if}\; -a < x < a \\ 0 &
      \text{otherwise} \;, \end{array} \right.

    where :math:`a \in \mathbb{Z}^+`.

    Parameters
    ----------
    x : float or ndarray
      Sampling point(s) at which to compute the kernel
    a : int, optional (default 3)
      Kernel size parameter

    Returns
    -------
    y : float or ndarray
      Kernel evaluated at sampling point(s)
    """

    return np.logical_and(x > -a, x < a) * np.sinc(x) * np.sinc(x / a)



def interpolation_points(N, include_zero=True):
    """Evenly spaced interpolation points.

    Construct a set of `N` evenly spaced interpolation points for
    samples on an integer grid.

    Parameters
    ----------
    N : int
      Number of interpolation points
    include_zero : bool, optional (default True)
      Flag indicating whether to include zero in the set of points

    Returns
    -------
    y : ndarray
      Array of interpolation points
    """

    if include_zero:
        return np.arange(-((N - 1) // 2), (N // 2) + 1) / float(N)
    else:
        return np.hstack((np.arange(-(N // 2), 0),
                          np.arange(1, ((N + 1) // 2) + 1))) / (N + 1.0)



def lanczos_filters(sz, a=3, collapse_axes=True):
    """Multi-dimensional Lanczos interpolation filters.

    Construct a set of `Lanczos interpolation filters
    <https://en.wikipedia.org/wiki/Lanczos_resampling>`__.
    Multi-dimensional filters are constructed as tensor products of
    one-dimensional filters.

    Parameters
    ----------
    sz : tuple of int or tuple of array_like
      Tuple specifying the resampling points for each filter dimension.
      Each entry may be an array of resampling points or an integer, in
      which case the resampling grid consists of the specified number of
      equi-spaced points
    a : int, optional (default 3)
      Kernel size parameter
    collapse_axes : bool, optional (default True)
      Flag indicating whether to collapse the output axes corresponding
      to different filters for each filter dimension

    Returns
    -------
    y : ndarray
      Array of interpolation filters
    """

    if isinstance(sz, int):
        sz = (sz,)
    ndim = len(sz)
    h = 1.0
    for n in range(ndim):
        if isinstance(sz[n], int):
            x = interpolation_points(sz[n])
        else:
            x = np.array(sz[n])
            if x.ndim != 1:
                raise ValueError('Size tuple entry not an integer or 1d array')
            if x.min() < -1.0 or x.max() > 1.0:
                raise ValueError('Interpolation points must be in [-1, 1]')
        gx = np.arange(-a, a + 1)[:, np.newaxis] + x[np.newaxis, :]
        hn = lanczos_kernel(gx, a=a)
        hn /= np.sum(hn, axis=0, keepdims=True)
        shp = (1,) * n + (hn.shape[0],) + (1,) * (ndim - n - 1) + \
              (1,) * n + (hn.shape[1],) + (1,) * (ndim - n - 1)
        h = h * hn.reshape(shp)

    if collapse_axes:
        h = h.reshape(h.shape[0:ndim] + (-1,))

    return h
