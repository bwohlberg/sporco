# -*- coding: utf-8 -*-
# Copyright (C) 2015-2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Linear algebra functions"""

from __future__ import division
from builtins import range

import multiprocessing
import numpy as np
import scipy
from scipy import linalg
from scipy import fftpack
from scipy.sparse.linalg import LinearOperator, cg
import pyfftw
try:
    import numexpr as ne
except ImportError:
    have_numexpr = False
else:
    have_numexpr = True

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



__all__ = ['complex_dtype', 'pyfftw_byte_aligned', 'pyfftw_empty_aligned',
           'pyfftw_rfftn_empty_aligned', 'fftn', 'ifftn', 'rfftn', 'irfftn',
           'dctii', 'idctii', 'fftconv', 'inner', 'dot', 'solvedbi_sm',
           'solvedbi_sm_c', 'solvedbd_sm', 'solvedbd_sm_c', 'solvemdbi_ism',
           'solvemdbi_rsm', 'solvemdbi_cg', 'lu_factor', 'lu_solve_ATAI',
           'lu_solve_AATI', 'cho_factor', 'cho_solve_ATAI', 'cho_solve_AATI',
           'zpad', 'Gax', 'GTax', 'GradientFilters', 'zdivide', 'shrink1',
           'shrink2', 'shrink12', 'proj_l2ball', 'promote16', 'atleast_nd',
           'split', 'blockcirculant', 'fl2norm2', 'rfl2norm2', 'rrs']



pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(300)

pyfftw_threads = multiprocessing.cpu_count()
"""Global variable setting the number of threads used in :mod:`pyfftw`
computations"""


def complex_dtype(dtype):
    """
    Construct the corresponding complex dtype for a given real dtype,
    e.g. the complex dtype corresponding to ``np.float32`` is
    ``np.complex64``.

    Parameters
    ----------
    dtype : dtype
      A real dtype, e.g. np.float32, np.float64

    Returns
    -------
    cdtype : dtype
      The complex dtype corresponding to the input dtype
    """

    return (np.zeros(1, dtype) + 1j).dtype



def pyfftw_byte_aligned(array, dtype=None, n=None):
    """
    Construct a byte-aligned array for efficient use by :mod:`pyfftw`.
    This function is a wrapper for :func:`pyfftw.byte_align`

    Parameters
    ----------
    array : ndarray
      Input array
    dtype : dtype, optional (default None)
      Output array dtype
    n : int, optional (default None)
      Output array should be aligned to n-byte boundary

    Returns
    -------
    a :  ndarray
      Array with required byte-alignment
    """

    return pyfftw.byte_align(array, n=n, dtype=dtype)



def pyfftw_empty_aligned(shape, dtype, order='C', n=None):
    """
    Construct an empty byte-aligned array for efficient use by :mod:`pyfftw`.
    This function is a wrapper for :func:`pyfftw.empty_aligned`

    Parameters
    ----------
    shape : sequence of ints
      Output array shape
    dtype : dtype
      Output array dtype
    order : {'C', 'F'}, optional (default 'C')
      Specify whether arrays should be stored in row-major (C-style) or
      column-major (Fortran-style) order
    n : int, optional (default None)
      Output array should be aligned to n-byte boundary

    Returns
    -------
    a :  ndarray
      Empty array with required byte-alignment
    """

    return pyfftw.empty_aligned(shape, dtype, order, n)



def pyfftw_rfftn_empty_aligned(shape, axes, dtype, order='C', n=None):
    """
    Construct an empty byte-aligned array for efficient use by :mod:`pyfftw`
    functions :func:`pyfftw.interfaces.numpy_fft.rfftn` and
    :func:`pyfftw.interfaces.numpy_fft.irfftn`. The shape of the
    empty array is appropriate for the output of
    :func:`pyfftw.interfaces.numpy_fft.rfftn` applied
    to an array of the shape specified by parameter `shape`, and for the
    input of the corresponding :func:`pyfftw.interfaces.numpy_fft.irfftn`
    call that reverses this operation.

    Parameters
    ----------
    shape : sequence of ints
      Output array shape
    axes : sequence of ints
      Axes on which the FFT will be computed
    dtype : dtype
      Real dtype from which the complex dtype of the output array is derived
    order : {'C', 'F'}, optional (default 'C')
      Specify whether arrays should be stored in row-major (C-style) or
      column-major (Fortran-style) order
    n : int, optional (default None)
      Output array should be aligned to n-byte boundary

    Returns
    -------
    a :  ndarray
      Empty array with required byte-alignment
    """

    ashp = list(shape)
    raxis = axes[-1]
    ashp[raxis] = ashp[raxis] // 2 + 1
    cdtype = complex_dtype(dtype)
    return pyfftw.empty_aligned(ashp, cdtype, order, n)



def fftn(a, s=None, axes=None):
    """
    Compute the multi-dimensional discrete Fourier transform. This function
    is a wrapper for :func:`pyfftw.interfaces.numpy_fft.fftn`,
    with an interface similar to that of :func:`numpy.fft.fftn`.

    Parameters
    ----------
    a : array_like
      Input array (can be complex)
    s : sequence of ints, optional (default None)
      Shape of the output along each transformed axis (input is cropped or
      zero-padded to match).
    axes : sequence of ints, optional (default None)
      Axes over which to compute the DFT.

    Returns
    -------
    af : complex ndarray
      DFT of input array
    """

    return pyfftw.interfaces.numpy_fft.fftn(
        a, s=s, axes=axes, overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=pyfftw_threads)



def ifftn(a, s=None, axes=None):
    """
    Compute the multi-dimensional inverse discrete Fourier transform.
    This function is a wrapper for :func:`pyfftw.interfaces.numpy_fft.ifftn`,
    with an interface similar to that of :func:`numpy.fft.ifftn`.

    Parameters
    ----------
    a : array_like
      Input array (can be complex)
    s : sequence of ints, optional (default None)
      Shape of the output along each transformed axis (input is cropped or
      zero-padded to match).
    axes : sequence of ints, optional (default None)
      Axes over which to compute the inverse DFT.

    Returns
    -------
    af : complex ndarray
      Inverse DFT of input array
    """

    return pyfftw.interfaces.numpy_fft.ifftn(
        a, s=s, axes=axes, overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=pyfftw_threads)



def rfftn(a, s=None, axes=None):
    """
    Compute the multi-dimensional discrete Fourier transform for real input.
    This function is a wrapper for :func:`pyfftw.interfaces.numpy_fft.rfftn`,
    with an interface similar to that of :func:`numpy.fft.rfftn`.

    Parameters
    ----------
    a : array_like
      Input array (taken to be real)
    s : sequence of ints, optional (default None)
      Shape of the output along each transformed axis (input is cropped or
      zero-padded to match).
    axes : sequence of ints, optional (default None)
      Axes over which to compute the DFT.

    Returns
    -------
    af : complex ndarray
      DFT of input array
    """

    return pyfftw.interfaces.numpy_fft.rfftn(
        a, s=s, axes=axes, overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=pyfftw_threads)



def irfftn(a, s, axes=None):
    """
    Compute the inverse of the multi-dimensional discrete Fourier transform
    for real input. This function is a wrapper for
    :func:`pyfftw.interfaces.numpy_fft.irfftn`, with an interface similar to
    that of :func:`numpy.fft.irfftn`.

    Parameters
    ----------
    a : array_like
      Input array
    s : sequence of ints
      Shape of the output along each transformed axis (input is cropped or
      zero-padded to match). This parameter is not optional because, unlike
      :func:`ifftn`, the output shape cannot be uniquely determined from
      the input shape.
    axes : sequence of ints, optional (default None)
      Axes over which to compute the inverse DFT.

    Returns
    -------
    af : ndarray
      Inverse DFT of input array
    """

    return pyfftw.interfaces.numpy_fft.irfftn(
        a, s=s, axes=axes, overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=pyfftw_threads)



def dctii(x, axes=None):
    """
    Compute a multi-dimensional DCT-II over specified array axes. This
    function is implemented by calling the one-dimensional DCT-II
    :func:`scipy.fftpack.dct` with normalization mode 'ortho' for each
    of the specified axes.

    Parameters
    ----------
    a : array_like
      Input array
    axes : sequence of ints, optional (default None)
      Axes over which to compute the DCT-II.

    Returns
    -------
    y : ndarray
      DCT-II of input array
    """

    if axes is None:
        axes = list(range(x.ndim))
    for ax in axes:
        x = fftpack.dct(x, type=2, axis=ax, norm='ortho')
    return x



def idctii(x, axes=None):
    """
    Compute a multi-dimensional inverse DCT-II over specified array axes.
    This function is implemented by calling the one-dimensional inverse
    DCT-II :func:`scipy.fftpack.idct` with normalization mode 'ortho'
    for each of the specified axes.

    Parameters
    ----------
    a : array_like
      Input array
    axes : sequence of ints, optional (default None)
      Axes over which to compute the inverse DCT-II.

    Returns
    -------
    y : ndarray
      Inverse DCT-II of input array
    """

    if axes is None:
        axes = list(range(x.ndim))
    for ax in axes[::-1]:
        x = fftpack.idct(x, type=2, axis=ax, norm='ortho')
    return x



def fftconv(a, b, axes=(0, 1)):
    """
    Compute a multi-dimensional convolution via the Discrete Fourier Transform.

    Parameters
    ----------
    a : array_like
      Input array
    b : array_like
      Input array
    axes : sequence of ints, optional (default (0, 1))
      Axes on which to perform convolution

    Returns
    -------
    ab : ndarray
      Convolution of input arrays, a and b, along specified axes
    """

    if np.isrealobj(a) and np.isrealobj(b):
        fft = rfftn
        ifft = irfftn
    else:
        fft = fftn
        ifft = ifftn
    dims = np.maximum([a.shape[i] for i in axes], [b.shape[i] for i in axes])
    af = fft(a, dims, axes)
    bf = fft(b, dims, axes)
    return ifft(af * bf, dims, axes)



def inner(x, y, axis=-1):
    """
    Compute inner product of x and y on specified axis, equivalent to
    :code:`np.sum(x * y, axis=axis, keepdims=True)`.

    Parameters
    ----------
    x : array_like
      Input array x
    y : array_like
      Input array y
    axis : int, optional (default -1)
      Axis over which to compute the sum

    Returns
    -------
    y : ndarray
      Inner product array equivalent to summing x*y over the specified
      axis
    """

    # Convert negative axis to positive
    if axis < 0:
        axis = x.ndim + axis

    # If sum not on axis 0, roll specified axis to 0 position
    if axis == 0:
        xr = x
        yr = y
    else:
        xr = np.rollaxis(x, axis, 0)
        yr = np.rollaxis(y, axis, 0)

    # Efficient inner product on axis 0
    if np.__version__ == '1.14.0':
        # Setting of optimize flag due to
        #    https://github.com/numpy/numpy/issues/10343
        ip = np.einsum(xr, [0, Ellipsis], yr, [0, Ellipsis],
                       optimize=False)[np.newaxis, ...]
    else:
        ip = np.einsum(xr, [0, Ellipsis], yr, [0, Ellipsis])[np.newaxis, ...]

    # Roll axis back to original position if necessary
    if axis != 0:
        ip = np.rollaxis(ip, 0, axis + 1)

    return ip



def dot(a, b, axis=-2):
    """
    Compute the matrix product of `a` and the specified axes of `b`,
    with broadcasting over the remaining axes of `b`. This function is
    a generalisation of :func:`numpy.dot`, supporting sum product over
    an arbitrary axis instead of just over the last axis.

    If `a` and `b` are both 2D arrays, `dot` gives the same result as
    :func:`numpy.dot`. If `b` has more than 2 axes, the result is
    obtained as follows (where `a` has shape ``(M0, M1)`` and `b` has
    shape ``(N0, N1, ..., M1, Nn, ...)``):

       #. Reshape `a` to shape ``( 1,  1, ..., M0, M1,  1, ...)``
       #. Reshape `b` to shape ``(N0, N1, ...,  1, M1, Nn, ...)``
       #. Take the broadcast product and sum over the specified axis (the
          axis with dimension `M1` in this example) to give an array of
          shape ``(N0, N1, ...,  M0,  1, Nn, ...)``
       #. Remove the singleton axis created by the summation to give
          an array of shape ``(N0, N1, ...,  M0, Nn, ...)``

    Parameters
    ----------
    a : array_like, 2D
      First component of product
    b : array_like, 2D or greater
      Second component of product
    axis : integer, optional (default -2)
      Axis of `b` over which sum is to be taken

    Returns
    -------
    prod : ndarray
      Matrix product of `a` and specified axes of `b`, with broadcasting
      over the remaining axes of `b`
    """

    # Ensure axis specification is positive
    if axis < 0:
        axis = b.ndim + axis
    # Insert singleton axis into b
    bx = np.expand_dims(b, axis)
    # Calculate index of required singleton axis in a and insert it
    axshp = [1] * bx.ndim
    axshp[axis:axis + 2] = a.shape
    ax = a.reshape(axshp)
    # Calculate indexing expression required to remove singleton axis in
    # product
    idxexp = [slice(None)] * bx.ndim
    idxexp[axis + 1] = 0
    # Compute and return product
    return np.sum(ax * bx, axis=axis+1, keepdims=True)[tuple(idxexp)]



def solvedbi_sm(ah, rho, b, c=None, axis=4):
    r"""
    Solve a diagonal block linear system with a scaled identity term
    using the Sherman-Morrison equation.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

    .. math::
      (\rho I + \mathbf{a} \mathbf{a}^H ) \; \mathbf{x} = \mathbf{b} \;\;.

    In this equation inner products and matrix products are taken along
    the specified axis of the corresponding multi-dimensional arrays; the
    solutions are independent over the other axes.

    Parameters
    ----------
    ah : array_like
      Linear system component :math:`\mathbf{a}^H`
    rho : float
      Linear system parameter :math:`\rho`
    b : array_like
      Linear system component :math:`\mathbf{b}`
    c : array_like, optional (default None)
      Solution component :math:`\mathbf{c}` that may be pre-computed using
      :func:`solvedbi_sm_c` and cached for re-use.
    axis : int, optional (default 4)
      Axis along which to solve the linear system

    Returns
    -------
    x : ndarray
      Linear system solution :math:`\mathbf{x}`
    """

    a = np.conj(ah)
    if c is None:
        c = solvedbi_sm_c(ah, a, rho, axis)
    if have_numexpr:
        cb = inner(c, b, axis=axis)
        return ne.evaluate('(b - (a * cb)) / rho')
    else:
        return (b - (a * inner(c, b, axis=axis))) / rho



def solvedbi_sm_c(ah, a, rho, axis=4):
    r"""
    Compute cached component used by :func:`solvedbi_sm`.

    Parameters
    ----------
    ah : array_like
      Linear system component :math:`\mathbf{a}^H`
    a : array_like
      Linear system component :math:`\mathbf{a}`
    rho : float
      Linear system parameter :math:`\rho`
    axis : int, optional (default 4)
      Axis along which to solve the linear system

    Returns
    -------
    c : ndarray
      Argument :math:`\mathbf{c}` used by :func:`solvedbi_sm`
    """

    return ah / (inner(ah, a, axis=axis) + rho)



def solvedbd_sm(ah, d, b, c=None, axis=4):
    r"""
    Solve a diagonal block linear system with a diagonal term
    using the Sherman-Morrison equation.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

    .. math::
      (\mathbf{d}  + \mathbf{a} \mathbf{a}^H ) \; \mathbf{x} = \mathbf{b} \;\;.

    In this equation inner products and matrix products are taken along
    the specified axis of the corresponding multi-dimensional arrays; the
    solutions are independent over the other axes.

    Parameters
    ----------
    ah : array_like
      Linear system component :math:`\mathbf{a}^H`
    d : array_like
      Linear system parameter :math:`\mathbf{d}`
    b : array_like
      Linear system component :math:`\mathbf{b}`
    c : array_like, optional (default None)
      Solution component :math:`\mathbf{c}` that may be pre-computed using
      :func:`solvedbd_sm_c` and cached for re-use.
    axis : int, optional (default 4)
      Axis along which to solve the linear system

    Returns
    -------
    x : ndarray
      Linear system solution :math:`\mathbf{x}`
    """

    a = np.conj(ah)
    if c is None:
        c = solvedbd_sm_c(ah, a, d, axis)
    if have_numexpr:
        cb = inner(c, b, axis=axis)
        return ne.evaluate('(b - (a * cb)) / d')
    else:
        return (b - (a * inner(c, b, axis=axis))) / d



def solvedbd_sm_c(ah, a, d, axis=4):
    r"""
    Compute cached component used by :func:`solvedbd_sm`.

    Parameters
    ----------
    ah : array_like
      Linear system component :math:`\mathbf{a}^H`
    a : array_like
      Linear system component :math:`\mathbf{a}`
    d : array_like
      Linear system parameter :math:`\mathbf{d}`
    axis : int, optional (default 4)
      Axis along which to solve the linear system

    Returns
    -------
    c : ndarray
      Argument :math:`\mathbf{c}` used by :func:`solvedbd_sm`
    """

    return (ah / d) / (inner(ah, (a / d), axis=axis) + 1.0)



def solvemdbi_ism(ah, rho, b, axisM, axisK):
    r"""
    Solve a multiple diagonal block linear system with a scaled
    identity term by iterated application of the Sherman-Morrison
    equation. The computation is performed in a way that avoids
    explictly constructing the inverse operator, leading to an
    :math:`O(K^2)` time cost.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

    .. math::
      (\rho I + \mathbf{a}_0 \mathbf{a}_0^H + \mathbf{a}_1 \mathbf{a}_1^H +
       \; \ldots \; + \mathbf{a}_{K-1} \mathbf{a}_{K-1}^H) \; \mathbf{x} =
       \mathbf{b}

    where each :math:`\mathbf{a}_k` is an :math:`M`-vector.
    The sums, inner products, and matrix products in this equation are taken
    along the M and K axes of the corresponding multi-dimensional arrays;
    the solutions are independent over the other axes.

    Parameters
    ----------
    ah : array_like
      Linear system component :math:`\mathbf{a}^H`
    rho : float
      Linear system parameter :math:`\rho`
    b : array_like
      Linear system component :math:`\mathbf{b}`
    axisM : int
      Axis in input corresponding to index m in linear system
    axisK : int
      Axis in input corresponding to index k in linear system

    Returns
    -------
    x : ndarray
      Linear system solution :math:`\mathbf{x}`
    """

    if axisM < 0:
        axisM += ah.ndim
    if axisK < 0:
        axisK += ah.ndim

    K = ah.shape[axisK]
    a = np.conj(ah)
    gamma = np.zeros(a.shape, a.dtype)
    dltshp = list(a.shape)
    dltshp[axisM] = 1
    delta = np.zeros(dltshp, a.dtype)
    slcnc = (slice(None),) * axisK
    alpha = np.take(a, [0], axisK) / rho
    beta = b / rho

    del b
    for k in range(0, K):

        slck = slcnc + (slice(k, k + 1),)
        gamma[slck] = alpha
        delta[slck] = 1.0 + inner(ah[slck], gamma[slck], axis=axisM)

        d = gamma[slck] * inner(ah[slck], beta, axis=axisM)
        beta[:] -= d / delta[slck]

        if k < K - 1:
            alpha[:] = np.take(a, [k + 1], axisK) / rho
            for l in range(0, k + 1):
                slcl = slcnc + (slice(l, l + 1),)
                d = gamma[slcl] * inner(ah[slcl], alpha, axis=axisM)
                alpha[:] -= d / delta[slcl]

    return beta



def solvemdbi_rsm(ah, rho, b, axisK, dimN=2):
    r"""
    Solve a multiple diagonal block linear system with a scaled
    identity term by repeated application of the Sherman-Morrison
    equation. The computation is performed by explictly constructing
    the inverse operator, leading to an :math:`O(K)` time cost and
    :math:`O(M^2)` memory cost, where :math:`M` is the dimension of
    the axis over which inner products are taken.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

    .. math::
      (\rho I + \mathbf{a}_0 \mathbf{a}_0^H + \mathbf{a}_1 \mathbf{a}_1^H +
       \; \ldots \; + \mathbf{a}_{K-1} \mathbf{a}_{K-1}^H) \; \mathbf{x} =
       \mathbf{b}

    where each :math:`\mathbf{a}_k` is an :math:`M`-vector.
    The sums, inner products, and matrix products in this equation are taken
    along the M and K axes of the corresponding multi-dimensional arrays;
    the solutions are independent over the other axes.

    Parameters
    ----------
    ah : array_like
      Linear system component :math:`\mathbf{a}^H`
    rho : float
      Linear system parameter :math:`\rho`
    b : array_like
      Linear system component :math:`\mathbf{b}`
    axisK : int
      Axis in input corresponding to index k in linear system
    dimN : int, optional (default 2)
      Number of spatial dimensions arranged as leading axes in input array.
      Axis M is taken to be at dimN+2.

    Returns
    -------
    x : ndarray
      Linear system solution :math:`\mathbf{x}`
    """

    axisM = dimN + 2
    slcnc = (slice(None),) * axisK
    M = ah.shape[axisM]
    K = ah.shape[axisK]
    a = np.conj(ah)
    Ainv = np.ones(ah.shape[0:dimN] + (1,)*4) * \
        np.reshape(np.eye(M, M) / rho, (1,)*(dimN + 2) + (M, M))

    for k in range(0, K):
        slck = slcnc + (slice(k, k + 1),) + (slice(None), np.newaxis,)
        Aia = inner(Ainv, np.swapaxes(a[slck], dimN + 2, dimN + 3),
                    axis=dimN + 3)
        ahAia = 1.0 + inner(ah[slck], Aia, axis=dimN + 2)
        ahAi = inner(ah[slck], Ainv, axis=dimN + 2)
        AiaahAi = Aia * ahAi
        Ainv = Ainv - AiaahAi / ahAia

    return np.sum(Ainv * np.swapaxes(b[(slice(None),) * b.ndim +
                                       (np.newaxis,)], dimN + 2, dimN + 3),
                  dimN + 3)



# Deal with introduction of new atol parameter for scipy.sparse.linalg.cg
# in SciPy 1.1.0
_spv = scipy.__version__.split('.')
if int(_spv[0]) > 1 or (int(_spv[0]) == 1 and int(_spv[1]) >= 1):
    def _cg_wrapper(A, b, x0=None, tol=1e-5, maxiter=None):
        return cg(A, b, x0=x0, tol=tol, maxiter=maxiter, atol=0.0)
else:
    def _cg_wrapper(A, b, x0=None, tol=1e-5, maxiter=None):
        return cg(A, b, x0=x0, tol=tol, maxiter=maxiter)



def solvemdbi_cg(ah, rho, b, axisM, axisK, tol=1e-5, mit=1000, isn=None):
    r"""
    Solve a multiple diagonal block linear system with a scaled
    identity term using Conjugate Gradient (CG) via
    :func:`scipy.sparse.linalg.cg`.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

     .. math::
      (\rho I + \mathbf{a}_0 \mathbf{a}_0^H + \mathbf{a}_1 \mathbf{a}_1^H +
       \; \ldots \; + \mathbf{a}_{K-1} \mathbf{a}_{K-1}^H) \; \mathbf{x} =
       \mathbf{b}

    where each :math:`\mathbf{a}_k` is an :math:`M`-vector.
    The inner products and matrix products in this equation are taken
    along the M and K axes of the corresponding multi-dimensional arrays;
    the solutions are independent over the other axes.

    Parameters
    ----------
    ah : array_like
      Linear system component :math:`\mathbf{a}^H`
    rho : float
      Parameter rho
    b : array_like
      Linear system component :math:`\mathbf{b}`
    axisM : int
      Axis in input corresponding to index m in linear system
    axisK : int
      Axis in input corresponding to index k in linear system
    tol : float
      CG tolerance
    mit : int
      CG maximum iterations
    isn : array_like
      CG initial solution

    Returns
    -------
    x : ndarray
      Linear system solution :math:`\mathbf{x}`
    cgit : int
      Number of CG iterations
    """

    a = np.conj(ah)
    if isn is not None:
        isn = isn.ravel()
    Aop = lambda x: inner(ah, x, axis=axisM)
    AHop = lambda x: inner(a, x, axis=axisK)
    AHAop = lambda x: AHop(Aop(x))
    vAHAoprI = lambda x: AHAop(x.reshape(b.shape)).ravel() + rho * x.ravel()
    lop = LinearOperator((b.size, b.size), matvec=vAHAoprI, dtype=b.dtype)
    vx, cgit = _cg_wrapper(lop, b.ravel(), isn, tol, mit)
    return vx.reshape(b.shape), cgit



def lu_factor(A, rho, check_finite=True):
    r"""
    Compute LU factorisation of either :math:`A^T A + \rho I` or
    :math:`A A^T + \rho I`, depending on which matrix is smaller.

    Parameters
    ----------
    A : array_like
      Array :math:`A`
    rho : float
      Scalar :math:`\rho`
    check_finite : bool, optional (default False)
      Flag indicating whether the input array should be checked for Inf
      and NaN values

    Returns
    -------
    lu : ndarray
      Matrix containing U in its upper triangle, and L in its lower triangle,
      as returned by :func:`scipy.linalg.lu_factor`
    piv : ndarray
      Pivot indices representing the permutation matrix P, as returned by
      :func:`scipy.linalg.lu_factor`
    """

    N, M = A.shape
    # If N < M it is cheaper to factorise A*A^T + rho*I and then use the
    # matrix inversion lemma to compute the inverse of A^T*A + rho*I
    if N >= M:
        lu, piv = linalg.lu_factor(A.T.dot(A) +
                                   rho * np.identity(M, dtype=A.dtype),
                                   check_finite=check_finite)
    else:
        lu, piv = linalg.lu_factor(A.dot(A.T) +
                                   rho * np.identity(N, dtype=A.dtype),
                                   check_finite=check_finite)
    return lu, piv



def lu_solve_ATAI(A, rho, b, lu, piv, check_finite=True):
    r"""
    Solve the linear system :math:`(A^T A + \rho I)\mathbf{x} = \mathbf{b}`
    or :math:`(A^T A + \rho I)X = B` using :func:`scipy.linalg.lu_solve`.

    Parameters
    ----------
    A : array_like
      Matrix :math:`A`
    rho : float
      Scalar :math:`\rho`
    b : array_like
      Vector :math:`\mathbf{b}` or matrix :math:`B`
    lu : array_like
      Matrix containing U in its upper triangle, and L in its lower triangle,
      as returned by :func:`scipy.linalg.lu_factor`
    piv : array_like
      Pivot indices representing the permutation matrix P, as returned by
      :func:`scipy.linalg.lu_factor`
    check_finite : bool, optional (default False)
      Flag indicating whether the input array should be checked for Inf
      and NaN values

    Returns
    -------
    x : ndarray
      Solution to the linear system
    """

    N, M = A.shape
    if N >= M:
        x = linalg.lu_solve((lu, piv), b, check_finite=check_finite)
    else:
        x = (b - A.T.dot(linalg.lu_solve((lu, piv), A.dot(b), 1,
                                         check_finite=check_finite))) / rho
    return x



def lu_solve_AATI(A, rho, b, lu, piv, check_finite=True):
    r"""
    Solve the linear system :math:`(A A^T + \rho I)\mathbf{x} = \mathbf{b}`
    or :math:`(A A^T + \rho I)X = B` using :func:`scipy.linalg.lu_solve`.

    Parameters
    ----------
    A : array_like
      Matrix :math:`A`
    rho : float
      Scalar :math:`\rho`
    b : array_like
      Vector :math:`\mathbf{b}` or matrix :math:`B`
    lu : array_like
      Matrix containing U in its upper triangle, and L in its lower triangle,
      as returned by :func:`scipy.linalg.lu_factor`
    piv : array_like
      Pivot indices representing the permutation matrix P, as returned by
      :func:`scipy.linalg.lu_factor`
    check_finite : bool, optional (default False)
      Flag indicating whether the input array should be checked for Inf
      and NaN values

    Returns
    -------
    x : ndarray
      Solution to the linear system
    """

    N, M = A.shape
    if N >= M:
        x = (b - linalg.lu_solve((lu, piv), b.dot(A).T,
                                 check_finite=check_finite).T.dot(A.T)) / rho
    else:
        x = linalg.lu_solve((lu, piv), b.T, check_finite=check_finite).T
    return x



def cho_factor(A, rho, lower=False, check_finite=True):
    r"""
    Compute Cholesky factorisation of either :math:`A^T A + \rho I` or
    :math:`A A^T + \rho I`, depending on which matrix is smaller.

    Parameters
    ----------
    A : array_like
      Array :math:`A`
    rho : float
      Scalar :math:`\rho`
    lower : bool, optional (default False)
      Flag indicating whether lower or upper triangular factors are
      computed
    check_finite : bool, optional (default False)
      Flag indicating whether the input array should be checked for Inf
      and NaN values

    Returns
    -------
    c : ndarray
      Matrix containing lower or upper triangular Cholesky factor,
      as returned by :func:`scipy.linalg.cho_factor`
    lwr : bool
      Flag indicating whether the factor is lower or upper triangular
    """

    N, M = A.shape
    # If N < M it is cheaper to factorise A*A^T + rho*I and then use the
    # matrix inversion lemma to compute the inverse of A^T*A + rho*I
    if N >= M:
        c, lwr = linalg.cho_factor(
            A.T.dot(A) + rho * np.identity(M, dtype=A.dtype), lower=lower,
            check_finite=check_finite)
    else:
        c, lwr = linalg.cho_factor(
            A.dot(A.T) + rho * np.identity(N, dtype=A.dtype), lower=lower,
            check_finite=check_finite)
    return c, lwr



def cho_solve_ATAI(A, rho, b, c, lwr, check_finite=True):
    r"""
    Solve the linear system :math:`(A^T A + \rho I)\mathbf{x} = \mathbf{b}`
    or :math:`(A^T A + \rho I)X = B` using :func:`scipy.linalg.cho_solve`.

    Parameters
    ----------
    A : array_like
      Matrix :math:`A`
    rho : float
      Scalar :math:`\rho`
    b : array_like
      Vector :math:`\mathbf{b}` or matrix :math:`B`
    c : array_like
      Matrix containing lower or upper triangular Cholesky factor,
      as returned by :func:`scipy.linalg.cho_factor`
    lwr : bool
      Flag indicating whether the factor is lower or upper triangular

    Returns
    -------
    x : ndarray
      Solution to the linear system
    """

    N, M = A.shape
    if N >= M:
        x = linalg.cho_solve((c, lwr), b, check_finite=check_finite)
    else:
        x = (b - A.T.dot(linalg.cho_solve((c, lwr), A.dot(b),
                                          check_finite=check_finite))) / rho
    return x



def cho_solve_AATI(A, rho, b, c, lwr, check_finite=True):
    r"""
    Solve the linear system :math:`(A A^T + \rho I)\mathbf{x} = \mathbf{b}`
    or :math:`(A A^T + \rho I)X = B` using :func:`scipy.linalg.cho_solve`.

    Parameters
    ----------
    A : array_like
      Matrix :math:`A`
    rho : float
      Scalar :math:`\rho`
    b : array_like
      Vector :math:`\mathbf{b}` or matrix :math:`B`
    c : array_like
      Matrix containing lower or upper triangular Cholesky factor,
      as returned by :func:`scipy.linalg.cho_factor`
    lwr : bool
      Flag indicating whether the factor is lower or upper triangular

    Returns
    -------
    x : ndarray
      Solution to the linear system
    """

    N, M = A.shape
    if N >= M:
        x = (b - linalg.cho_solve((c, lwr), b.dot(A).T,
            check_finite=check_finite).T.dot(A.T)) / rho
    else:
        x = linalg.cho_solve((c, lwr), b.T, check_finite=check_finite).T
    return x



def zpad(x, pd, ax):
    """
    Zero-pad array `x` with `pd = (leading, trailing)` zeros on axis `ax`.

    Parameters
    ----------
    x : array_like
      Array to be padded
    pd : tuple
      Sequence of two ints (leading,trailing) specifying number of zeros
      for padding
    ax : int
      Axis to be padded

    Returns
    -------
    xp : array_like
      Padded array
    """

    xpd = ((0, 0),)*ax + (pd,) + ((0, 0),)*(x.ndim-ax-1)
    return np.pad(x, xpd, 'constant')



def Gax(x, ax):
    """
    Compute gradient of `x` along axis `ax`.

    Parameters
    ----------
    x : array_like
      Input array
    ax : int
      Axis on which gradient is to be computed

    Returns
    -------
    xg : ndarray
      Output array
    """

    slc = (slice(None),)*ax + (slice(-1, None),)
    xg = np.roll(x, -1, axis=ax) - x
    xg[slc] = 0.0
    return xg



def GTax(x, ax):
    """
    Compute transpose of gradient of `x` along axis `ax`.

    Parameters
    ----------
    x : array_like
      Input array
    ax : int
      Axis on which gradient transpose is to be computed

    Returns
    -------
    xg : ndarray
      Output array
    """

    slc0 = (slice(None),) * ax
    xg = np.roll(x, 1, axis=ax) - x
    xg[slc0 + (slice(0, 1),)] = -x[slc0 + (slice(0, 1),)]
    xg[slc0 + (slice(-1, None),)] = x[slc0 + (slice(-2, -1),)]
    return xg



def GradientFilters(ndim, axes, axshp, dtype=None):
    r"""
    Construct a set of filters for computing gradients in the frequency
    domain.

    Parameters
    ----------
    ndim : integer
      Total number of dimensions in array in which gradients are to be
      computed
    axes : tuple of integers
      Axes on which gradients are to be computed
    axshp : tuple of integers
      Shape of axes on which gradients are to be computed
    dtype : dtype
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
    Gf = rfftn(g, axshp, axes=axes)
    GHGf = np.sum(np.conj(Gf) * Gf, axis=-1).real
    return Gf, GHGf



def zdivide(x, y):
    """
    Return `x`/`y`, with 0 instead of NaN where `y` is 0.

    Parameters
    ----------
    x : array_like
      Numerator
    y : array_like
      Denominator

    Returns
    -------
    z : ndarray
      Quotient `x`/`y`
    """

    # See https://stackoverflow.com/a/37977222
    return np.divide(x, y, out=np.zeros_like(x), where=(y != 0))



def shrink1(x, alpha):
    r"""
    Scalar shrinkage/soft thresholding function

     .. math::
      \mathcal{S}_{1,\alpha}(\mathbf{x}) = \mathrm{sign}(\mathbf{x}) \odot
      \max(0, |\mathbf{x}| - \alpha) = \mathrm{prox}_f(\mathbf{x}) \;\;
      \text{where} \;\; f(\mathbf{u}) = \alpha \|\mathbf{u}\|_1 \;\;.

    Parameters
    ----------
    x : array_like
      Input array :math:`\mathbf{x}`
    alpha : float or array_like
      Shrinkage parameter :math:`\alpha`

    Returns
    -------
    x : ndarray
      Output array
    """

    if have_numexpr:
        return ne.evaluate(
            'where(abs(x)-alpha > 0, where(x >= 0, 1, -1) * (abs(x)-alpha), 0)'
        )
    else:
        return np.sign(x) * (np.clip(np.abs(x) - alpha, 0, float('Inf')))



def shrink2(x, alpha, axis=-1):
    r"""
    Vector shrinkage/soft thresholding function

     .. math::
      \mathcal{S}_{2,\alpha}(\mathbf{x}) =
      \frac{\mathbf{x}}{\|\mathbf{x}\|_2} \max(0, \|\mathbf{x}\|_2 - \alpha)
      = \mathrm{prox}_f(\mathbf{x}) \;\;
      \text{where} \;\; f(\mathbf{u}) = \alpha \|\mathbf{u}\|_2 \;\;.

    The :math:`\ell_2` norm is applied over the specified axis of a
    multi-dimensional input (the last axis by default).

    Parameters
    ----------
    x : array_like
      Input array :math:`\mathbf{x}`
    alpha : float or array_like
      Shrinkage parameter :math:`\alpha`
    axis : int, optional (default -1)
      Axis of x over which the :math:`\ell_2` norm

    Returns
    -------
    x : ndarray
      Output array
    """

    a = np.sqrt(np.sum(x**2, axis=axis, keepdims=True))
    b = np.maximum(0, a - alpha)
    b = zdivide(b, a)
    return np.asarray(b * x, dtype=x.dtype)



def shrink12(x, alpha, beta, axis=-1):
    r"""
    Compound shrinkage/soft thresholding function
    :cite:`wohlberg-2012-local` :cite:`chartrand-2013-nonconvex`

     .. math::
      \mathcal{S}_{1,2,\alpha,\beta}(\mathbf{x}) =
      \mathcal{S}_{2,\beta}(\mathcal{S}_{1,\alpha}(\mathbf{x}))
      = \mathrm{prox}_f(\mathbf{x}) \;\;
      \text{where} \;\; f(\mathbf{u}) = \alpha \|\mathbf{u}\|_1 +
      \beta \|\mathbf{u}\|_2 \;\;.

    The :math:`\ell_2` norm is applied over the specified axis of a
    multi-dimensional input (the last axis by default).

    Parameters
    ----------
    x : array_like
      Input array :math:`\mathbf{x}`
    alpha : float or array_like
      Shrinkage parameter :math:`\alpha`
    beta : float or array_like
      Shrinkage parameter :math:`\beta`
    axis : int, optional (default -1)
      Axis of x over which the :math:`\ell_2` norm

    Returns
    -------
    x : ndarray
      Output array
    """

    return shrink2(shrink1(x, alpha), beta, axis)



def proj_l2ball(b, s, r, axes=None):
    r"""
    Project :math:`\mathbf{b}` into the :math:`\ell_2` ball of radius
    :math:`r` about :math:`\mathbf{s}`, i.e.
    :math:`\{ \mathbf{x} : \|\mathbf{x} - \mathbf{s} \|_2 \leq r \}`.

    Parameters
    ----------
    b : array_like
      Vector :math:`\mathbf{b}` to be projected
    s : array_like
      Centre of :math:`\ell_2` ball :math:`\mathbf{s}`
    r : float
      Radius of ball
    axes : sequence of ints, optional (default all axes)
      Axes over which to compute :math:`\ell_2` norms

    Returns
    -------
    x : ndarray
      Projection of :math:`\mathbf{b}` into ball
    """

    d = np.sqrt(np.sum((b - s)**2, axis=axes, keepdims=True))
    p = zdivide(b - s, d)
    return np.asarray((d <= r) * b + (d > r) * (s + r*p), b.dtype)



def promote16(u, fn=None, *args, **kwargs):
    r"""
    Utility function for use with functions that do not support arrays
    of dtype ``np.float16``. This function has two distinct modes of
    operation. If called with only the `u` parameter specified, the
    returned value is either `u` itself if `u` is not of dtype
    ``np.float16``, or `u` promoted to ``np.float32`` dtype if it is. If
    the function parameter `fn` is specified then `u` is conditionally
    promoted as described above, passed as the first argument to
    function `fn`, and the returned values are converted back to dtype
    ``np.float16`` if `u` is of that dtype. Note that if parameter `fn`
    is specified, it may not be be specified as a keyword argument if it
    is followed by any non-keyword arguments.

    Parameters
    ----------
    u : array_like
      Array to be promoted to np.float32 if it is of dtype ``np.float16``
    fn : function or None, optional (default None)
      Function to be called with promoted `u` as first parameter and
      \*args and \*\*kwargs as additional parameters
    *args
      Variable length list of arguments for function `fn`
    **kwargs
      Keyword arguments for function `fn`

    Returns
    -------
    up : ndarray
      Conditionally dtype-promoted version of `u` if `fn` is None,
      or value(s) returned by `fn`, converted to the same dtype as `u`,
      if `fn` is a function
    """

    dtype = np.float32 if u.dtype == np.float16 else u.dtype
    up = np.asarray(u, dtype=dtype)
    if fn is None:
        return up
    else:
        v = fn(up, *args, **kwargs)
        if isinstance(v, tuple):
            vp = tuple([np.asarray(vk, dtype=u.dtype) for vk in v])
        else:
            vp = np.asarray(v, dtype=u.dtype)
        return vp



def atleast_nd(n, u):
    """
    If the input array has fewer than n dimensions, append singleton
    dimensions so that it is n dimensional. Note that the interface
    differs substantially from that of :func:`numpy.atleast_3d` etc.

    Parameters
    ----------
    n : int
      Minimum number of required dimensions
    u : array_like
      Input array

    Returns
    -------
    v : ndarray
      Output array with at least n dimensions
    """

    if u.ndim >= n:
        return u
    else:
        return u.reshape(u.shape + (1,)*(n-u.ndim))



def split(u, axis=0):
    """
    Split an array into a list of arrays on the specified axis. The length
    of the list is the shape of the array on the specified axis, and the
    corresponding axis is removed from each entry in the list. This function
    does not have the same behaviour as :func:`numpy.split`.

    Parameters
    ----------
    u : array_like
      Input array
    axis : int, optional (default 0)
      Axis on which to split the input array

    Returns
    -------
    v : list of ndarray
      List of arrays
    """

    # Convert negative axis to positive
    if axis < 0:
        axis = u.ndim + axis

    # Construct axis selection slice
    slct0 = (slice(None),) * axis
    return [u[slct0 + (k,)] for k in range(u.shape[axis])]



def blockcirculant(A):
    """
    Construct a block circulant matrix from a tuple of arrays. This is a
    block-matrix variant of :func:`scipy.linalg.circulant`.

    Parameters
    ----------
    A : tuple of array_like
      Tuple of arrays corresponding to the first block column of the output
      block matrix

    Returns
    -------
    B : ndarray
      Output array
    """

    r, c = A[0].shape
    B = np.zeros((len(A) * r, len(A) * c), dtype=A[0].dtype)
    for k in range(len(A)):
        for l in range(len(A)):
            kl = np.mod(k + l, len(A))
            B[r*kl:r*(kl + 1), c*k:c*(k + 1)] = A[l]
    return B



def fl2norm2(xf, axis=(0, 1)):
    r"""
    Compute the squared :math:`\ell_2` norm in the DFT domain, taking
    into account the unnormalised DFT scaling, i.e. given the DFT of a
    multi-dimensional array computed via :func:`fftn`, return the
    squared :math:`\ell_2` norm of the original array.

    Parameters
    ----------
    xf : array_like
      Input array
    axis : sequence of ints, optional (default (0,1))
      Axes on which the input is in the frequency domain

    Returns
    -------
    x : float
      :math:`\|\mathbf{x}\|_2^2` where the input array is the result of
      applying :func:`fftn` to the specified axes of multi-dimensional
      array :math:`\mathbf{x}`
    """

    xfs = xf.shape
    return (np.linalg.norm(xf)**2) / np.prod(np.array([xfs[k] for k in axis]))



def rfl2norm2(xf, xs, axis=(0, 1)):
    r"""
    Compute the squared :math:`\ell_2` norm in the DFT domain, taking
    into account the unnormalised DFT scaling, i.e. given the DFT of a
    multi-dimensional array computed via :func:`rfftn`, return the
    squared :math:`\ell_2` norm of the original array.

    Parameters
    ----------
    xf : array_like
      Input array
    xs : sequence of ints
      Shape of original array to which :func:`rfftn` was applied to
      obtain the input array
    axis : sequence of ints, optional (default (0,1))
      Axes on which the input is in the frequency domain

    Returns
    -------
    x : float
      :math:`\|\mathbf{x}\|_2^2` where the input array is the result of
      applying :func:`rfftn` to the specified axes of multi-dimensional
      array :math:`\mathbf{x}`
    """

    scl = 1.0 / np.prod(np.array([xs[k] for k in axis]))
    slc0 = (slice(None),) * axis[-1]
    nrm0 = np.linalg.norm(xf[slc0 + (0,)])
    idx1 = (xs[axis[-1]] + 1) // 2
    nrm1 = np.linalg.norm(xf[slc0 + (slice(1, idx1),)])
    if xs[axis[-1]] % 2 == 0:
        nrm2 = np.linalg.norm(xf[slc0 + (slice(-1, None),)])
    else:
        nrm2 = 0.0
    return scl*(nrm0**2 + 2.0*nrm1**2 + nrm2**2)



def rrs(ax, b):
    r"""
    Compute relative residual :math:`\|\mathbf{b} - A \mathbf{x}\|_2 /
    \|\mathbf{b}\|_2` of the solution to a linear equation :math:`A \mathbf{x}
    = \mathbf{b}`. Returns 1.0 if :math:`\mathbf{b} = 0`.

    Parameters
    ----------
    ax : array_like
      Linear component :math:`A \mathbf{x}` of equation
    b : array_like
      Constant component :math:`\mathbf{b}` of equation

    Returns
    -------
    x : float
      Relative residual
    """

    nrm = np.linalg.norm(b.ravel())
    if nrm == 0.0:
        return 1.0
    else:
        return np.linalg.norm((ax - b).ravel()) / nrm
