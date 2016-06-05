#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Linear algebra functions"""

from __future__ import division
from builtins import range

import numpy as np
from scipy import linalg
from scipy import fftpack
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
import multiprocessing
import pyfftw
try:
    import numexpr as ne
except ImportError:
    have_numexpr = False
else:
    have_numexpr = True

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(300)



def complex_dtype(dtype):
    """Construct the corresponding complex dtype for a given real dtype,
    e.g. the complex dtype corresponding to np.float32 is np.complex64.

    Parameters
    ----------
    dtype : dtype
      A real dtype, e.g. np.float32, np.float64

    Returns
    -------
    cdtype : dtype
      The complex dtype corresponding to the input dtype
    """

    return (np.zeros(1, dtype)+1j).dtype



def pyfftw_empty_aligned(shape, dtype, order='C', n=None):
    """Construct an empty byte-aligned array for efficient use by
    pyfftw. This function is a wrapper for :func:`pyfftw.empty_aligned`

    Parameters
    ----------
    shape : sequence of ints
      Output array shape
    dtype : dtype
      Output array dtype
    n : int, optional
      Output array should be aligned to n-byte boundary

    Returns
    -------
    a :  ndarray
      Empty array with required byte-alignment
    """

    return pyfftw.empty_aligned(shape, dtype, order, n)



def fftn(a, s=None, axes=None):
    """
    Compute the multi-dimensional discrete Fourier transform. This function
    is a wrapper for :func:`pyfftw.interfaces.numpy_fft.fftn`,
    with an interface similar to that of :func:`numpy.fft.fftn`.


    Parameters
    ----------
    a : array_like
      Input array (can be complex)
    s : sequence of ints, optional
      Shape of the output along each axis (input is cropped or zero-padded
      to match).
    axes : sequence of ints, optional
      Axes over which to compute the DFT.

    Returns
    -------
    af : complex ndarray
      DFT of input array
    """

    return pyfftw.interfaces.numpy_fft.fftn(a, s=s, axes=axes, 
                    overwrite_input=False, planner_effort='FFTW_MEASURE',
                    threads=multiprocessing.cpu_count())



def ifftn(a, s=None, axes=None):
    """
    Compute the multi-dimensional inverse discrete Fourier transform.
    This function is a wrapper for :func:`pyfftw.interfaces.numpy_fft.ifftn`,
    with an interface similar to that of :func:`numpy.fft.ifftn`.


    Parameters
    ----------
    a : array_like
      Input array (can be complex)
    s : sequence of ints, optional
      Shape of the output along each axis (input is cropped or zero-padded
      to match).
    axes : sequence of ints, optional
      Axes over which to compute the inverse DFT.

    Returns
    -------
    af : complex ndarray
      Inverse DFT of input array
    """

    return pyfftw.interfaces.numpy_fft.ifftn(a, s=s, axes=axes,
                    overwrite_input=False, planner_effort='FFTW_MEASURE',
                    threads=multiprocessing.cpu_count())



def rfftn(a, s=None, axes=None):
    """
    Compute the multi-dimensional discrete Fourier transform for real input.
    This function is a wrapper for :func:`pyfftw.interfaces.numpy_fft.rfftn`,
    with an interface similar to that of :func:`numpy.fft.rfftn`.


    Parameters
    ----------
    a : array_like
      Input array (taken to be real)
    s : sequence of ints, optional
      Shape of the output along each axis (input is cropped or zero-padded
      to match).
    axes : sequence of ints, optional
      Axes over which to compute the DFT.

    Returns
    -------
    af : complex ndarray
      DFT of input array
    """

    return pyfftw.interfaces.numpy_fft.rfftn(a, s=s, axes=axes,
                    overwrite_input=False, planner_effort='FFTW_MEASURE',
                    threads=multiprocessing.cpu_count())



def irfftn(a, s=None, axes=None):
    """
    Compute the inverse of the multi-dimensional discrete Fourier transform
    for real input. This function is a wrapper for
    :func:`pyfftw.interfaces.numpy_fft.irfftn`, with an interface similar to
    that of :func:`numpy.fft.irfftn`.


    Parameters
    ----------
    a : array_like
      Input array
    s : sequence of ints, optional
      Shape of the output along each axis (input is cropped or zero-padded
      to match).
    axes : sequence of ints, optional
      Axes over which to compute the inverse DFT.

    Returns
    -------
    af : ndarray
      Inverse DFT of input array
    """

    return pyfftw.interfaces.numpy_fft.irfftn(a, s=s, axes=axes,
                    overwrite_input=False, planner_effort='FFTW_MEASURE',
                    threads=multiprocessing.cpu_count())



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
    axes : sequence of ints, optional
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
    axes : sequence of ints, optional
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



def solvedbi_sm(ah, rho, b, c=None, axis=4):
    """
    Solve a diagonal block linear system with a scaled identity term
    using the Sherman-Morrison equation.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

    .. math::
      (\\rho I + \mathbf{a} \mathbf{a}^H ) \; \mathbf{x} = \mathbf{b}

    In this equation inner products and matrix products are taken along
    the specified axis of the corresponding multi-dimensional arrays; the
    solutions are independent over the other axes.


    Parameters
    ----------
    ah : array_like
      Linear system component :math:`\mathbf{a}^H`
    rho : float
      Linear system parameter :math:`\\rho`
    b : array_like
      Linear system component :math:`\mathbf{b}`
    c : array_like, optional
      Solution component :math:`\mathbf{c}` that may be pre-computed using
      :func:`solvedbi_sm_c` and cached for re-use.
    axis : int, optional
      Axis along which to solve the linear system

    Returns
    -------
    x : array_like
      Linear system solution :math:`\mathbf{x}`

    """

    a = np.conj(ah)
    if c is None:
        c = solvedbi_sm_c(ah, a, rho, axis)
    if have_numexpr:
        cb = np.sum(c * b, axis=axis, keepdims=True)
        return ne.evaluate('(b - (a * cb)) / rho')
    else:
        return (b - (a * np.sum(c * b, axis=axis, keepdims=True))) / rho




def solvedbi_sm_c(ah, a, rho, axis=4):
    """Compute cached component used by :func:`solvedbi_sm`.


    Parameters
    ----------
    ah : array_like
      Linear system component :math:`\mathbf{a}^H`
    a : array_like
      Linear system component :math:`\mathbf{a}`
    rho : float
      Linear system parameter :math:`\\rho`
    axis : int, optional
      Axis along which to solve the linear system

    Returns
    -------
    c : array_like
      Argument :math:`\mathbf{c}` used by :func:`solvedbi_sm`
    """

    return ah / (np.sum(ah * a, axis=axis, keepdims=True) + rho)



def solvemdbi_ism(ah, rho, b, axisM, axisK):
    """
    Solve a multiple diagonal block linear system with a scaled
    identity term by iterated application of the Sherman-Morrison
    equation. The computation is performed in a way that avoids
    explictly constructing the inverse operator, leading to an
    :math:`O(K^2)` time cost.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

    .. math::
      (\\rho I + \mathbf{a}_0 \mathbf{a}_0^H + \mathbf{a}_1 \mathbf{a}_1^H +
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
      Linear system parameter :math:`\\rho`
    b : array_like
      Linear system component :math:`\mathbf{b}`
    axisM : int
      Axis in input corresponding to index m in linear system
    axisK : int
      Axis in input corresponding to index k in linear system

    Returns
    -------
    x : array_like
      Linear system solution :math:`\mathbf{x}`
    """

    K = ah.shape[axisK]
    a = np.conj(ah)
    gamma = np.zeros(a.shape, a.dtype)
    delta = np.zeros(a.shape[0:axisM] + (1,), a.dtype)
    slcnc = (slice(None),)*axisK
    alpha = a[slcnc + (slice(0, 1),)] / rho
    beta = b / rho

    del b
    for k in range(0, K):

        slck = slcnc + (slice(k, k+1),)
        gamma[slck] = alpha
        delta[slck] = 1.0 + np.sum(ah[slck] * gamma[slck], axisM, keepdims=True)

        c = np.sum(ah[slck] * beta, axisM, keepdims=True)
        d = c * gamma[slck]
        beta = beta - (d / delta[slck])

        if k < K-1:
            alpha = a[slcnc + (slice(k+1, k+2),)] / rho
            for l in range(0, k+1):
                slcl = slcnc + (slice(l, l+1),)
                c = np.sum(ah[slcl] * alpha, axisM, keepdims=True)
                d = c * gamma[slcl]
                alpha = alpha - (d / delta[slcl])

    return beta



def solvemdbi_rsm(ah, rho, b, axisK, dimN=2):
    """
    Solve a multiple diagonal block linear system with a scaled
    identity term by repeated application of the Sherman-Morrison
    equation. The computation is performed by explictly constructing
    the inverse operator, leading to an :math:`O(K)` time cost and
    :math:`O(M^2)` memory cost, where M is the dimension of the axis
    over which inner products are taken.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

    .. math::
      (\\rho I + \mathbf{a}_0 \mathbf{a}_0^H + \mathbf{a}_1 \mathbf{a}_1^H +
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
      Linear system parameter :math:`\\rho`
    b : array_like
      Linear system component :math:`\mathbf{b}`
    axisK : int
      Axis in input corresponding to index k in linear system
    dimN : int, optional
      Number of spatial dimensions arranged as leading axes in input array.
      Axis M is taken to be at dimN+2.

    Returns
    -------
    x : array_like
      Linear system solution :math:`\mathbf{x}`

    """

    axisM = dimN + 2
    slcnc = (slice(None),)*axisK
    M = ah.shape[axisM]
    K = ah.shape[axisK]
    a = np.conj(ah)
    Ainv = np.ones(ah.shape[0:dimN] + (1,)*4) * \
        np.reshape(np.eye(M,M) / rho, (1,)*(dimN+2) + (M, M))

    for k in range(0, K):
        slck = slcnc + (slice(k, k+1),) + (slice(None), np.newaxis,)
        Aia = np.sum(Ainv * np.swapaxes(a[slck], dimN+2, dimN+3),
                     dimN+3, keepdims=True)
        ahAia = 1.0 + np.sum(ah[slck] * Aia, dimN+2, keepdims=True)
        ahAi = np.sum(ah[slck] * Ainv, dimN+2, keepdims=True)
        AiaahAi = Aia * ahAi
        Ainv = Ainv - AiaahAi / ahAia

    return np.sum(Ainv * np.swapaxes(b[(slice(None),)*b.ndim + (np.newaxis,)],
                                        dimN+2, dimN+3), dimN+3)



def solvemdbi_cg(ah, rho, b, axisM, axisK, tol=1e-5, mit=1000, isn=None):
    """
    Solve a multiple diagonal block linear system with a scaled
    identity term using Conjugate Gradient (CG) via
    :func:`scipy.sparse.linalg.cg`.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

     .. math::
      (\\rho I + \mathbf{a}_0 \mathbf{a}_0^H + \mathbf{a}_1 \mathbf{a}_1^H +
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
    x : array_like
      Linear system solution :math:`\mathbf{x}`
    cgit : int
      Number of CG iterations
    """

    a = np.conj(ah)
    if isn is not None:
        isn = isn.ravel()
    Aop = lambda x: np.sum(ah * x, axis=axisM, keepdims=True)
    AHop = lambda x: np.sum(a * x, axis=axisK, keepdims=True)
    AHAop = lambda x: AHop(Aop(x))
    vAHAoprI = lambda x: AHAop(x.reshape(b.shape)).ravel() + rho*x.ravel()
    lop = LinearOperator((b.size, b.size), matvec=vAHAoprI, dtype=b.dtype)
    vx, cgit = cg(lop, b.ravel(), isn, tol, mit)
    return vx.reshape(b.shape), cgit




def zpad(x, pd, ax):
    """
    Zero-pad array x with pd=(leading,trailing) zeros on axis ax.


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

    xpd = ((0,0),)*ax + (pd,) + ((0,0),)*(x.ndim-ax-1)
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
    xg : array_like
      Output array
    """

    slc0 = (slice(None),)*ax
    return zpad(x[slc0 + (slice(1,None),)] - x[slc0 + (slice(-1),)], (0,1), ax)



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
    xg : array_like
      Output array
    """

    slc0 = (slice(None),)*ax
    return zpad(x[slc0 + (slice(-1),)], (1,0), ax) - \
      zpad(x[slc0 + (slice(-1),)], (0,1), ax)




def shrink1(x, alpha):
    """
    Shrinkage/soft thresholding function

     .. math::
      \mathcal{S}_{1,\\alpha}(\mathbf{x}) =
      \mathrm{sign}(\mathbf{x}) \odot \max(0, |\mathbf{x}| - \\alpha)


    Parameters
    ----------
    x : array_like
      Input array :math:`\mathbf{x}`
    alpha : float or array_like
      Shrinkage parameter :math:`\\alpha`

    Returns
    -------
    x : array_like
      Output array
    """

    if have_numexpr:
        return ne.evaluate(
        'where(abs(x)-alpha > 0, where(x >= 0, 1.0, -1.0) * (abs(x)-alpha), 0)'
        )
    else:
        return np.sign(x) * (np.clip(np.abs(x) - alpha, 0.0, float('Inf')))



def zquotient(x, y):
    """
    Return x/y, with 0 instead of NaN where y is 0.


    Parameters
    ----------
    x : array_like
      Numerator
    y : array_like
      Denominator

    Returns
    -------
    z : array_like
      Quotient `x`/`y`
    """

    x = x.copy()
    y = y.copy()
    zm = y == 0.0
    x[zm] = 0.0
    y[zm] = 1.0
    return x / y



def shrink2(x, alpha):
    """Vector shrinkage/soft thresholding function

     .. math::
      \mathcal{S}_{2,\\alpha}(\mathbf{x}) =
      \\frac{\mathbf{x}}{\|\mathbf{x}\|_2} \max(0, \|\mathbf{x}\|_2 - \\alpha)

    The :math:`\ell^2` norm is applied over the last axis of a
    multi-dimensional input.


    Parameters
    ----------
    x : array_like
      Input array :math:`\mathbf{x}`
    alpha : float or array_like
      Shrinkage parameter :math:`\\alpha`

    Returns
    -------
    x : array_like
      Output array

    """

    a = np.sqrt(np.sum(x**2, axis=x.ndim-1, keepdims=True))
    b = np.maximum(0, a - alpha)
    b = zquotient(b, a)
    return b*x



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
    v : array_like
      Output array with at least n dimensions

    """

    if u.ndim >= n:
        return u
    else:
        return u.reshape(u.shape + (1,)*(n-u.ndim))



def fl2norm2(xf, axis=(0,1)):
    """
    Compute the squared :math:`\ell^2` norm in the DFT domain, taking
    into account the unnormalised DFT scaling, i.e. given the DFT of a
    multi-dimensional array computed via :func:`fftn`, return the
    squared :math:`\ell^2` norm of the original array.


    Parameters
    ----------
    xf : array_like
      Input array
    axis : sequence of ints, optional
      Axes on which the input is in the frequency domain

    Returns
    -------
    x : float
      :math:`\|\mathbf{x}\|_2^2` where the input array is the result of
      applying :func:`fftn` to the specified axes of multi-dimensional
      array :math:`\mathbf{x}`
    """

    xfs = xf.shape
    return 0.5*(linalg.norm(xf)**2)/np.prod(np.array([xfs[k] for k in axis]))



def rfl2norm2(xf, xs, axis=(0,1)):
    """
    Compute the squared :math:`\ell^2` norm in the DFT domain, taking
    into account the unnormalised DFT scaling, i.e. given the DFT of a
    multi-dimensional array computed via :func:`rfftn`, return the
    squared :math:`\ell^2` norm of the original array.


    Parameters
    ----------
    xf : array_like
      Input array
    xs : sequence of ints
      Shape of original array to which :func:`rfftn` was applied to
      obtain the input array
    axis : sequence of ints, optional
      Axes on which the input is in the frequency domain

    Returns
    -------
    x : float
      :math:`\|\mathbf{x}\|_2^2` where the input array is the result of
      applying :func:`rfftn` to the specified axes of multi-dimensional
      array :math:`\mathbf{x}`
    """

    scl = 1.0 / np.prod(np.array([xs[k] for k in axis]))
    slc0 = (slice(None),)*axis[-1]
    nrm0 = linalg.norm(xf[slc0 + (0,)])
    idx1 = (xs[axis[-1]]+1)//2
    nrm1 = linalg.norm(xf[slc0 + (slice(1,idx1),)])
    if xs[axis[-1]]%2 == 0:
        nrm2 = linalg.norm(xf[slc0 + (slice(-1,None),)])
    else:
        nrm2 = 0.0
    return scl*(nrm0**2 + 2.0*nrm1**2 + nrm2**2)



def rrs(ax, b):
    """
    Compute relative residual :math:`\|\mathbf{b} - A \mathbf{x}\|_2 /
    \|\mathbf{b}\|_2` of the solution to a linear equation :math:`A \mathbf{x}
    = \mathbf{b}`.


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

    return linalg.norm((ax - b).ravel()) / linalg.norm(b.ravel())




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
      MSE between vref and vcmp
    """

    return np.mean(np.fabs(vref.ravel()-vcmp.ravel())**2)



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
      SNR of vcmp with respect to vref
    """

    r = vref.ravel()
    c = vcmp.ravel()
    mse = np.mean((r - c)**2)
    dv = np.var(r)
    return 10.0*np.log10(dv / mse)



def psnr(vref, vcmp):
    """
    Compute Peak Signal to Noise Ratio (PSNR) of two images. The PSNR
    calculation uses the less common definition in terms of the actual
    range (i.e. max minus min) of the reference signal instead of the
    maximum possible range for the data type (i.e. :math:`2^b-1` for a
    :math:`b` bit representation).


    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image

    Returns
    -------
    x : float
      PSNR of vcmp with respect to vref

    """

    r = vref.ravel()
    c = vcmp.ravel()
    mse = np.mean((r - c)**2)
    dv = (r.max() - r.min())**2
    return 10.0*np.log10(dv / mse)
