# -*- coding: utf-8 -*-
# Copyright (C) 2015-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Variants of the Fast Fourier Transform and associated functions."""

from __future__ import division
from builtins import range

import multiprocessing
import numpy as np
from scipy import fftpack
try:
    import pyfftw
except ImportError:
    have_pyfftw = False
    import warnings
    warnings.warn('Module pyfftw could not be imported. FFT '
                  'computations will be performed using numpy.fft, '
                  'which may be substantially slower', RuntimeWarning)
    import numpy.fft as npfft
else:
    have_pyfftw = True

from sporco._util import renamed_function


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


if have_pyfftw:
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(300)

pyfftw_threads = multiprocessing.cpu_count()
"""Global variable setting the number of threads used in :mod:`pyfftw`
computations"""
pyfftw_planner_effort = 'FFTW_MEASURE'
"""FFTW planning rigor flag used in :mod:`pyfftw` computations"""


@renamed_function(depname='complex_dtype', depmod='sporco.linalg')
def complex_dtype(dtype):
    """Construct the corresponding complex dtype for a given real dtype.

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



def real_dtype(dtype):
    """Construct the corresponding real dtype for a given complex dtype.

    Construct the corresponding real dtype for a given complex dtype,
    e.g. the real dtype corresponding to ``np.complex64`` is
    ``np.float32``.

    Parameters
    ----------
    dtype : dtype
      A complex dtype, e.g. np.complex64, np.complex128

    Returns
    -------
    cdtype : dtype
      The real dtype corresponding to the input dtype
    """

    return np.zeros(1, dtype).real.dtype



@renamed_function(depname='pyfftw_byte_aligned', depmod='sporco.linalg')
def byte_aligned(array, dtype=None, n=None):
    """Construct a byte-aligned array for FFTs.

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



@renamed_function(depname='pyfftw_empty_aligned', depmod='sporco.linalg')
def empty_aligned(shape, dtype, order='C', n=None):
    """Construct an empty byte-aligned array for FFTs.

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



@renamed_function(depname='pyfftw_rfftn_empty_aligned', depmod='sporco.linalg')
def rfftn_empty_aligned(shape, axes, dtype, order='C', n=None):
    """Construct an empty byte-aligned array for real FFTs.

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



@renamed_function(depname='fftn', depmod='sporco.linalg')
def fftn(a, s=None, axes=None):
    """Multi-dimensional discrete Fourier transform.

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
        planner_effort=pyfftw_planner_effort, threads=pyfftw_threads)



@renamed_function(depname='ifftn', depmod='sporco.linalg')
def ifftn(a, s=None, axes=None):
    """Multi-dimensional inverse discrete Fourier transform.

    Compute the multi-dimensional inverse discrete Fourier transform.
    This function is a wrapper for :func:`pyfftw.interfaces.numpy_fft.ifftn`,
    with an interface similar to that of :func:`numpy.fft.ifftn`.

    Parameters
    ----------
    a : array_like
      Input array (can be complex)
    s : sequence of ints, optional (default None)
      Shape of the output along each transformed axis (input is cropped
      or zero-padded to match).
    axes : sequence of ints, optional (default None)
      Axes over which to compute the inverse DFT.

    Returns
    -------
    af : complex ndarray
      Inverse DFT of input array
    """

    return pyfftw.interfaces.numpy_fft.ifftn(
        a, s=s, axes=axes, overwrite_input=False,
        planner_effort=pyfftw_planner_effort, threads=pyfftw_threads)



@renamed_function(depname='rfftn', depmod='sporco.linalg')
def rfftn(a, s=None, axes=None):
    """Multi-dimensional discrete Fourier transform for real input.

    Compute the multi-dimensional discrete Fourier transform for real input.
    This function is a wrapper for :func:`pyfftw.interfaces.numpy_fft.rfftn`,
    with an interface similar to that of :func:`numpy.fft.rfftn`.

    Parameters
    ----------
    a : array_like
      Input array (taken to be real)
    s : sequence of ints, optional (default None)
      Shape of the output along each transformed axis (input is cropped
      or zero-padded to match).
    axes : sequence of ints, optional (default None)
      Axes over which to compute the DFT.

    Returns
    -------
    af : complex ndarray
      DFT of input array
    """

    return pyfftw.interfaces.numpy_fft.rfftn(
        a, s=s, axes=axes, overwrite_input=False,
        planner_effort=pyfftw_planner_effort, threads=pyfftw_threads)



@renamed_function(depname='irfftn', depmod='sporco.linalg')
def irfftn(a, s, axes=None):
    """Multi-dimensional inverse discrete Fourier transform for real input.

    Compute the inverse of the multi-dimensional discrete Fourier
    transform for real input. This function is a wrapper for
    :func:`pyfftw.interfaces.numpy_fft.irfftn`, with an interface similar
    to that of :func:`numpy.fft.irfftn`.

    Parameters
    ----------
    a : array_like
      Input array
    s : sequence of ints
      Shape of the output along each transformed axis (input is cropped
      or zero-padded to match). This parameter is not optional because,
      unlike :func:`ifftn`, the output shape cannot be uniquely
      determined from the input shape.
    axes : sequence of ints, optional (default None)
      Axes over which to compute the inverse DFT.

    Returns
    -------
    af : ndarray
      Inverse DFT of input array
    """

    return pyfftw.interfaces.numpy_fft.irfftn(
        a, s=s, axes=axes, overwrite_input=False,
        planner_effort=pyfftw_planner_effort, threads=pyfftw_threads)



@renamed_function(depname='dctii', depmod='sporco.linalg')
def dctii(x, axes=None):
    """Multi-dimensional DCT-II.

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



@renamed_function(depname='idctii', depmod='sporco.linalg')
def idctii(x, axes=None):
    """Multi-dimensional inverse DCT-II.

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



@renamed_function(depname='fftconv', depmod='sporco.linalg')
def fftconv(a, b, axes=None, origin=None):
    """Multi-dimensional convolution via the Discrete Fourier Transform.

    Compute a multi-dimensional convolution via the Discrete Fourier
    Transform. Note that the output has a phase shift relative to the
    output of :func:`scipy.ndimage.convolve` with the default `origin`
    parameter.

    Parameters
    ----------
    a : array_like
      Input array
    b : array_like
      Input array
    axes : sequence of ints or None optional (default None)
      Axes on which to perform convolution. The default of None
      selects all axes of `a`
    origin : sequence of ints or None optional (default None)
      Indices of centre of `a` filter. The default of None corresponds
      to a centre at 0 on all axes of `a`

    Returns
    -------
    ab : ndarray
      Convolution of input arrays, `a` and `b`, along specified `axes`
    """

    if axes is None:
        axes = tuple(range(a.ndim))
    if np.isrealobj(a) and np.isrealobj(b):
        fft = rfftn
        ifft = irfftn
    else:
        fft = fftn
        ifft = ifftn
    dims = np.maximum([a.shape[i] for i in axes], [b.shape[i] for i in axes])
    af = fft(a, dims, axes)
    bf = fft(b, dims, axes)
    ab = ifft(af * bf, dims, axes)
    if origin is not None:
        ab = np.roll(ab, -np.array(origin), axis=axes)
    return ab



@renamed_function(depname='fl2norm2', depmod='sporco.linalg')
def fl2norm2(xf, axis=(0, 1)):
    r"""Compute the squared :math:`\ell_2` norm in the DFT domain.

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



@renamed_function(depname='rfl2norm2', depmod='sporco.linalg')
def rfl2norm2(xf, xs, axis=(0, 1)):
    r"""Compute the squared :math:`\ell_2` norm in the real DFT domain.

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





if not have_pyfftw:

    __all__ = ['complex_dtype', 'real_dtype', 'byte_aligned', 'empty_aligned',
               'rfftn_empty_aligned', 'fftn', 'ifftn', 'rfftn', 'irfftn',
               'dctii', 'idctii', 'fftconv', 'fl2norm2', 'rfl2norm2']

    def _aligned(array, dtype=None, n=None):
        if dtype is None:
            return array
        else:
            return array.astype(dtype)
    _aligned.__doc__ = byte_aligned.__doc__
    byte_aligned = _aligned

    def _empty(shape, dtype, order='C', n=None):
        return np.empty(shape, dtype=dtype)
    _empty.__doc__ = empty_aligned.__doc__
    empty_aligned = _empty

    def _rfftn_empty(shape, axes, dtype, order='C', n=None):
        ashp = list(shape)
        raxis = axes[-1]
        ashp[raxis] = ashp[raxis] // 2 + 1
        cdtype = complex_dtype(dtype)
        return np.empty(ashp, dtype=cdtype)
    _rfftn_empty.__doc__ = rfftn_empty_aligned.__doc__
    rfftn_empty_aligned = _rfftn_empty

    def _fftn(a, s=None, axes=None):
        return  npfft.fftn(a, s, axes).astype(complex_dtype(a.dtype))
    _fftn.__doc__ = fftn.__doc__
    fftn = _fftn

    def _ifftn(a, s=None, axes=None):
        return  npfft.ifftn(a, s, axes).astype(a.dtype)
    _ifftn.__doc__ = ifftn.__doc__
    ifftn = _ifftn

    def _rfftn(a, s=None, axes=None):
        return  npfft.rfftn(a, s, axes).astype(complex_dtype(a.dtype))
    _rfftn.__doc__ = rfftn.__doc__
    rfftn = _rfftn

    def _irfftn(a, s=None, axes=None):
        return  npfft.irfftn(a, s, axes).astype(real_dtype(a.dtype))
    _irfftn.__doc__ = irfftn.__doc__
    irfftn = _irfftn
