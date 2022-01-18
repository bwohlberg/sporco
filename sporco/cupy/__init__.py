# -*- coding: utf-8 -*-
# Copyright (C) 2018-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Construct variants of solvers and support code that use cupy instead
of numpy"""

from __future__ import absolute_import

import sys
import re
from functools import reduce
try:
    import importlib.util
except ImportError:
    sys.stderr.write('The sporco.cupy subpackage is not supported under '
                     'Python 2.7\n')
    raise

try:
    # Try to import cupy
    import cupy as cp
    import cupyx.scipy.linalg as cpxl
    # Try to access a device
    cp.cuda.Device(0).compute_capability
    # Flag indicating successful import
    have_cupy = True
    # Import appropriate versions of utility functions
    from ._cp_util import *
    try:
        # Try to import GPUtil
        import GPUtil
        # Check whether GPUtil is functional
        gpus = GPUtil.getGPUs()
        if gpus:
            have_gputil = True
        else:
            have_gputil = False
    except ImportError:
        have_gputil = False
    except ValueError:
        have_gputil = False
    if have_gputil:
        from ._gputil import *
    else:
        from ._nogputil import *
except Exception:
    # If cupy import or device access fails, import numpy to the same alias
    import numpy as cp
    # Flag indicating unsuccessful import
    have_cupy = False
    # Import appropriate versions of utility functions
    from ._np_util import *
    # Import appropriate versions of utility functions
    from ._nogputil import *

import numpy as np

# Unlike numpy, cupy has a prod function but no product function
if not hasattr(cp, 'product'):
    cp.product = cp.prod


def cupy_enabled():
    """Return ``True`` if CuPy is installed and a GPU device is available,
    otherwise return ``False``.
    """

    return have_cupy


def rgetattr(obj, name):
    """Recursive version of :func:`getattr`."""

    return reduce(getattr, name.split('.'), obj)


def rsetattr(obj, name, value):
    """Recursive version of :func:`setattr`."""

    # See goo.gl/BVJ7MN
    path = name.split('.')
    setattr(reduce(getattr, path[:-1], obj), path[-1], value)


def load_module(name):
    """Load the named module without registering it in ``sys.modules``.

    Parameters
    ----------
    name : string
      Module name

    Returns
    -------
    mod : module
      Loaded module
    """

    spec = importlib.util.find_spec(name)
    mod = importlib.util.module_from_spec(spec)
    mod.__spec__ = spec
    mod.__loader__ = spec.loader
    spec.loader.exec_module(mod)
    return mod


def patch_module(name, pname, pfile=None, attrib=None):
    """Create a patched copy of the named module and register it in
    ``sys.modules``.

    Parameters
    ----------
    name : string
      Name of source module
    pname : string
      Name of patched copy of module
    pfile : string or None, optional (default None)
      Value to assign as source file name of patched module
    attrib : dict or None, optional (default None)
      Dict of attribute names and values to assign to patched module

    Returns
    -------
    mod : module
      Patched module
    """

    if attrib is None:
        attrib = {}
    spec = importlib.util.find_spec(name)
    spec.name = pname
    if pfile is not None:
        spec.origin = pfile
    spec.loader.name = pname
    mod = importlib.util.module_from_spec(spec)
    mod.__spec__ = spec
    mod.__loader__ = spec.loader
    sys.modules[pname] = mod
    spec.loader.exec_module(mod)
    for k, v in attrib.items():
        setattr(mod, k, v)
    return mod


def sporco_cupy_patch_module(name, attrib=None):
    """Create a copy of the named sporco module, patch it to replace
    numpy with cupy, and register it in ``sys.modules``.

    Parameters
    ----------
    name : string
      Name of source module
    attrib : dict or None, optional (default None)
      Dict of attribute names and values to assign to patched module

    Returns
    -------
    mod : module
      Patched module
    """

    # Patched module name is constructed from source module name
    # by replacing 'sporco.' with 'sporco.cupy.'
    pname = re.sub('^sporco.', 'sporco.cupy.', name)
    # Attribute dict always maps cupy module to 'np' attribute in
    # patched module
    if attrib is None:
        attrib = {}
    attrib.update({'np': cp})
    # Create patched module
    mod = patch_module(name, pname, pfile='patched', attrib=attrib)
    mod.__spec__.has_location = False
    return mod


def _list2array(lst):
    """Convert a list to a numpy array."""

    if lst and isinstance(lst[0], cp.ndarray):
        return cp.hstack(lst)
    else:
        return cp.asarray(lst)


if have_cupy:
    def _zdivide(x, y):
        """Patched version of :func:`sporco.array.zdivide`."""

        div = x / y
        div[cp.logical_or(cp.isnan(div), cp.isinf(div))] = 0
        return div
else:
    def _zdivide(x, y):
        return np.divide(x, y, out=np.zeros_like(x), where=(y != 0))


def _promote16(u, fn=None, *args, **kwargs):
    """Patched version of :func:`sporco.linalg.promote16`."""

    dtype = np.float32 if u.dtype == np.float16 else u.dtype
    up = u.astype(dtype=dtype)
    if fn is None:
        return up
    else:
        v = fn(up, *args, **kwargs)
        if isinstance(v, tuple):
            vp = tuple([vk.astype(dtype=u.dtype) for vk in v])
        else:
            vp = v.astype(dtype=u.dtype)
        return vp


# Construct sporco.cupy.array
array = sporco_cupy_patch_module('sporco.array',
        {'list2array': _list2array, 'zdivide': _zdivide,
         'promote16': _promote16})


# Construct sporco.cupy.cnvrep
cnvrep = sporco_cupy_patch_module('sporco.cnvrep')


# Construct sporco.cupy.common
common = sporco_cupy_patch_module('sporco.common')


def _complex_dtype(dtype):
    """Patched version of :func:`sporco.fft.complex_dtype`."""

    real_cplx = {'float128': 'complex256', 'float64': 'complex128', 'float32': 'complex64'}
    dt = cp.dtype(dtype)
    for real, cplx in real_cplx.items():
        try:
            cpdt = cp.dtype(real)
        except TypeError:
            continue
        if dt == cpdt:
            return cp.dtype(cplx)
    return cp.dtype('complex64')


def _byte_aligned(array, dtype=None, n=None):
    """Patched version of :func:`sporco.fft.byte_aligned`."""

    return array


def _empty_aligned(shape, dtype, order='C', n=None):
    """Patched version of :func:`sporco.fft.empty_aligned`."""

    return cp.empty(shape, dtype, order)


def _rfftn_empty_aligned(shape, axes, dtype, order='C', n=None):
    """Patched version of :func:`sporco.fft.rfftn_empty_aligned`.
    """

    ashp = list(shape)
    raxis = axes[-1]
    ashp[raxis] = ashp[raxis] // 2 + 1
    cdtype = _complex_dtype(dtype)
    return cp.empty(ashp, cdtype, order)


def _fftconv(a, b, axes=(0, 1)):
    """Patched version of :func:`sporco.fft.fftconv`."""

    if cp.isrealobj(a) and cp.isrealobj(b):
        fft = cp.fft.rfftn
        ifft = cp.fft.irfftn
    else:
        fft = cp.fft.fftn
        ifft = cp.fft.ifftn
    dims = cp.maximum(cp.asarray([a.shape[i] for i in axes]),
                      cp.asarray([b.shape[i] for i in axes]))
    dims = [int(d) for d in dims]
    af = fft(a, dims, axes)
    bf = fft(b, dims, axes)
    return ifft(af * bf, dims, axes)


def _empty_aligned_func(real=False):
    """Patched version of :func:`sporco.fft.empty_aligned_func`.
    """

    if real:
        return _rfftn_empty_aligned
    else:
        def empty_aligned_wrapper(shape, axes, dtype, order='C', n=None):
            return _empty_aligned(shape, dtype, order=order, n=n)
        return empty_aligned_wrapper


def _fftn_func(real=False):
    """Patched version of :func:`sporco.fft.fftn_func`.
    """

    if real:
        return cp.fft.rfftn
    else:
        return cp.fft.fftn


def _ifftn_func(real=False):
    """Patched version of :func:`sporco.fft.ifftn_func`.
    """

    if real:
        return cp.fft.irfftn
    else:
        return cp.fft.ifftn


# Construct sporco.cupy.fft
fft = sporco_cupy_patch_module('sporco.fft',
    {'complex_dtype': _complex_dtype, 'fftn': cp.fft.fftn,
     'ifftn': cp.fft.ifftn, 'rfftn': cp.fft.rfftn,
     'irfftn': cp.fft.irfftn, 'byte_aligned': _byte_aligned,
     'empty_aligned': _empty_aligned,
     'rfftn_empty_aligned': _rfftn_empty_aligned,
     'fftconv': _fftconv, 'empty_aligned_func': _empty_aligned_func,
     'fftn_func': _fftn_func, 'ifftn_func': _ifftn_func})


def _inner(x, y, axis=-1):
    """Patched version of :func:`sporco.linalg.inner`."""

    return cp.sum(x * y, axis=axis, keepdims=True)


def _cho_factor(A, lower=True, check_finite=True):
    """Implementaton of :func:`scipy.linalg.cho_factor` using
    a function supported in cupy."""

    return cp.linalg.cholesky(A), True


def _cho_solve(c_and_lower, b, check_finite=True):
    """Implementaton of :func:`scipy.linalg.cho_solve` using
    a function supported in cupy."""

    L = c_and_lower[0]
    y = cpxl.solve_triangular(L, b, trans=0, lower=True,
                              check_finite=check_finite)
    return cpxl.solve_triangular(L, y, trans=1, lower=True,
                                 check_finite=check_finite)


def _linalg_cho_factor(A, rho, lower=False, check_finite=True):
    """Patched version of :func:`sporco.linalg.cho_factor`."""

    N, M = A.shape
    if N >= M:
        c, lwr = _cho_factor(
            A.T.dot(A) + rho * cp.identity(M, dtype=A.dtype), lower=lower,
            check_finite=check_finite)
    else:
        c, lwr = _cho_factor(
            A.dot(A.T) + rho * cp.identity(N, dtype=A.dtype), lower=lower,
            check_finite=check_finite)
    return c, lwr


def _cho_solve_ATAI(A, rho, b, c, lwr, check_finite=True):
    """Patched version of :func:`sporco.linalg.cho_solve_ATAI`."""

    N, M = A.shape
    if N >= M:
        x = _cho_solve((c, lwr), b, check_finite=check_finite)
    else:
        x = (b - A.T.dot(_cho_solve((c, lwr), A.dot(b),
                                    check_finite=check_finite))) / rho
    return x


def _cho_solve_AATI(A, rho, b, c, lwr, check_finite=True):
    """Patched version of :func:`sporco.linalg.cho_solve_AATI`."""

    N, M = A.shape
    if N >= M:
        x = (b - _cho_solve((c, lwr), b.dot(A).T,
                            check_finite=check_finite).T.dot(A.T)) / rho
    else:
        x = _cho_solve((c, lwr), b.T, check_finite=check_finite).T
    return x




# Construct sporco.cupy.linalg
linalg = sporco_cupy_patch_module('sporco.linalg',
    {'have_numexpr': False, 'inner': _inner, 'cho_factor': _linalg_cho_factor,
     'cho_solve_ATAI': _cho_solve_ATAI, 'cho_solve_AATI': _cho_solve_AATI,
     'subsample_array': array.subsample_array, 'zdivide': array.zdivide})


# Construct sporco.cupy.metric
metric = sporco_cupy_patch_module('sporco.metric')


# Construct sporco.cupy.util
signal = sporco_cupy_patch_module('sporco.signal',
            {'fftn': fft.fftn, 'ifftn': fft.ifftn,
             'rfftn': fft.rfftn, 'irfftn': fft.irfftn,
             'fftconv': fft.fftconv})


# Construct sporco.cupy.util
util = sporco_cupy_patch_module('sporco.util', {'signal': signal})


# Construct sporco.cupy.prox
prox_lp = sporco_cupy_patch_module('sporco.prox._lp',
            {'have_numexpr': False, 'zdivide': _zdivide})
prox_util = sporco_cupy_patch_module('sporco.prox._util')
prox_l1proj = sporco_cupy_patch_module('sporco.prox._l1proj')
prox_nuclear = sporco_cupy_patch_module('sporco.prox._nuclear',
                                        {'promote16': _promote16})
prox = sporco_cupy_patch_module('sporco.prox', {'have_numexpr': False})
