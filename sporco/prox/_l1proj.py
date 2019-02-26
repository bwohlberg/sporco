# -*- coding: utf-8 -*-
# Copyright (C) 2015-2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Projection operator of the :math:`\ell_1` norm"""


from __future__ import division
from builtins import range

import numpy as np
import scipy.optimize as optim

from ._util import ndto2d, ndfrom2d
from ._lp import norm_l1, prox_l1


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


def proj_l1(v, gamma, axis=None, method=None):
    r"""Projection operator of the :math:`\ell_1` norm.


    Parameters
    ----------
    v : array_like
      Input array :math:`\mathbf{v}`
    gamma : float
      Parameter :math:`\gamma`
    axis : None or int or tuple of ints, optional (default None)
      Axes of `v` over which to compute the :math:`\ell_1` norm. If
      `None`, an entire multi-dimensional array is treated as a
      vector. If axes are specified, then distinct norm values are
      computed over the indices of the remaining axes of input array
      `v`.
    method : None or str, optional (default None)
      Solver method to use. If `None`, the most appropriate choice is
      made based on the `axis` parameter. Valid methods are

         - 'scalarroot'
            The solution is computed via the method of Sec. 6.5.2 in
            :cite:`parikh-2014-proximal`.
         - 'sortcumsum'
            The solution is computed via the method of
            :cite:`duchi-2008-efficient`.

    Returns
    -------
    x : ndarray
      Output array
    """

    if method is None:
        if axis is None:
            method = 'scalarroot'
        else:
            method = 'sortcumsum'

    if method == 'scalarroot':
        if axis is not None:
            raise ValueError('Method scalarroot only supports axis=None')
        return _proj_l1_scalar_root(v, gamma)
    elif method == 'sortcumsum':
        if isinstance(axis, tuple):
            vtr, rsi = ndto2d(v, axis)
            xtr = _proj_l1_sortsum(vtr, gamma, axis=1)
            return ndfrom2d(xtr, rsi)
        else:
            return _proj_l1_sortsum(v, gamma, axis)
    else:
        raise ValueError('Unknown solver method %s' % method)



def _proj_l1_scalar_root(v, gamma):
    r"""Projection operator of the :math:`\ell_1` norm. The solution is
    computed via the method of Sec. 6.5.2 in :cite:`parikh-2014-proximal`.

    There is no `axis` parameter since the algorithm for computing the
    solution treats the input `v` as a single vector.


    Parameters
    ----------
    v : array_like
      Input array :math:`\mathbf{v}`
    gamma : float
      Parameter :math:`\gamma`

    Returns
    -------
    x : ndarray
      Output array
    """

    if norm_l1(v) <= gamma:
        return v
    else:
        av = np.abs(v)
        fn = lambda t: np.sum(np.maximum(0, av - t)) - gamma
        t = optim.brentq(fn, 0, av.max())
        return prox_l1(v, t)



def _proj_l1_sortsum(v, gamma, axis=None):
    r"""Projection operator of the :math:`\ell_1` norm. The solution is
    computed via the method of :cite:`duchi-2008-efficient`.


    Parameters
    ----------
    v : array_like
      Input array :math:`\mathbf{v}`
    gamma : float
      Parameter :math:`\gamma`
    axis : None or int, optional (default None)
      Axes of `v` over which to compute the :math:`\ell_1` norm. If
      `None`, an entire multi-dimensional array is treated as a
      vector. If axes are specified, then distinct norm values are
      computed over the indices of the remaining axes of input array
      `v`. **Note:** specifying a tuple of ints is not supported by
      this function.

    Returns
    -------
    x : ndarray
      Output array
    """

    if axis is None and norm_l1(v) <= gamma:
        return v
    if axis is not None and axis < 0:
        axis = v.ndim + axis
    av = np.abs(v)
    vs = np.sort(av, axis=axis)
    if axis is None:
        N = v.size
        c = 1.0 / np.arange(1, N + 1, dtype=v.dtype).reshape(v.shape)
        vs = vs[::-1].reshape(v.shape)
    else:
        N = v.shape[axis]
        ns = [v.shape[k] if k == axis else 1 for k in range(v.ndim)]
        c = 1.0 / np.arange(1, N + 1, dtype=v.dtype).reshape(ns)
        vs = vs[(slice(None),) * axis + (slice(None, None, -1),)]
    t = c * (np.cumsum(vs, axis=axis).reshape(v.shape) - gamma)
    K = np.sum(vs >= t, axis=axis, keepdims=True)
    t = (np.sum(vs * (vs >= t), axis=axis, keepdims=True) - gamma) / K
    t = np.asarray(np.maximum(0, t), dtype=v.dtype)
    return np.sign(v) * np.where(av > t, av - t, 0)
