# -*- coding: utf-8 -*-
# Copyright (C) 2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Compute the difference of :math:`\ell_1` and :math:`\ell_2` norms and the corresponding proximal operator"""

from __future__ import division

import numpy as np

from ._lp import norm_l1, norm_l2


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def norm_dl1l2(x, beta=1.0, axis=None):
    r"""Compute the difference of :math:`\ell_1` and :math:`\ell_2`
    norms, i.e. :math:`\| X \|_1 - \beta \| X \|_2`, for matrix
    :math:`X`.

    Parameters
    ----------
    x : array_like
      Input array :math:`X`
    beta : float, optional (default 1.0)
      Parameter :math:`\beta \geq 0`
    axis : `None` or int or tuple of ints, optional (default None)
      Axes of `x` over which to compute the :math:`\ell_1` and
      :math:`\ell_2` norms. If `None`, an entire multi-dimensional array
      is treated as a vector. If axes are specified, then distinct
      values are computed over the indices of the remaining axes of
      input array `x`.

    Returns
    -------
    dl1l2 : float or ndarray
      Difference of :math:`\ell_1` and :math:`\ell_2` norms of :math:`X`
    """

    return norm_l1(x, axis=axis) - beta * norm_l2(x, axis=axis)



def prox_dl1l2(v, alpha, beta=1.0, axis=None):
    r"""Compute the proximal operator of the difference of :math:`\ell_1`
    and :math:`\ell_2` norms, i.e. :math:`\alpha \left( \| X \|_1 -
    \beta \| X \|_2 \right)` :cite:`lou-2018-fast`. The function block
    for the `axis=None` case is a Python translation of the `Matlab
    implementation
    <https://github.com/mingyan08/ProxL1-L2/blob/master/shrinkL12.m>`__
    by the authors of :cite:`lou-2018-fast`.


    Parameters
    ----------
    v : array_like
      Input array :math:`\mathbf{v}`
    alpha : float
      Parameter :math:`\alpha \geq 0`
    beta : float, optional (default 1.0)
      Parameter :math:`1 \leq \beta \geq 0`
    axis : None or int, optional (default None)
      Axis of `v` over which to compute the difference of :math:`\ell_1`
      and :math:`\ell_2` norms. If `None`, an entire multi-dimensional
      array is treated as a vector. If the axis is specified, then
      distinct norm values are computed over the indices of the remaining
      axes of input array `v`, which is equivalent to the proximal
      operator of the sum over these components.

    Returns
    -------
    x : ndarray
      Output array
    """

    va = np.abs(v)
    if axis is None:
        vamx = np.max(va)
        if vamx > 0:
            if vamx > alpha:
                u = np.maximum(va - alpha, 0) * np.sign(v)
                u *= (norm_l2(u) + alpha * beta) / norm_l2(u)
            else:
                u = np.zeros(v.shape)
                if vamx >= (1 - beta) * alpha:
                    idx = va.ravel().argmax()
                    u.ravel()[idx] = (va.ravel()[idx] + (beta - 1) * alpha) * \
                                     np.sign(v.ravel()[idx])
        else:
            u = np.zeros(v.shape)
    else:
        vamx = np.max(va, axis=axis, keepdims=True)
        u1 = np.maximum(va - alpha, 0) * np.sign(v)
        u1l2 = norm_l2(u1, axis=axis)
        u1 *= 1.0 + np.divide(alpha * beta, u1l2, out=np.zeros_like(u1l2),
                              where=(u1l2 != 0))
        u2 = np.zeros(v.shape)
        idx = np.expand_dims(va.argmax(axis=axis), axis=axis)
        vsgn = np.sign(np.take_along_axis(v, idx, axis=axis))
        np.put_along_axis(u2, idx, (vamx + (beta - 1) * alpha) * vsgn,
                          axis=axis)
        u = (vamx > alpha) * u1 + np.logical_and(vamx > (1 - beta) * alpha,
                                                 vamx <= alpha) * u2

    return u
