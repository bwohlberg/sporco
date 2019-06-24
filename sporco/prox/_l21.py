# -*- coding: utf-8 -*-
# Copyright (C) 2015-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Compute the :math:`\ell_{2,1}` mixed norm and its proximal operator"""

from __future__ import division

import numpy as np

from ._lp import prox_l1, prox_l2, norm_l2


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def norm_l21(x, axis=-1):
    r"""Compute the :math:`\ell_{2,1}` mixed norm

    .. math::
      \| X \|_{2,1} = \sum_i \sqrt{ \sum_j X_{i,j}^2 }

    where :math:`X_{i,j}` is element :math:`i,j` of matrix :math:`X`.

    Parameters
    ----------
    x : array_like
      Input array :math:`X`
    axis : None or int or tuple of ints, optional (default -1)
      Axes of `x` over which to compute the :math:`\ell_2` norm. If
      `None`, an entire multi-dimensional array is treated as a
      vector, in which case the result is just the :math:`\ell_2` norm.
      If axes are specified, then the sum over the :math:`\ell_2` norm
      values is computed over the indices of the remaining axes of input
      array `x`.

    Returns
    -------
    nl21 : float
      Norm of :math:`X`
    """

    return np.sum(norm_l2(x, axis=axis))



def prox_sl1l2(v, alpha, beta, axis=None):
    r"""Compute the proximal operator of the sum of :math:`\ell_1` and
    :math:`\ell_2` norms (compound shrinkage/soft thresholding)
    :cite:`wohlberg-2012-local` :cite:`chartrand-2013-nonconvex`

     .. math::
      \mathrm{prox}_{f}(\mathbf{v}) =
      \mathcal{S}_{1,2,\alpha,\beta}(\mathbf{v}) =
      \mathcal{S}_{2,\beta}(\mathcal{S}_{1,\alpha}(\mathbf{v}))

    where :math:`f(\mathbf{x}) = \alpha \|\mathbf{x}\|_1 +
    \beta \|\mathbf{x}\|_2`, with :math:`\alpha \geq 0`,
    :math:`\beta \geq 0`.


    Parameters
    ----------
    v : array_like
      Input array :math:`\mathbf{v}`
    alpha : float or array_like
      Parameter :math:`\alpha`
    beta : float or array_like
      Parameter :math:`\beta`
    axis : None or int or tuple of ints, optional (default None)
      Axes of `v` over which to compute the :math:`\ell_2` norm. If
      `None`, an entire multi-dimensional array is treated as a
      vector. If axes are specified, then distinct norm values are
      computed over the indices of the remaining axes of input array
      `v`, which is equivalent to the proximal operator of the sum over
      these values (i.e. an :math:`\ell_{2,1}` norm).

    Returns
    -------
    x : ndarray
      Output array
    """

    return prox_l2(prox_l1(v, alpha), beta, axis)
