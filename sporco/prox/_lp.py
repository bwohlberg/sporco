# -*- coding: utf-8 -*-
# Copyright (C) 2015-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r""":math:`\ell_p` norms (and the :math:`\ell_0` "norm") and their
proximal operators"""


from __future__ import division

import numpy as np
try:
    import numexpr as ne
except ImportError:
    have_numexpr = False
else:
    have_numexpr = True

from sporco.array import zdivide


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def norm_l0(x, axis=None, eps=0.0):
    r"""Compute the :math:`\ell_0` "norm" (it is not really a norm)

    .. math::
     \| \mathbf{x} \|_0 = \sum_i \left\{ \begin{array}{ccc}
     0 & \text{if} & x_i = 0 \\ 1 &\text{if}  & x_i \neq 0
     \end{array} \right.

    where :math:`x_i` is element :math:`i` of vector :math:`\mathbf{x}`.


    Parameters
    ----------
    x : array_like
      Input array :math:`\mathbf{x}`
    axis : `None` or int or tuple of ints, optional (default None)
      Axes of `x` over which to compute the :math:`\ell_0` "norm". If
      `None`, an entire multi-dimensional array is treated as a
      vector. If axes are specified, then distinct values are computed
      over the indices of the remaining axes of input array `x`.
    eps : float, optional (default 0.0)
      Absolute value threshold below which a number is considered to be
      zero.

    Returns
    -------
    nl0 : float or ndarray
      Norm of `x`, or array of norms treating specified axes of `x`
      as a vector
    """

    nl0 = np.sum(np.abs(x) > eps, axis=axis, keepdims=True)
    # If the result has a single element, convert it to a scalar
    if nl0.size == 1:
        nl0 = nl0.ravel()[0]
    return nl0



def prox_l0(v, alpha):
    r"""Compute the proximal operator of the :math:`\ell_0` "norm" (hard
    thresholding)

     .. math::
      \mathrm{prox}_{\alpha f}(v) = \mathcal{S}_{0,\alpha}(\mathbf{v})
      = \left\{ \begin{array}{ccc} 0 & \text{if} &
      | v | < \sqrt{2 \alpha} \\ v &\text{if}  &
      | v | \geq \sqrt{2 \alpha} \end{array} \right. \;,

    where :math:`f(\mathbf{x}) = \|\mathbf{x}\|_0`. The approach taken
    here is to start with the definition of the :math:`\ell_0` "norm"
    and derive the corresponding proximal operator. Note, however, that
    some authors (e.g. see Sec. 2.3 of :cite:`kowalski-2014-thresholding`)
    start by defining the hard thresholding rule and then derive the
    corresponding penalty function, which leads to a simpler form for
    the thresholding rule and a more complicated form for the penalty
    function.

    Unlike the corresponding :func:`norm_l0`, there is no need for an
    `axis` parameter since the proximal operator of the :math:`\ell_0`
    norm is the same when taken independently over each element, or
    over their sum.


    Parameters
    ----------
    v : array_like
      Input array :math:`\mathbf{v}`
    alpha : float or array_like
      Parameter :math:`\alpha`

    Returns
    -------
    x : ndarray
      Output array
    """

    return (np.abs(v) >= np.sqrt(2.0 * alpha)) * v



def norm_l1(x, axis=None):
    r"""Compute the :math:`\ell_1` norm

    .. math::
     \| \mathbf{x} \|_1 = \sum_i | x_i |

    where :math:`x_i` is element :math:`i` of vector :math:`\mathbf{x}`.


    Parameters
    ----------
    x : array_like
      Input array :math:`\mathbf{x}`
    axis : `None` or int or tuple of ints, optional (default None)
      Axes of `x` over which to compute the :math:`\ell_1` norm. If
      `None`, an entire multi-dimensional array is treated as a
      vector. If axes are specified, then distinct values are computed
      over the indices of the remaining axes of input array `x`.

    Returns
    -------
    nl1 : float or ndarray
      Norm of `x`, or array of norms treating specified axes of `x`
      as a vector
    """

    nl1 = np.sum(np.abs(x), axis=axis, keepdims=True)
    # If the result has a single element, convert it to a scalar
    if nl1.size == 1:
        nl1 = nl1.ravel()[0]
    return nl1



def prox_l1(v, alpha):
    r"""Compute the proximal operator of the :math:`\ell_1` norm (scalar
    shrinkage/soft thresholding)

     .. math::
      \mathrm{prox}_{\alpha f}(\mathbf{v}) =
      \mathcal{S}_{1,\alpha}(\mathbf{v}) = \mathrm{sign}(\mathbf{v})
      \odot \max(0, |\mathbf{v}| - \alpha)

    where :math:`f(\mathbf{x}) = \|\mathbf{x}\|_1`.

    Unlike the corresponding :func:`norm_l1`, there is no need for an
    `axis` parameter since the proximal operator of the :math:`\ell_1`
    norm is the same when taken independently over each element, or
    over their sum.


    Parameters
    ----------
    v : array_like
      Input array :math:`\mathbf{v}`
    alpha : float or array_like
      Parameter :math:`\alpha`

    Returns
    -------
    x : ndarray
      Output array
    """

    if np.isrealobj(v):
        if have_numexpr:
            return ne.evaluate(
                'where(abs(v)-alpha > 0, where(v >= 0, 1, -1) * '
                '(abs(v)-alpha), 0)'
            )
        else:
            return np.sign(v) * (np.clip(np.abs(v) - alpha, 0, float('Inf')))
    else:
        return (v / np.abs(v)) * (np.clip(np.abs(v) - alpha, 0, float('Inf')))



def norm_2l2(x, axis=None):
    r"""Compute the squared :math:`\ell_2` norm

    .. math::
      \| \mathbf{x} \|_2^2 = \sum_i x_i^2

    where :math:`x_i` is element :math:`i` of vector :math:`\mathbf{x}`.

    Parameters
    ----------
    x : array_like
      Input array :math:`\mathbf{x}`
    axis : `None` or int or tuple of ints, optional (default None)
      Axes of `x` over which to compute the :math:`\ell_2` norm. If
      `None`, an entire multi-dimensional array is treated as a
      vector. If axes are specified, then distinct values are computed
      over the indices of the remaining axes of input array `x`.

    Returns
    -------
    nl2 : float or ndarray
      Norm of `x`, or array of norms treating specified axes of `x`
      as a vector
    """

    if np.isrealobj(x):
        nl2 = np.sum(x**2, axis=axis, keepdims=True)
    else:
        nl2 = np.sum(np.abs(x)**2, axis=axis, keepdims=True)
    # If the result has a single element, convert it to a scalar
    if nl2.size == 1:
        nl2 = nl2.ravel()[0]
    return nl2



def norm_l2(x, axis=None):
    r"""Compute the :math:`\ell_2` norm

    .. math::
      \| \mathbf{x} \|_2 = \sqrt{ \sum_i x_i^2 }

    where :math:`x_i` is element :math:`i` of vector :math:`\mathbf{x}`.

    Parameters
    ----------
    x : array_like
      Input array :math:`\mathbf{x}`
    axis : `None` or int or tuple of ints, optional (default None)
      Axes of `x` over which to compute the :math:`\ell_2` norm. If
      `None`, an entire multi-dimensional array is treated as a
      vector. If axes are specified, then distinct values are computed
      over the indices of the remaining axes of input array `x`.

    Returns
    -------
    nl2 : float or ndarray
      Norm of `x`, or array of norms treating specified axes of `x`
      as a vector.
    """

    return np.sqrt(norm_2l2(x, axis))



def prox_l2(v, alpha, axis=None):
    r"""Compute the proximal operator of the :math:`\ell_2` norm (vector
    shrinkage/soft thresholding)

    .. math::
     \mathrm{prox}_{\alpha f}(\mathbf{v}) = \mathcal{S}_{2,\alpha}
     (\mathbf{v}) = \frac{\mathbf{v}} {\|\mathbf{v}\|_2} \max(0,
     \|\mathbf{v}\|_2 - \alpha) \;,

    where :math:`f(\mathbf{x}) = \|\mathbf{x}\|_2`.


    Parameters
    ----------
    v : array_like
      Input array :math:`\mathbf{v}`
    alpha : float or array_like
      Parameter :math:`\alpha`
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

    if np.isrealobj(v):
        a = np.sqrt(np.sum(v**2, axis=axis, keepdims=True))
    else:
        a = np.sqrt(np.sum(np.abs(v)**2, axis=axis, keepdims=True))
    b = np.maximum(0, a - alpha)
    b = zdivide(b, a)
    return np.asarray(b * v, dtype=v.dtype)



def proj_l2(v, gamma, axis=None):
    r"""Compute the projection operator of the :math:`\ell_2` norm

    .. math::
     \mathrm{proj}_{f, \gamma}(\mathbf{v}) = \mathrm{argmin}_{\mathbf{x}}
     (1/2) \| \mathbf{x} - \mathbf{v} \|_2^2 \;
     \text{ s.t. } \; \| \mathbf{x} \|_2 \leq \gamma \;,

    where :math:`f(\mathbf{x}) = \|\mathbf{x}\|_2`.

    Note that the projection operator of the :math:`\ell_2` norm
    centered at :math:`\mathbf{s}`,

    .. math::
      \mathrm{argmin}_{\mathbf{x}} (1/2) \| \mathbf{x} - \mathbf{v}
      \|_2^2 \; \text{ s.t. } \; \| \mathbf{x} - \mathbf{s} \|_2 \leq
      \gamma \;,

    can be computed as :math:`\mathbf{s} + \mathrm{proj}_{f,\gamma}
    (\mathbf{v} - \mathbf{s})`.


    Parameters
    ----------
    v : array_like
      Input array :math:`\mathbf{v}`
    gamma : float
      Parameter :math:`\gamma`
    axis : None or int or tuple of ints, optional (default None)
      Axes of `v` over which to compute the :math:`\ell_2` norm. If
      `None`, an entire multi-dimensional array is treated as a vector.
      If axes are specified, then distinct norm values are computed
      over the indices of the remaining axes of input array `v`.

    Returns
    -------
    x : ndarray
      Output array
    """

    if np.isrealobj(v):
        d = np.sqrt(np.sum(v**2, axis=axis, keepdims=True))
    else:
        d = np.sqrt(np.sum(np.abs(v)**2, axis=axis, keepdims=True))
    return np.asarray((d <= gamma) * v +
                      (d > gamma) * (gamma * zdivide(v, d)),
                      dtype=v.dtype)
