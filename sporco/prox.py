# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Norms and their associated proximal maps and projections

   The :math:`p`-norm of a vector is defined as

   .. math::
    \| \mathbf{x} \|_p = \left( \sum_i | x_i |^p \right)^{1/p}

   where :math:`x_i` is element :math:`i` of vector :math:`\mathbf{x}`.
   The max norm is a special case

   .. math::
    \| \mathbf{x} \|_{\infty} = \max_i | x_i | \;\;.

   The mixed matrix norm :math:`\|X\|_{p,q}` is defined here as
   :cite:`kowalski-2009-sparse`

   .. math::
     \|X\|_{p,q} = \left( \sum_i \left( \sum_j |X_{i,j}|^p \right)^{q/p}
     \right)^{1/q} = \left( \sum_i \| \mathbf{x}_i  \|_p^q \right)^{1/q}

   where :math:`\mathbf{x}_i` is row :math:`i` of matrix
   :math:`X`. Note that some authors use a notation that reverses the
   positions of :math:`p` and :math:`q`.

   The proximal operator of function :math:`f` is defined as

   .. math::
    \mathrm{prox}_f(\mathbf{v}) = \mathrm{argmin}_{\mathbf{x}}
    \left\{ (1/2) \| \mathbf{x} - \mathbf{v} \|_2^2 + f(\mathbf{x})
    \right\} \;\;.

   The projection operator of function :math:`f` is defined as

   .. math::
    \mathrm{proj}_{f,\gamma}(\mathbf{v}) &= \mathrm{argmin}_{\mathbf{x}}
    (1/2) \| \mathbf{x} - \mathbf{v} \|_2^2 \; \text{ s.t. } \;
    f(\mathbf{x}) \leq \gamma \\ &= \mathrm{prox}_g(\mathbf{v})

   where :math:`g(\mathbf{v}) = \iota_C(\mathbf{v})`, with
   :math:`\iota_C` denoting the indicator function of set
   :math:`C = \{ \mathbf{x} \; | \; f(\mathbf{x}) \leq \gamma \}`.
"""

from __future__ import division
from builtins import range

import numpy as np
import scipy.optimize as optim
try:
    import numexpr as ne
except ImportError:
    have_numexpr = False
else:
    have_numexpr = True

import sporco.linalg as sl


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
      Absolute value threshold below which a number is considered to be zero.

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
    r"""Proximal operator of the :math:`\ell_0` "norm" (hard thresholding)

     .. math::
      \mathrm{prox}_{\alpha f}(v) = \mathcal{S}_{0,\alpha}(\mathbf{v})
      = \left\{ \begin{array}{ccc} 0 & \text{if} &
      | v | < \sqrt{2 \alpha} \\ v &\text{if}  &
      | v | \geq \sqrt{2 \alpha} \end{array} \right.

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

    return (np.abs(v) >= np.sqrt(2.0*alpha)) * v



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
    r"""Proximal operator of the :math:`\ell_1` norm (scalar
    shrinkage/soft thresholding)

     .. math::
      \mathrm{prox}_{\alpha f}(\mathbf{v}) =
      \mathcal{S}_{1,\alpha}(\mathbf{v}) = \mathrm{sign}(\mathbf{v}) \odot
      \max(0, |\mathbf{v}| - \alpha)

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

    if have_numexpr:
        return ne.evaluate(
            'where(abs(v)-alpha > 0, where(v >= 0, 1, -1) * (abs(v)-alpha), 0)'
        )
    else:
        return np.sign(v) * (np.clip(np.abs(v) - alpha, 0, float('Inf')))



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
      `v`. **Note:** specifying a tuple of ints is not supported by
      this function.
    method : None or str, optional (default None)
      Solver method to use. If `None`, the most appropriate choice is made
      based on the `axis` parameter. Valid methods are

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
            raise ValueError('Method sortcumsum does not support tuple axis'
                             ' values')
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
        c = 1.0 / np.arange(1, N+1, dtype=v.dtype).reshape(v.shape)
        vs = vs[::-1].reshape(v.shape)
    else:
        N = v.shape[axis]
        ns = [v.shape[k] if k == axis else 1 for k in range(v.ndim)]
        c = 1.0 / np.arange(1, N+1, dtype=v.dtype).reshape(ns)
        vs = vs[(slice(None),)*axis + (slice(None, None, -1),)]
    t = c * (np.cumsum(vs, axis=axis).reshape(v.shape) - gamma)
    K = np.sum(vs >= t, axis=axis, keepdims=True)
    t = (np.sum(vs * (vs >= t), axis=axis, keepdims=True) - gamma) / K
    t = np.asarray(np.maximum(0, t), dtype=v.dtype)
    return np.sign(v) * np.where(av > t, av - t, 0)



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
      as a vector.
    """

    nl2 = np.sum(x**2, axis=axis, keepdims=True)
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



def prox_l2(v, alpha, axis=None):
    r"""Proximal operator of the :math:`\ell_2` norm (vector shrinkage/soft
    thresholding)

    .. math::
     \mathrm{prox}_{\alpha f}(\mathbf{v}) = \frac{\mathbf{v}}
     {\|\mathbf{v}\|_2} \max(0, \|\mathbf{v}\|_2 - \alpha) =
     \mathcal{S}_{2,\alpha}(\mathbf{v})

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

    a = np.sqrt(np.sum(v**2, axis=axis, keepdims=True))
    b = np.maximum(0, a - alpha)
    b = sl.zdivide(b, a)
    return b*v



def proj_l2(v, gamma, axis=None):
    r"""Projection operator of the :math:`\ell_2` norm.

    The projection operator of the uncentered :math:`\ell_2` norm,

    .. math::
      \mathrm{argmin}_{\mathbf{x}} (1/2) \| \mathbf{x} - \mathbf{v} \|_2^2 \;
      \text{ s.t. } \; \| \mathbf{x} - \mathbf{s} \|_2 \leq \gamma

    can be computed as :math:`\mathbf{s} + \mathrm{proj}_{f,\gamma}
    (\mathbf{v} - \mathbf{s})` where :math:`f(\mathbf{x}) =
    \| \mathbf{x} \|_2`.


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

    d = np.sqrt(np.sum(v**2, axis=axis, keepdims=True))
    return (d <= gamma)*v + (d > gamma)*(gamma*sl.zdivide(v, d))



def prox_l1l2(v, alpha, beta, axis=None):
    r"""Proximal operator of the :math:`\ell_1` plus :math:`\ell_2` norm
    (compound shrinkage/soft thresholding) :cite:`wohlberg-2012-local`
    :cite:`chartrand-2013-nonconvex`

     .. math::
      \mathrm{prox}_{f}(\mathbf{v}) =
      \mathcal{S}_{1,2,\alpha,\beta}(\mathbf{v}) =
      \mathcal{S}_{2,\beta}(\mathcal{S}_{1,\alpha}(\mathbf{v}))

    where :math:`f(\mathbf{x}) = \alpha \|\mathbf{x}\|_1 +
    \beta \|\mathbf{x}\|_2`.


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



def norm_nuclear(x):
    r"""Compute the nuclear norm

    .. math::
      \| X \|_1 = \sum_i \sigma_i

    where :math:`\sigma_i` are the singular values of matrix :math:`X`.


    Parameters
    ----------
    x : array_like
      Input array :math:`X`

    Returns
    -------
    nncl : float
      Norm of `x`
    """

    return np.sum(np.linalg.svd(sl.promote16(x), compute_uv=False))



def prox_nuclear(v, alpha):
    r"""Proximal operator of the nuclear norm :cite:`cai-2010-singular`.


    Parameters
    ----------
    v : array_like
      Input array :math:`V`
    alpha : float
      Parameter :math:`\alpha`

    Returns
    -------
    x : ndarray
      Output array
    s : ndarray
      Singular values of `x`
    """

    U, s, V = sl.promote16(v, fn=np.linalg.svd, full_matrices=False)
    ss = np.maximum(0, s - alpha)
    return np.dot(U, np.dot(np.diag(ss), V)), ss
