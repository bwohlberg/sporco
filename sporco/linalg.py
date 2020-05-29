# -*- coding: utf-8 -*-
# Copyright (C) 2015-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Linear algebra functions."""

from __future__ import division
from builtins import range

import warnings
import numpy as np
import scipy
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, cg
try:
    import numexpr as ne
except ImportError:
    have_numexpr = False
else:
    have_numexpr = True

from sporco._util import renamed_function
from sporco.array import zdivide, subsample_array


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



__all__ = ['inner', 'dot', 'solvedbi_sm', 'solvedbi_sm_c', 'solvedbd_sm',
           'solvedbd_sm_c', 'solvemdbi_ism', 'solvemdbi_rsm',
           'solvemdbi_cg', 'lu_factor', 'lu_solve_ATAI', 'lu_solve_AATI',
           'cho_factor', 'cho_solve_ATAI', 'cho_solve_AATI',
           'block_circulant', 'rrs', 'pca', 'nkp', 'kpsvd',
           'proj_l2ball']



def inner(x, y, axis=-1):
    """Inner product of `x` and `y` on specified axis.

    Compute inner product of `x` and `y` on specified axis, equivalent to
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
    """Matrix product of `a` and the specified axes of `b`.

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



@renamed_function(depname='blockcirculant')
def block_circulant(A):
    """Construct a block circulant matrix from a tuple of arrays.

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



def solvedbi_sm(ah, rho, b, c=None, axis=4):
    r"""Solve a diagonal block linear system with a scaled identity term
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
        return ne.evaluate('(b - (a * cb)) / rho').astype(a.dtype)
    else:
        return (b - (a * inner(c, b, axis=axis))) / rho



def solvedbi_sm_c(ah, a, rho, axis=4):
    r"""Compute cached component used by :func:`solvedbi_sm`.

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
    r"""Solve a diagonal block linear system with a diagonal term
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
    r"""Compute cached component used by :func:`solvedbd_sm`.

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
    r"""Solve a multiple diagonal block linear system with a scaled
    identity term by iterated application of the Sherman-Morrison
    equation.

    The computation is performed in a way that avoids explictly
    constructing the inverse operator, leading to an :math:`O(K^2)`
    time cost.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

    .. math::
      (\rho I + \mathbf{a}_0 \mathbf{a}_0^H + \mathbf{a}_1
       \mathbf{a}_1^H + \; \ldots \; + \mathbf{a}_{K-1}
       \mathbf{a}_{K-1}^H) \; \mathbf{x} = \mathbf{b}

    where each :math:`\mathbf{a}_k` is an :math:`M`-vector.
    The sums, inner products, and matrix products in this equation are
    taken along the :math:`M` and :math:`K` axes of the corresponding
    multi-dimensional arrays; the solutions are independent over the
    other axes.

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
        beta -= d / delta[slck]

        if k < K - 1:
            alpha[:] = np.take(a, [k + 1], axisK) / rho
            for l in range(0, k + 1):
                slcl = slcnc + (slice(l, l + 1),)
                d = gamma[slcl] * inner(ah[slcl], alpha, axis=axisM)
                alpha -= d / delta[slcl]

    return beta



def solvemdbi_rsm(ah, rho, b, axisK, dimN=2):
    r"""Solve a multiple diagonal block linear system with a scaled
    identity term by repeated application of the Sherman-Morrison
    equation.

    The computation is performed by explictly constructing the inverse
    operator, leading to an :math:`O(K)` time cost and :math:`O(M^2)`
    memory cost, where :math:`M` is the dimension of the axis over which
    :math:`M` inner products are taken.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

    .. math::
      (\rho I + \mathbf{a}_0 \mathbf{a}_0^H + \mathbf{a}_1
       \mathbf{a}_1^H + \; \ldots \; + \mathbf{a}_{K-1}
       \mathbf{a}_{K-1}^H) \; \mathbf{x} = \mathbf{b}

    where each :math:`\mathbf{a}_k` is an :math:`M`-vector.
    The sums, inner products, and matrix products in this equation are
    taken along the :math:`M` and :math:`K` axes of the corresponding
    multi-dimensional arrays; the solutions are independent over the
    other axes.

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
      Number of spatial dimensions arranged as leading axes in input
      array. Axis M is taken to be at dimN+2.

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
    r"""Solve a multiple diagonal block linear system with a scaled
    identity term using CG.

    Solve a multiple diagonal block linear system with a scaled
    identity term using Conjugate Gradient (CG) via
    :func:`scipy.sparse.linalg.cg`.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

     .. math::
      (\rho I + \mathbf{a}_0 \mathbf{a}_0^H + \mathbf{a}_1
       \mathbf{a}_1^H + \; \ldots \; + \mathbf{a}_{K-1}
       \mathbf{a}_{K-1}^H) \; \mathbf{x} = \mathbf{b}

    where each :math:`\mathbf{a}_k` is an :math:`M`-vector. The inner
    products and matrix products in this equation are taken along the
    :math:`M` and :math:`K` axes of the corresponding multi-dimensional
    arrays; the solutions are independent over the other axes.

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
    r"""Compute LU factorisation of either :math:`A^T A + \rho I` or
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
      Matrix containing U in its upper triangle, and L in its lower
      triangle, as returned by :func:`scipy.linalg.lu_factor`
    piv : ndarray
      Pivot indices representing the permutation matrix P, as returned
      by :func:`scipy.linalg.lu_factor`
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
    r"""Solve the linear system :math:`(A^T A + \rho I)\mathbf{x} = \mathbf{b}`
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
    r"""Solve the linear system :math:`(A A^T + \rho I)\mathbf{x} = \mathbf{b}`
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
    r"""Compute Cholesky factorisation of either :math:`A^T A + \rho I` or
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
    r"""Solve the linear system :math:`(A^T A + \rho I)\mathbf{x} = \mathbf{b}`
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
    r"""Solve the linear system :math:`(A A^T + \rho I)\mathbf{x} = \mathbf{b}`
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



def rrs(ax, b):
    r"""Relative residual of the solution to a linear equation.

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



@renamed_function(depname='pca', depmod='sporco.util')
def pca(U, centre=False):
    """Compute the PCA basis for columns of input array `U`.

    Parameters
    ----------
    U : array_like
      2D data array with rows corresponding to different variables and
      columns corresponding to different observations
    center : bool, optional (default False)
      Flag indicating whether to centre data

    Returns
    -------
    B : ndarray
      A 2D array representing the PCA basis; each column is a PCA
      component. `B.T` is the analysis transform into the PCA
      representation, and `B` is the corresponding synthesis transform
    S : ndarray
      The eigenvalues of the PCA components
    C : ndarray or None
      None if centering is disabled, otherwise the mean of the data
      matrix subtracted in performing the centering
    """

    if centre:
        C = np.mean(U, axis=1, keepdims=True)
        U = U - C
    else:
        C = None

    B, S, _ = np.linalg.svd(U, full_matrices=False, compute_uv=True)
    return B, S**2, C



@renamed_function(depname='nkp', depmod='sporco.util')
def nkp(A, bshape, cshape):
    r"""Solve the Nearest Kronecker Product problem.

    Given matrix :math:`A`, find matrices :math:`B` and :math:`C`, of the
    specified sizes, such that :math:`B` and :math:`C` solve the problem
    :cite:`loan-2000-ubiquitous`

    .. math::
      \mathrm{argmin}_{B, C} \| A - B \otimes C \| \;.

    Parameters
    ----------
    A : array_like
      2D input array
    bshape : tuple (Mb, Nb)
      The desired shape of returned array :math:`B`
    cshape : tuple (Mc, Nc)
      The desired shape of returned array :math:`C`

    Returns
    -------
    B : ndarray
      2D output array :math:`B`
    C : ndarray
      2D output array :math:`C`
    """

    ashape = A.shape
    if ashape[0] != bshape[0] * cshape[0] or \
       ashape[1] != bshape[1] * cshape[1]:
        raise ValueError("Shape of A is not compatible with bshape and cshape")

    atshape = (bshape[0] * bshape[1], cshape[0] * cshape[1])
    Atilde = subsample_array(A, cshape).transpose().reshape(atshape)
    U, S, Vt = np.linalg.svd(Atilde, full_matrices=False)
    B = np.sqrt(S[0]) * U[:, [0]].reshape(bshape, order='F')
    C = np.sqrt(S[0]) * Vt[[0], :].reshape(cshape, order='F')
    return B, C



@renamed_function(depname='kpsvd', depmod='sporco.util')
def kpsvd(A, bshape, cshape):
    r"""Compute the Kronecker Product SVD.

    Given matrix :math:`A`, find matrices :math:`B_i` and :math:`C_i`,
    of the specified sizes, such that

    .. math::
      A = \sum_i \sigma_i B_i \otimes C_i

    and :math:`\sum_i^n \sigma_i B_i \otimes C_i` is the best :math:`n`
    term approximation to :math:`A` :cite:`loan-2000-ubiquitous`.

    Parameters
    ----------
    A : array_like
      2D input array
    bshape : tuple (Mb, Nb)
      The desired shape of arrays :math:`B_i`
    cshape : tuple (Mc, Nc)
      The desired shape of arrays :math:`C_i`

    Returns
    -------
    S : ndarray
      1D array of :math:`\sigma_i` values
    B : ndarray
      3D array of :math:`B_i` matrices with index :math:`i` on the last
      axis
    C : ndarray
      3D array of :math:`C_i` matrices with index :math:`i` on the last
      axis
    """

    ashape = A.shape
    if ashape[0] != bshape[0] * cshape[0] or \
       ashape[1] != bshape[1] * cshape[1]:
        raise ValueError("Shape of A is not compatible with bshape and cshape")

    atshape = (bshape[0] * bshape[1], cshape[0] * cshape[1])
    Atilde = subsample_array(A, cshape).transpose().reshape(atshape)
    U, S, Vt = np.linalg.svd(Atilde, full_matrices=False)
    B = U.reshape(bshape + (U.shape[1],), order='F')
    C = Vt.T.reshape(cshape + (Vt.shape[0],), order='F')
    return S, B, C



def proj_l2ball(b, s, r, axes=None):
    r"""Projection onto the :math:`\ell_2` ball.

    Project :math:`\mathbf{b}` onto the :math:`\ell_2` ball of radius
    :math:`r` about :math:`\mathbf{s}`, i.e.
    :math:`\{ \mathbf{x} : \|\mathbf{x} - \mathbf{s} \|_2 \leq r \}`.
    Note that ``proj_l2ball(b, s, r)`` is equivalent to
    :func:`.prox.proj_l2` ``(b - s, r) + s``.

    **NB**: This function is to be deprecated; please use
    :func:`.prox.proj_l2` instead (see note above about interface
    differences).

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

    wstr = "Function sporco.linalg.proj_l2ball is deprecated; please " \
           "use sporco.prox.proj_l2 (noting the interface difference) " \
           "instead."
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(wstr, DeprecationWarning, stacklevel=2)
    warnings.simplefilter('default', DeprecationWarning)
    d = np.sqrt(np.sum((b - s)**2, axis=axes, keepdims=True))
    p = zdivide(b - s, d)
    return np.asarray((d <= r) * b + (d > r) * (s + r*p), b.dtype)
