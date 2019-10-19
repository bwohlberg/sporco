# -*- coding: utf-8 -*-
# Copyright (C) 2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Interpolation and regression functions."""

from __future__ import absolute_import, division
from builtins import range

import warnings
import numpy as np
import scipy.optimize as sco

from sporco._util import renamed_function


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



@renamed_function(depname='lstabsdev', depmod='sporco.util')
def lstabsdev(A, b):
    r"""Least absolute deviations (LAD) linear regression.

    Solve the linear regression problem

    .. math::
      \mathrm{argmin}_\mathbf{x} \; \left\| A \mathbf{x} - \mathbf{b}
      \right\|_1 \;\;.

    The interface is similar to that of :func:`numpy.linalg.lstsq` in
    that `np.linalg.lstsq(A, b)` solves the same linear regression
    problem, but with a least squares rather than a least absolute
    deviations objective. Unlike :func:`numpy.linalg.lstsq`, `b` is
    required to be a 1-d array. The solution is obtained via `mapping to
    a linear program <https://stats.stackexchange.com/a/12564>`__.

    Parameters
    ----------
    A : (M, N) array_like
      Regression coefficient matrix.
    b : (M,) array_like
      Regression ordinate / dependent variable

    Returns
    -------
    x : (N,) ndarray
      Least absolute deviations solution
    """

    M, N = A.shape
    c = np.zeros((M + N,))
    c[0:M] = 1.0
    I = np.identity(M)
    A_ub = np.hstack((np.vstack((-I, -I)), np.vstack((-A, A))))
    b_ub = np.hstack((-b, b))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=sco.OptimizeWarning)
        res = sco.linprog(c, A_ub, b_ub)
    if res.success is False:
        raise ValueError('scipy.optimize.linprog failed with status %d' %
                         res.status)
    return res.x[M:]



@renamed_function(depname='lstmaxdev', depmod='sporco.util')
def lstmaxdev(A, b):
    r"""Least maximum deviation (least maximum error) linear regression.

    Solve the linear regression problem

    .. math::
      \mathrm{argmin}_\mathbf{x} \; \left\| A \mathbf{x} - \mathbf{b}
      \right\|_{\infty} \;\;.

    The interface is similar to that of :func:`numpy.linalg.lstsq` in
    that `np.linalg.lstsq(A, b)` solves the same linear regression
    problem, but with a least squares rather than a least maximum
    error objective. Unlike :func:`numpy.linalg.lstsq`, `b` is required
    to be a 1-d array. The solution is obtained via `mapping to a linear
    program <https://stats.stackexchange.com/a/12564>`__.

    Parameters
    ----------
    A : (M, N) array_like
      Regression coefficient matrix.
    b : (M,) array_like
      Regression ordinate / dependent variable

    Returns
    -------
    x : (N,) ndarray
      Least maximum deviation solution
    """

    M, N = A.shape
    c = np.zeros((N + 1,))
    c[0] = 1.0
    one = np.ones((M, 1))
    A_ub = np.hstack((np.vstack((-one, -one)), np.vstack((-A, A))))
    b_ub = np.hstack((-b, b))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=sco.OptimizeWarning)
        res = sco.linprog(c, A_ub, b_ub)
    if res.success is False:
        raise ValueError('scipy.optimize.linprog failed with status %d' %
                         res.status)
    return res.x[1:]

