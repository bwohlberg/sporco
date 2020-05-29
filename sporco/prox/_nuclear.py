# -*- coding: utf-8 -*-
# Copyright (C) 2015-2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Compute the nuclear norm and its proximal operator"""

from __future__ import division

import numpy as np

from sporco.array import promote16


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def norm_nuclear(X):
    r"""Compute the nuclear norm

    .. math::
      \| X \|_* = \sum_i \sigma_i

    where :math:`\sigma_i` are the singular values of matrix :math:`X`.


    Parameters
    ----------
    X : array_like
      Input array :math:`X`

    Returns
    -------
    nncl : float
      Nuclear norm of `X`
    """

    return np.sum(np.linalg.svd(promote16(X), compute_uv=False))



def prox_nuclear(V, alpha):
    r"""Proximal operator of the nuclear norm :cite:`cai-2010-singular`
    with parameter :math:`\alpha`.


    Parameters
    ----------
    v : array_like
      Input array :math:`V`
    alpha : float
      Parameter :math:`\alpha`

    Returns
    -------
    X : ndarray
      Output array
    s : ndarray
      Singular values of `X`
    """

    Usvd, s, Vsvd = promote16(V, fn=np.linalg.svd, full_matrices=False)
    ss = np.maximum(0, s - alpha)
    return np.dot(Usvd, np.dot(np.diag(ss), Vsvd)), ss
