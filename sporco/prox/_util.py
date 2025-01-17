# -*- coding: utf-8 -*-
# Copyright (C) 2015-2025 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Utility functions for the prox module"""

from __future__ import division
from builtins import range

import numpy as np


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def ndto2d(x, axis=-1):
    """Convert a multi-dimensional array into a 2d array, with the axes
    specified by the `axis` parameter flattened into an index along
    rows, and the remaining axes flattened into an index along the
    columns. This operation can not be properly achieved by a simple
    reshape operation since a reshape would shuffle element order if
    the axes to be grouped together were not consecutive: this is
    avoided by first permuting the axes so that the grouped axes are
    consecutive.


    Parameters
    ----------
    x : array_like
      Multi-dimensional input array
    axis : int or tuple of ints, optional (default -1)
      Axes of `x` to be grouped together to form the rows of the output
      2d array.

    Returns
    -------
    xtr : ndarray
      2D output array
    rsi : tuple
      A tuple containing the details of transformation applied in the
      conversion to 2D
    """

    # Convert int axis into a tuple
    if isinstance(axis, int):
        axis = (axis,)
    # Handle negative axis indices
    axis = tuple([k if k >= 0 else x.ndim + k for k in axis])
    # Complement of axis set on full set of axes of input v
    caxis = tuple(set(range(x.ndim)) - set(axis))
    # Permute axes of x (generalised transpose) so that axes over
    # which operation is to be applied are all at the end
    prm = caxis + axis
    xt = np.transpose(x, axes=prm)
    xts = xt.shape
    # Reshape into a 2D array with the axes specified by the axis
    # parameter flattened into an index along rows, and the remaining
    # axes flattened into an index along the columns
    xtr = xt.reshape((np.prod(xts[0:len(caxis)]), -1))
    # Return reshaped array and a tuple containing the information
    # necessary to undo the entire operation
    return xtr, (xts, prm)



def ndfrom2d(xtr, rsi):
    """Undo the array shape conversion applied by :func:`ndto2d`,
    returning the input 2D array to its original shape.


    Parameters
    ----------
    xtr : array_like
      Two-dimensional input array
    rsi : tuple
      A tuple containing the shape of the axis-permuted array and the
      permutation order applied in :func:`ndto2d`.

    Returns
    -------
    x : ndarray
      Multi-dimensional output array
    """

    # Extract components of conversion information tuple
    xts = rsi[0]
    prm = rsi[1]
    # Reshape x to the shape obtained after permuting axes in ndto2d
    xt = xtr.reshape(xts)
    # Undo axis permutation performed in ndto2d
    x = np.transpose(xt, np.argsort(prm))
    # Return array with shape corresponding to that of the input to ndto2d
    return x
