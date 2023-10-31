# -*- coding: utf-8 -*-
# Copyright (C) 2015-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Functions operating on numpy arrays etc."""

from __future__ import division
from builtins import range

import collections
from future.moves.itertools import zip_longest

import numpy as np


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


def ntpl2array(ntpl):
    """Convert a namedtuple to an array.

    Convert a :func:`collections.namedtuple` object to a
    :class:`numpy.ndarray` object that can be saved using
    :func:`numpy.savez`.

    Parameters
    ----------
    ntpl : collections.namedtuple object
      Named tuple object to be converted to ndarray

    Returns
    -------
    arr : ndarray
      Array representation of input named tuple
    """

    return np.asarray((np.hstack([col for col in ntpl]), ntpl._fields,
                       ntpl.__class__.__name__), dtype=object)



def array2ntpl(arr):
    """Convert an array representation of a namedtuple back to a namedtuple.

    Convert a :class:`numpy.ndarray` object constructed by
    :func:`ntpl2array` back to the original
    :func:`collections.namedtuple` representation.

    Parameters
    ----------
    arr : ndarray
      Array representation of named tuple constructed by :func:`ntpl2array`

    Returns
    -------
    ntpl : collections.namedtuple object
      Named tuple object with the same name and fields as the original
      named typle object provided to :func:`ntpl2array`
    """

    cls = collections.namedtuple(arr[2], arr[1])
    return cls(*tuple(arr[0]))



def transpose_ntpl_list(lst):
    """Transpose a list of named tuple objects (of the same type) into a
    named tuple of lists.

    Parameters
    ----------
    lst : list of collections.namedtuple object
      List of named tuple objects of the same type

    Returns
    -------
    ntpl : collections.namedtuple object
      Named tuple object with each entry consisting of a list of the
      corresponding fields of the named tuple objects in list ``lst``
    """

    if not lst:
        return None
    else:
        cls = collections.namedtuple(lst[0].__class__.__name__,
                                     lst[0]._fields)
        return cls(*[[lst[k][l] for k in range(len(lst))]
                     for l in range(len(lst[0]))])



def zpad(x, pd, ax):
    """Zero-pad array `x` with `pd = (leading, trailing)` zeros on axis `ax`.

    Parameters
    ----------
    x : array_like
      Array to be padded
    pd : tuple
      Sequence of two ints (leading,trailing) specifying number of zeros
      for padding
    ax : int
      Axis to be padded

    Returns
    -------
    xp : array_like
      Padded array
    """

    xpd = ((0, 0),)*ax + (pd,) + ((0, 0),)*(x.ndim-ax-1)
    return np.pad(x, xpd, 'constant')



def zdivide(x, y):
    """Return `x`/`y`, with 0 instead of NaN where `y` is 0.

    Parameters
    ----------
    x : array_like
      Numerator
    y : array_like
      Denominator

    Returns
    -------
    z : ndarray
      Quotient `x`/`y`
    """

    # See https://stackoverflow.com/a/37977222
    return np.divide(x, y, out=np.zeros_like(x), where=(y != 0))



def promote16(u, fn=None, *args, **kwargs):
    r"""Promote ``np.float16`` arguments to ``np.float32`` dtype.

    Utility function for use with functions that do not support arrays
    of dtype ``np.float16``. This function has two distinct modes of
    operation. If called with only the `u` parameter specified, the
    returned value is either `u` itself if `u` is not of dtype
    ``np.float16``, or `u` promoted to ``np.float32`` dtype if it is. If
    the function parameter `fn` is specified then `u` is conditionally
    promoted as described above, passed as the first argument to
    function `fn`, and the returned values are converted back to dtype
    ``np.float16`` if `u` is of that dtype. Note that if parameter `fn`
    is specified, it may not be be specified as a keyword argument if it
    is followed by any non-keyword arguments.

    Parameters
    ----------
    u : array_like
      Array to be promoted to np.float32 if it is of dtype ``np.float16``
    fn : function or None, optional (default None)
      Function to be called with promoted `u` as first parameter and
      \*args and \*\*kwargs as additional parameters
    *args
      Variable length list of arguments for function `fn`
    **kwargs
      Keyword arguments for function `fn`

    Returns
    -------
    up : ndarray
      Conditionally dtype-promoted version of `u` if `fn` is None,
      or value(s) returned by `fn`, converted to the same dtype as `u`,
      if `fn` is a function
    """

    dtype = np.float32 if u.dtype == np.float16 else u.dtype
    up = np.asarray(u, dtype=dtype)
    if fn is None:
        return up
    else:
        v = fn(up, *args, **kwargs)
        if isinstance(v, tuple):
            vp = tuple([np.asarray(vk, dtype=u.dtype) for vk in v])
        else:
            vp = np.asarray(v, dtype=u.dtype)
        return vp



def atleast_nd(n, u):
    """Append axes to an array so that it is ``n`` dimensional.

    If the input array has fewer than ``n`` dimensions, append singleton
    dimensions so that it is ``n`` dimensional. Note that the interface
    differs substantially from that of :func:`numpy.atleast_3d` etc.

    Parameters
    ----------
    n : int
      Minimum number of required dimensions
    u : array_like
      Input array

    Returns
    -------
    v : ndarray
      Output array with at least `n` dimensions
    """

    if u.ndim >= n:
        return u
    else:
        return u.reshape(u.shape + (1,)*(n-u.ndim))



def split(u, axis=0):
    """Split an array into a list of arrays on the specified axis.

    Split an array into a list of arrays on the specified axis. The
    length of the list is the shape of the array on the specified axis,
    and the corresponding axis is removed from each entry in the list.
    This function does not have the same behaviour as :func:`numpy.split`.

    Parameters
    ----------
    u : array_like
      Input array
    axis : int, optional (default 0)
      Axis on which to split the input array

    Returns
    -------
    v : list of ndarray
      List of arrays
    """

    # Convert negative axis to positive
    if axis < 0:
        axis = u.ndim + axis

    # Construct axis selection slice
    slct0 = (slice(None),) * axis
    return [u[slct0 + (k,)] for k in range(u.shape[axis])]



def rolling_window(x, wsz, wnm=None, pad='wrap'):
    """Construct a rolling window view of the input array.

    Use :func:`numpy.lib.stride_tricks.as_strided` to construct a view
    of the input array that represents different positions of a rolling
    window as additional axes of the array. If the number of shifts
    requested is such that the window extends beyond the boundary of the
    input array, it is padded before the view is constructed. For
    example, if ``x`` is 4 x 5 array, the output of
    ``y = rolling_window(x, (3, 3))`` is a 3 x 3 x 2 x 3 array, with the
    first window position indexed as ``y[..., 0, 0]``.

    Parameters
    ----------
    x : ndarray
      Input array
    wsz : tuple
      Window size
    wnm : tuple, optional (default None)
      Number of shifts of window on each axis. If None, the number of
      shifts is set so that the end sample in the array is also the end
      sample in the final window position.
    pad : string, optional (default 'wrap')
      A pad mode specification for :func:`numpy.pad`

    Returns
    -------
    xw : ndarray
      An array of shape wsz + wnm representing all requested shifts of
      the window within the input array
    """

    if wnm is None:
        wnm = tuple(np.array(x.shape) - np.array(wsz) + 1)
    else:
        over = np.clip(np.array(wsz) + np.array(wnm) - np.array(x.shape) - 1,
                       0, np.iinfo(int).max)
        if np.any(over > 0):
            psz = [(0, p) for p in over]
            x = np.pad(x, psz, mode=pad)
    outsz = wsz + wnm
    outstrd = x.strides + x.strides
    return np.lib.stride_tricks.as_strided(x, outsz, outstrd)



def subsample_array(x, step, pad=False, mode='reflect'):
    """Construct a subsampled view of the input array.

    Use :func:`numpy.lib.stride_tricks.as_strided` to construct a view
    of the input array that represents a subsampling of the array by the
    specified step, with different offsets of the subsampling as
    additional axes of the array. If the input array shape is not evenly
    divisible by the subsampling step, it is padded before the view
    is constructed. For example, if ``x`` is 6 x 6 array, the output of
    ``y = subsample_array(x, (2, 2))`` is a 2 x 2 x 3 x 3 array, with
    the first subsampling offset indexed as ``y[0, 0]``.

    Parameters
    ----------
    x : ndarray
      Input array
    step : tuple
      Subsampling step size
    pad : bool, optional (default False)
      Flag indicating whether the input array should be padded
      when its size is not integer divisible by the step size
    mode : string, optional (default 'reflect')
      A pad mode specification for :func:`numpy.pad`

    Returns
    -------
    xs : ndarray
      An array representing different subsampling offsets in the input
      array
    """

    if np.any(np.greater_equal(step, x.shape)):
        raise ValueError('Step size must be less than array size on each axis')
    sbsz, dvmd = np.divmod(x.shape, step)
    if pad and np.any(dvmd):
        sbsz += np.clip(dvmd, 0, 1)
        psz = np.subtract(np.multiply(sbsz, step), x.shape)
        pdt = [(0, p) for p in psz]
        x = np.pad(x, pdt, mode=mode)
    outsz = step + tuple(sbsz)
    outstrd = x.strides + tuple(np.multiply(step, x.strides))
    return np.lib.stride_tricks.as_strided(x, outsz, outstrd)



def extract_blocks(img, blksz, stpsz=None):
    """Extract blocks from an ndarray signal into an ndarray.

    Parameters
    ----------
    img : ndarray or tuple of ndarrays
      nd array of images, or tuple of images
    blksz : tuple
      tuple of block sizes, blocks are taken starting from the first index
      of img
    stpsz : tuple, optional (default None, corresponds to steps of 1)
      tuple of step sizes between neighboring blocks

    Returns
    -------
    blks : ndarray
      image blocks
    """

    # See http://stackoverflow.com/questions/16774148 and
    # sklearn.feature_extraction.image.extract_patches_2d
    if isinstance(img, tuple):
        img = np.stack(img, axis=-1)

    if stpsz is None:
        stpsz = (1,) * len(blksz)

    imgsz = img.shape

    # Calculate the number of blocks that can fit in each dimension of
    # the images
    numblocks = tuple(int(np.floor((a - b) / c) + 1) for a, b, c in
                      zip_longest(imgsz, blksz, stpsz, fillvalue=1))

    # Calculate the strides for blocks
    blockstrides = tuple(a * b for a, b in zip_longest(img.strides, stpsz,
                                                       fillvalue=1))

    new_shape = blksz + numblocks
    new_strides = img.strides[:len(blksz)] + blockstrides
    blks = np.lib.stride_tricks.as_strided(img, new_shape, new_strides)
    return np.reshape(blks, blksz + (-1,))



def average_blocks(blks, imgsz, stpsz=None):
    """Average blocks together from an ndarray to reconstruct ndarray signal.

    Parameters
    ----------
    blks : ndarray
      Array of blocks of a signal
    imgsz : tuple
      Tuple of the signal size
    stpsz : tuple, optional (default None, corresponds to steps of 1)
      Tuple of step sizes between neighboring blocks

    Returns
    -------
    imgs : ndarray
      Reconstructed signal, unknown pixels are returned as np.nan
    """

    blksz = blks.shape[:-1]

    if stpsz is None:
        stpsz = tuple(1 for _ in blksz)


    # Calculate the number of blocks that can fit in each dimension of
    # the images
    numblocks = tuple(int(np.floor((a-b)/c)+1) for a, b, c in
                      zip_longest(imgsz, blksz, stpsz, fillvalue=1))

    new_shape = blksz + numblocks
    blks = np.reshape(blks, new_shape)

    # Construct an imgs matrix of empty lists
    imgs = np.zeros(imgsz, dtype=blks.dtype)
    normalizer = np.zeros(imgsz, dtype=blks.dtype)

    # Iterate over each block and append the values to the corresponding
    # imgs cell
    for pos in np.ndindex(numblocks):
        slices = tuple(slice(a*c, a*c+b) for a, b, c in
                       zip(pos, blksz, stpsz))
        imgs[slices+pos[len(blksz):]] += blks[(Ellipsis, )+pos]
        normalizer[slices+pos[len(blksz):]] += blks.dtype.type(1)

    return np.where(normalizer > 0, (imgs/normalizer).astype(blks.dtype),
                    np.nan)



def combine_blocks(blks, imgsz, stpsz=None, fn=np.median):
    """Combine blocks from an ndarray to reconstruct ndarray signal.

    Parameters
    ----------
    blks : ndarray
      Array of blocks of a signal
    imgsz : tuple
      Tuple of the signal size
    stpsz : tuple, optional (default None, corresponds to steps of 1)
      Tuple of step sizes between neighboring blocks
    fn : function, optional (default np.median)
      Function used to resolve multivalued cells

    Returns
    -------
    imgs : ndarray
      Reconstructed signal, unknown pixels are returned as np.nan
    """

    # Construct a vectorized append function
    def listapp(x, y):
        x.append(y)
    veclistapp = np.vectorize(listapp, otypes=[np.object_])

    blksz = blks.shape[:-1]

    if stpsz is None:
        stpsz = tuple(1 for _ in blksz)

    # Calculate the number of blocks that can fit in each dimension of
    # the images
    numblocks = tuple(int(np.floor((a-b)/c) + 1) for a, b, c in
                      zip_longest(imgsz, blksz, stpsz, fillvalue=1))

    new_shape = blksz + numblocks
    blks = np.reshape(blks, new_shape)

    # Construct an imgs matrix of empty lists
    imgs = np.empty(imgsz, dtype=np.object_)
    imgs.fill([])
    imgs = np.frompyfunc(list, 1, 1)(imgs)

    # Iterate over each block and append the values to the corresponding
    # imgs cell
    for pos in np.ndindex(numblocks):
        slices = tuple(slice(a*c, a*c + b) for a, b, c in
                       zip_longest(pos, blksz, stpsz, fillvalue=1))
        veclistapp(imgs[slices].squeeze(), blks[(Ellipsis, ) + pos].squeeze())

    return np.vectorize(fn, otypes=[blks.dtype])(imgs)
