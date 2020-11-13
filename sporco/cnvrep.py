# -*- coding: utf-8 -*-
# Copyright (C) 2015-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes and functions that support working with convolutional
representations.
"""

from __future__ import division, absolute_import, print_function
from builtins import range

import pprint
import functools
import numpy as np


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



class CSC_ConvRepIndexing(object):
    """Array dimensions and indexing for CSC problems.

    Manage the inference of problem dimensions and the roles of
    :class:`numpy.ndarray` indices for convolutional representations in
    convolutional sparse coding problems (e.g.
    :class:`.admm.cbpdn.ConvBPDN` and related classes).
    """

    def __init__(self, D, S, dimK=None, dimN=2):
        """Initialise a ConvRepIndexing object.

        Initialise a ConvRepIndexing object representing dimensions
        of S (input signal), D (dictionary), and X (coefficient array)
        in a convolutional representation.  These dimensions are
        inferred from the input `D` and `S` as well as from parameters
        `dimN` and `dimK`.  Management and inferrence of these problem
        dimensions is not entirely straightforward because
        :class:`.admm.cbpdn.ConvBPDN` and related classes make use
        *internally* of S, D, and X arrays with a standard layout
        (described below), but *input* `S` and `D` are allowed to
        deviate from this layout for the convenience of the user.

        The most fundamental parameter is `dimN`, which specifies the
        dimensionality of the spatial/temporal samples being
        represented (e.g. `dimN` = 2 for representations of 2D
        images).  This should be common to *input* S and D, and is
        also common to *internal* S, D, and X.  The remaining
        dimensions of input `S` can correspond to multiple channels
        (e.g. for RGB images) and/or multiple signals (e.g. the array
        contains multiple independent images).  If input `S` contains
        two additional dimensions (in addition to the `dimN` spatial
        dimensions), then those are considered to correspond, in
        order, to channel and signal indices.  If there is only a
        single additional dimension, then determination whether it
        represents a channel or signal index is more complicated.  The
        rule for making this determination is as follows:

        * if `dimK` is set to 0 or 1 instead of the default ``None``,
          then that value is taken as the number of signal indices in
          input `S` and any remaining indices are taken as channel
          indices (i.e. if `dimK` = 0 then dimC = 1 and if `dimK` = 1
          then dimC = 0).
        * if `dimK` is ``None`` then the number of channel dimensions is
          determined from the number of dimensions in the input
          dictionary `D`. Input `D` should have at least `dimN` + 1
          dimensions, with the final dimension indexing dictionary
          filters. If it has exactly `dimN` + 1 dimensions then it is a
          single-channel dictionary, and input `S` is also assumed to be
          single-channel, with the additional index in `S` assigned as a
          signal index (i.e. dimK = 1). Conversely, if input `D` has
          `dimN` + 2 dimensions it is a multi-channel dictionary, and
          the additional index in `S` is assigned as a channel index
          (i.e. dimC = 1).

        Note that it is an error to specify `dimK` = 1 if input `S`
        has `dimN` + 1 dimensions and input `D` has `dimN` + 2
        dimensions since a multi-channel dictionary requires a
        multi-channel signal. (The converse is not true: a
        multi-channel signal can be decomposed using a single-channel
        dictionary.)

        The *internal* data layout for S (signal), D (dictionary), and
        X (coefficient array) is (multi-channel dictionary)
        ::

            sptl.          chn  sig  flt
          S(N0,  N1, ...,  C,   K,   1)
          D(N0,  N1, ...,  C,   1,   M)
          X(N0,  N1, ...,  1,   K,   M)

        or (single-channel dictionary)

        ::

            sptl.          chn  sig  flt
          S(N0,  N1, ...,  C,   K,   1)
          D(N0,  N1, ...,  1,   1,   M)
          X(N0,  N1, ...,  C,   K,   M)

        where

        * Nv = [N0, N1, ...] and N = N0 x N1 x ... are the vector of sizes
          of the spatial/temporal indices and the total number of
          spatial/temporal samples respectively
        * C is the number of channels in S
        * K is the number of signals in S
        * M is the number of filters in D

        It should be emphasised that dimC and `dimK` may take on values
        0 or 1, and represent the number of channel and signal
        dimensions respectively *in input S*. In the internal layout
        of S there is always a dimension allocated for channels and
        signals. The number of channel dimensions in input `D` and the
        corresponding size of that index are represented by dimCd
        and Cd respectively.

        Parameters
        ----------
        D : array_like
          Input dictionary
        S : array_like
          Input signal
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions of signal samples
        """

        # Determine whether dictionary is single- or multi-channel
        self.dimCd = D.ndim - (dimN + 1)
        if self.dimCd == 0:
            self.Cd = 1
        else:
            self.Cd = D.shape[-2]

        # Numbers of spatial, channel, and signal dimensions in
        # external S are dimN, dimC, and dimK respectively. These need
        # to be calculated since inputs D and S do not already have
        # the standard data layout above, i.e. singleton dimensions
        # will not be present
        if dimK is None:
            rdim = S.ndim - dimN
            if rdim == 0:
                (dimC, dimK) = (0, 0)
            elif rdim == 1:
                dimC = self.dimCd  # Assume S has same number of channels as D
                dimK = S.ndim - dimN - dimC  # Assign remaining channels to K
            else:
                (dimC, dimK) = (1, 1)
        else:
            dimC = S.ndim - dimN - dimK  # Assign remaining channels to C

        self.dimN = dimN  # Number of spatial dimensions
        self.dimC = dimC  # Number of channel dimensions in S
        self.dimK = dimK  # Number of signal dimensions in S

        # Number of channels in S
        if self.dimC == 1:
            self.C = S.shape[dimN]
        else:
            self.C = 1
        Cx = self.C - self.Cd + 1

        # Ensure that multi-channel dictionaries used with a signal with a
        # matching number of channels
        if self.Cd > 1 and self.C != self.Cd:
            raise ValueError("Multi-channel dictionary with signal with "
                             "mismatched number of channels (Cd=%d, C=%d)" %
                             (self.Cd, self.C))

        # Number of signals in S
        if self.dimK == 1:
            self.K = S.shape[self.dimN + self.dimC]
        else:
            self.K = 1

        # Number of filters
        self.M = D.shape[-1]

        # Shape of spatial indices and number of spatial samples
        self.Nv = S.shape[0:dimN]
        self.N = np.prod(np.array(self.Nv))

        # Axis indices for each component of X and internal S and D
        self.axisN = tuple(range(0, dimN))
        self.axisC = dimN
        self.axisK = dimN + 1
        self.axisM = dimN + 2

        # Shapes of internal S, D, and X
        self.shpD = D.shape[0:dimN] + (self.Cd,) + (1,) + (self.M,)
        self.shpS = self.Nv + (self.C,) + (self.K,) + (1,)
        self.shpX = self.Nv + (Cx,) + (self.K,) + (self.M,)



    def __str__(self):
        """Return string representation of object."""

        return pprint.pformat(vars(self))





class DictionarySize(object):
    """Compute dictionary size parameters.

    Compute dictionary size parameters from a dictionary size
    specification tuple as in the dsz argument of :func:`bcrop`."""

    def __init__(self, dsz, dimN=2):
        """Initialise a DictionarySize object.

        Parameters
        ----------
        dsz : tuple
          Dictionary size specification (using the same format as the
          `dsz` argument of :func:`bcrop`)
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        self.dsz = dsz
        if isinstance(dsz[0], tuple):
            # Multi-scale dictionary specification
            if isinstance(dsz[0][0], tuple):
                self.ndim = len(dsz[0][0])
                self.nchn = 0
                for c in range(0, len(dsz[0])):
                    self.nchn += dsz[0][c][-2]
            else:
                self.ndim = len(dsz[0])
                if self.ndim == dimN + 1:
                    self.nchn = 1
                else:
                    self.nchn = dsz[0][-2]
            mxsz = np.zeros((dimN,), dtype=int)
            self.nflt = 0
            for m in range(0, len(dsz)):
                if isinstance(dsz[m][0], tuple):
                    # Separate channel specification
                    for c in range(0, len(dsz[m])):
                        mxsz = np.maximum(mxsz, np.asarray(dsz[m][c][0:dimN]))
                    self.nflt += dsz[m][0][-1]
                else:
                    # Combined channel specification
                    mxsz = np.maximum(mxsz, np.asarray(dsz[m][0:dimN]))
                    self.nflt += dsz[m][-1]
            self.mxsz = tuple(mxsz)
        else:
            # Single scale dictionary specification
            self.ndim = len(dsz)
            self.mxsz = dsz[0:dimN]
            self.nflt = dsz[-1]
            if self.ndim == dimN + 1:
                self.nchn = 1
            else:
                self.nchn = dsz[-2]



    def __str__(self):
        """Return string representation of object."""

        return pprint.pformat(vars(self))





class CDU_ConvRepIndexing(object):
    """Array dimensions and indexing for CDU problems.

    Manage the inference of problem dimensions and the roles of
    :class:`numpy.ndarray` indices for convolutional representations
    in convolutional dictionary update problems (e.g.
    :class:`.ConvCnstrMODBase` and derived classes).
    """

    def __init__(self, dsz, S, dimK=None, dimN=2):
        """Initialise a ConvRepIndexing object.

        Initialise a ConvRepIndexing object representing dimensions
        of S (input signal), D (dictionary), and X (coefficient array)
        in a convolutional representation. These dimensions are inferred
        from the input `dsz` and `S` as well as from parameters `dimN`
        and `dimK`. Management and inferrence of these problem
        dimensions is not entirely straightforward because
        :class:`.ConvCnstrMODBase` and related classes make use
        *internally* of S, D, and X arrays with a standard layout
        (described below), but *input* `S` and `dsz` are allowed to
        deviate from this layout for the convenience of the user. Note
        that S, D, and X refers to the names of signal, dictionary, and
        coefficient map arrays in :class:`.admm.cbpdn.ConvBPDN`; the
        corresponding variable names in :class:`.ConvCnstrMODBase` are
        S, X, and Z.

        The most fundamental parameter is `dimN`, which specifies the
        dimensionality of the spatial/temporal samples being represented
        (e.g. `dimN` = 2 for representations of 2D images). This should
        be common to *input* `S` and `dsz`, and is also common to
        *internal* S, D, and X. The remaining dimensions of input `S`
        can correspond to multiple channels (e.g. for RGB images) and/or
        multiple signals (e.g. the array contains multiple independent
        images). If input `S` contains two additional dimensions (in
        addition to the `dimN` spatial dimensions), then those are
        considered to correspond, in order, to channel and signal
        indices. If there is only a single additional dimension, then
        determination whether it represents a channel or signal index is
        more complicated. The rule for making this determination is as
        follows:

        * if `dimK` is set to 0 or 1 instead of the default ``None``,
          then that value is taken as the number of signal indices in
          input `S` and any remaining indices are taken as channel
          indices (i.e. if `dimK` = 0 then dimC = 1 and if `dimK` = 1
          then dimC = 0).
        * if `dimK` is ``None`` then the number of channel dimensions
          is determined from the number of dimensions specified in the
          input dictionary size `dsz`. Input `dsz` should specify at
          least `dimN` + 1 dimensions, with the final dimension
          indexing dictionary filters. If it has exactly `dimN` + 1
          dimensions then it is a single-channel dictionary, and input
          `S` is also assumed to be single-channel, with the
          additional index in `S` assigned as a signal index
          (i.e. `dimK` = 1).  Conversely, if input `dsz` specified
          `dimN` + 2 dimensions it is a multi-channel dictionary, and
          the additional index in `S` is assigned as a channel index
          (i.e. dimC = 1).

        Note that it is an error to specify `dimK` = 1 if input `S`
        has `dimN` + 1 dimensions and input `dsz` specified `dimN` + 2
        dimensions since a multi-channel dictionary requires a
        multi-channel signal. (The converse is not true: a
        multi-channel signal can be decomposed using a single-channel
        dictionary.)

        The *internal* data layout for S (signal), D (dictionary), and
        X (coefficient array) is (multi-channel dictionary)
        ::

            sptl.          chn  sig  flt
          S(N0,  N1, ...,  C,   K,   1)
          D(N0,  N1, ...,  C,   1,   M)
          X(N0,  N1, ...,  1,   K,   M)

        or (single-channel dictionary)

        ::

            sptl.          chn  sig  flt
          S(N0,  N1, ...,  C,   K,   1)
          D(N0,  N1, ...,  1,   1,   M)
          X(N0,  N1, ...,  C,   K,   M)

        where

        * Nv = [N0, N1, ...] and N = N0 x N1 x ... are the vector of
          sizes of the spatial/temporal indices and the total number of
          spatial/temporal samples respectively
        * C is the number of channels in S
        * K is the number of signals in S
        * M is the number of filters in D

        It should be emphasised that dimC and dimK may take on values
        0 or 1, and represent the number of channel and signal
        dimensions respectively *in input S*. In the internal layout
        of S there is always a dimension allocated for channels and
        signals. The number of channel dimensions in input `D` and the
        corresponding size of that index are represented by dimCd
        and Cd respectively.

        Parameters
        ----------
        dsz : tuple
          Dictionary size specification (using the same format as the
          `dsz` argument of :func:`bcrop`)
        S : array_like
          Input signal
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions of signal samples
        """

        # Extract properties of dictionary size specification tuple
        ds = DictionarySize(dsz, dimN)
        self.dimCd = ds.ndim - dimN - 1
        self.Cd = ds.nchn
        self.M = ds.nflt
        self.dsz = dsz

        # Numbers of spatial, channel, and signal dimensions in
        # external S are dimN, dimC, and dimK respectively. These need
        # to be calculated since inputs D and S do not already have
        # the standard data layout above, i.e. singleton dimensions
        # will not be present
        if dimK is None:
            rdim = S.ndim - dimN
            if rdim == 0:
                (dimC, dimK) = (0, 0)
            elif rdim == 1:
                dimC = self.dimCd  # Assume S has same number of channels as D
                dimK = S.ndim - dimN - dimC  # Assign remaining channels to K
            else:
                (dimC, dimK) = (1, 1)
        else:
            dimC = S.ndim - dimN - dimK  # Assign remaining channels to C

        self.dimN = dimN  # Number of spatial dimensions
        self.dimC = dimC  # Number of channel dimensions in S
        self.dimK = dimK  # Number of signal dimensions in S

        # Number of channels in S
        if self.dimC == 1:
            self.C = S.shape[dimN]
        else:
            self.C = 1
        self.Cx = self.C - self.Cd + 1

        # Ensure that multi-channel dictionaries used with a signal with a
        # matching number of channels
        if self.Cd > 1 and self.C != self.Cd:
            raise ValueError("Multi-channel dictionary with signal with "
                             "mismatched number of channels (Cd=%d, C=%d)" %
                             (self.Cd, self.C))

        # Number of signals in S
        if self.dimK == 1:
            self.K = S.shape[self.dimN + self.dimC]
        else:
            self.K = 1

        # Shape of spatial indices and number of spatial samples
        self.Nv = S.shape[0:dimN]
        self.N = np.prod(np.array(self.Nv))

        # Axis indices for each component of X and internal S and D
        self.axisN = tuple(range(0, dimN))
        self.axisC = dimN
        self.axisK = dimN + 1
        self.axisM = dimN + 2

        # Shapes of internal S, D, and X
        self.shpD = self.Nv + (self.Cd,) + (1,) + (self.M,)
        self.shpS = self.Nv + (self.C,) + (self.K,) + (1,)
        self.shpX = self.Nv + (self.Cx,) + (self.K,) + (self.M,)



    def __str__(self):
        """Return string representation of object."""

        return pprint.pformat(vars(self))



def stdformD(D, Cd, M, dimN=2):
    """Reshape dictionary array to internal standard form.

    Reshape dictionary array (`D` in :mod:`.admm.cbpdn` module, `X` in
    :mod:`.admm.ccmod` module) to internal standard form.

    Parameters
    ----------
    D : array_like
      Dictionary array
    Cd : int
      Size of dictionary channel index
    M : int
      Number of filters in dictionary
    dimN : int, optional (default 2)
      Number of problem spatial indices

    Returns
    -------
    Dr : ndarray
      Reshaped dictionary array
    """

    return D.reshape(D.shape[0:dimN] + (Cd,) + (1,) + (M,))



def l1Wshape(W, cri):
    r"""Get internal shape for an :math:`\ell_1` norm weight array.

    Get appropriate internal shape (see
    :class:`CSC_ConvRepIndexing`) for an :math:`\ell_1` norm weight
    array `W`, as in option ``L1Weight`` in
    :class:`.admm.cbpdn.ConvBPDN.Options` and related options classes.
    The external shape of `W` depends on the external shape of input
    data array `S` and the size of the final axis (i.e. the number of
    filters) in dictionary array `D`.  The internal shape of the
    weight array `W` is required to be compatible for multiplication
    with the internal sparse representation array `X`.  The simplest
    criterion for ensuring that the external `W` is compatible with
    `S` is to ensure that `W` has shape ``S.shape + D.shape[-1:]``,
    except that non-singleton dimensions may be replaced with
    singleton dimensions.  If `W` has a single additional axis that is
    neither a spatial axis nor a filter axis, it is assigned as a
    channel or multi-signal axis depending on the corresponding
    assignement in `S`.

    Parameters
    ----------
    W : array_like
      Weight array
    cri : :class:`CSC_ConvRepIndexing` object
      Object specifying convolutional representation dimensions

    Returns
    -------
    shp : tuple
      Appropriate internal weight array shape
    """

    # Number of dimensions in input array `S`
    sdim = cri.dimN + cri.dimC + cri.dimK

    if W.ndim < sdim:
        if W.size == 1:
            # Weight array is a scalar
            shpW = (1,) * (cri.dimN + 3)
        else:
            # Invalid weight array shape
            raise ValueError('weight array must be scalar or have at least '
                             'the same number of dimensions as input array')
    elif W.ndim == sdim:
        # Weight array has the same number of dimensions as the input array
        shpW = W.shape + (1,) * (3 - cri.dimC - cri.dimK)
    else:
        # Weight array has more dimensions than the input array
        if W.ndim == cri.dimN + 3:
            # Weight array is already of the appropriate shape
            shpW = W.shape
        else:
            # Assume that the final axis in the input array is the filter
            # index
            shpW = W.shape[0:-1] + (1,) * (2 - cri.dimC - cri.dimK) + \
                W.shape[-1:]

    return shpW



def mskWshape(W, cri):
    """Get internal shape for a data fidelity term mask array.

    Get appropriate internal shape (see
    :class:`CSC_ConvRepIndexing` and :class:`CDU_ConvRepIndexing`) for
    data fidelity term mask array `W`. The external shape of `W`
    depends on the external shape of input data array `S`.  The
    simplest criterion for ensuring that the external `W` is
    compatible with `S` is to ensure that `W` has the same shape as
    `S`, except that non-singleton dimensions in `S` may be singleton
    dimensions in `W`. If `W` has a single non-spatial axis, it is
    assigned as a channel or multi-signal axis depending on the
    corresponding assignement in `S` (if `S` has non-singleton channel
    and signal axes, the single non-spatial axis in `W` is taken as a
    channel axis).

    Parameters
    ----------
    W : array_like
      Data fidelity term weight/mask array
    cri : :class:`CSC_ConvRepIndexing` object or \
    :class:`CDU_ConvRepIndexing` object
      Object specifying convolutional representation dimensions

    Returns
    -------
    shp : tuple
      Appropriate internal mask array shape
    """

    # Number of axes in W available for C and/or K axes
    ckdim = W.ndim - cri.dimN
    if ckdim >= 2:
        # Both C and K axes are present in W
        shpW = W.shape + (1,) if ckdim == 2 else W.shape
    elif ckdim == 1:
        # Exactly one of C or K axes is present in W
        if cri.C == 1 and cri.K > 1:
            # Input S has a single channel and multiple signals
            shpW = W.shape[0:cri.dimN] + (1, W.shape[cri.dimN]) + (1,)
        elif cri.C > 1 and cri.K == 1:
            # Input S has multiple channels and a single signal
            shpW = W.shape[0:cri.dimN] + (W.shape[cri.dimN], 1) + (1,)
        else:
            # Input S has multiple channels and signals: resolve ambiguity
            # by taking extra axis in W as a channel axis
            shpW = W.shape[0:cri.dimN] + (W.shape[cri.dimN], 1) + (1,)
    else:
        # Neither C nor K axis is present in W
        shpW = W.shape + (1,) * (3 - ckdim)

    return shpW



def zeromean(v, dsz, dimN=2):
    """Subtract mean value from each filter in the input array `v`.

    The `dsz` parameter specifies the support sizes of each filter using
    the same format as the `dsz` parameter of :func:`bcrop`. Support
    sizes must be taken into account to ensure that the mean values are
    computed over the correct number of samples, ignoring the
    zero-padded region in which the filter is embedded.

    Parameters
    ----------
    v : array_like
      Input dictionary array
    dsz : tuple
      Filter support size(s)
    dimN : int, optional (default 2)
      Number of spatial dimensions

    Returns
    -------
    vz : ndarray
      Dictionary array with filter means subtracted
    """

    vz = v.copy()
    if isinstance(dsz[0], tuple):
        # Multi-scale dictionary specification
        axisN = tuple(range(0, dimN))
        m0 = 0  # Initial index of current block of equi-sized filters
        # Iterate over distinct filter sizes
        for mb in range(0, len(dsz)):
            # Determine end index of current block of filters
            if isinstance(dsz[mb][0], tuple):
                m1 = m0 + dsz[mb][0][-1]
                c0 = 0  # Init. idx. of current chnl-block of equi-sized flt.
                for cb in range(0, len(dsz[mb])):
                    c1 = c0 + dsz[mb][cb][-2]
                    # Construct slice corresponding to cropped part of
                    # current block of filters in output array and set from
                    # input array
                    cbslc = tuple([slice(0, x) for x in dsz[mb][cb][0:dimN]]
                                  ) + (slice(c0, c1),) + (Ellipsis,) + \
                                      (slice(m0, m1),)
                    vz[cbslc] -= np.mean(v[cbslc], axisN)
                    c0 = c1  # Update initial index for start of next block
            else:
                m1 = m0 + dsz[mb][-1]
                # Construct slice corresponding to cropped part of
                # current block of filters in output array and set from
                # input array
                mbslc = tuple([slice(0, x) for x in dsz[mb][0:-1]]
                              ) + (Ellipsis,) + (slice(m0, m1),)
                vz[mbslc] -= np.mean(v[mbslc], axisN)
            m0 = m1  # Update initial index for start of next block
    else:
        # Single scale dictionary specification
        axisN = tuple(range(0, dimN))
        axnslc = tuple([slice(0, x) for x in dsz[0:dimN]])
        vz[axnslc] -= np.mean(v[axnslc], axisN)

    return vz



def normalise(v, dimN=2):
    r"""Normalise vector components of input array.

    Normalise vectors, corresponding to slices along specified number
    of initial spatial dimensions of an array, to have unit
    :math:`\ell_2` norm. The remaining axes enumerate the distinct
    vectors to be normalised.

    Parameters
    ----------
    v : array_like
      Array with components to be normalised
    dimN : int, optional (default 2)
      Number of initial dimensions over which norm should be computed

    Returns
    -------
    vnrm : ndarray
      Normalised array
    """

    axisN = tuple(range(0, dimN))
    if np.isrealobj(v):
        vn = np.sqrt(np.sum(v**2, axisN, keepdims=True))
    else:
        vn = np.sqrt(np.sum(np.abs(v)**2, axisN, keepdims=True))
    vn[vn == 0] = 1.0
    return np.asarray(v / vn, dtype=v.dtype)



def zpad(v, Nv):
    """Zero-pad initial axes of array to specified size.

    Padding is applied to the right, top, etc. of the array indices.

    Parameters
    ----------
    v : array_like
      Array to be padded
    Nv : tuple
      Sizes to which each of initial indices should be padded

    Returns
    -------
    vp : ndarray
      Padded array
    """

    vp = np.zeros(Nv + v.shape[len(Nv):], dtype=v.dtype)
    axnslc = tuple([slice(0, x) for x in v.shape])
    vp[axnslc] = v
    return vp



def bcrop(v, dsz, dimN=2):
    """Crop dictionary array to specified size.

    Crop specified number of initial spatial dimensions of dictionary
    array to specified size. Parameter `dsz` must be a tuple having one
    of the following forms (the examples assume two spatial/temporal
    dimensions). If all filters are of the same size, then

    ::

      (flt_rows, filt_cols, num_filt)

    may be used when the dictionary has a single channel, and

    ::

      (flt_rows, filt_cols, num_chan, num_filt)

    should be used for a multi-channel dictionary. If the filters are
    not all of the same size, then

    ::

      (
       (flt_rows1, filt_cols1, num_filt1),
       (flt_rows2, filt_cols2, num_filt2),
       ...
      )

    may be used for a single-channel dictionary. A multi-channel
    dictionary may be specified in the form

    ::

      (
       (flt_rows1, filt_cols1, num_chan, num_filt1),
       (flt_rows2, filt_cols2, num_chan, num_filt2),
       ...
      )

    or

    ::

      (
       (
        (flt_rows11, filt_cols11, num_chan11, num_filt1),
        (flt_rows21, filt_cols21, num_chan21, num_filt1),
        ...
       )
       (
        (flt_rows12, filt_cols12, num_chan12, num_filt2),
        (flt_rows22, filt_cols22, num_chan22, num_filt2),
        ...
       )
       ...
      )

    depending on whether the filters for each channel are of the same
    size or not. The total number of dictionary filters, is either
    num_filt in the first two forms, or the sum of num_filt1,
    num_filt2, etc. in the other form. If the filters are not
    two-dimensional, then the dimensions above vary accordingly, i.e.,
    there may be fewer or more filter spatial dimensions than
    flt_rows, filt_cols, e.g.

    ::

      (flt_rows, num_filt)

    for one-dimensional signals, or

    ::

      (flt_rows, filt_cols, filt_planes, num_filt)

    for three-dimensional signals.

    Parameters
    ----------
    v : array_like
      Dictionary array to be cropped
    dsz : tuple
      Filter support size(s)
    dimN : int, optional (default 2)
      Number of spatial dimensions

    Returns
    -------
    vc : ndarray
      Cropped dictionary array
    """

    if isinstance(dsz[0], tuple):
        # Multi-scale dictionary specification
        maxsz = np.zeros((dimN,), dtype=int)  # Max. support size
        # Iterate over dsz to determine max. support size
        for mb in range(0, len(dsz)):
            if isinstance(dsz[mb][0], tuple):
                for cb in range(0, len(dsz[mb])):
                    maxsz = np.maximum(maxsz, np.asarray(dsz[mb][cb][0:dimN]))
            else:
                maxsz = np.maximum(maxsz, np.asarray(dsz[mb][0:dimN]))
        # Init. cropped array
        vc = np.zeros(maxsz.tolist() + list(v.shape[dimN:]), dtype=v.dtype)
        m0 = 0  # Initial index of current block of equi-sized filters
        # Iterate over distinct filter sizes
        for mb in range(0, len(dsz)):
            # Determine end index of current block of filters
            if isinstance(dsz[mb][0], tuple):
                m1 = m0 + dsz[mb][0][-1]
                c0 = 0  # Init. idx. of current chnl-block of equi-sized flt.
                for cb in range(0, len(dsz[mb])):
                    c1 = c0 + dsz[mb][cb][-2]
                    # Construct slice corresponding to cropped part of
                    # current block of filters in output array and set from
                    # input array
                    cbslc = tuple([slice(0, x) for x in dsz[mb][cb][0:dimN]]
                                  ) + (slice(c0, c1),) + (Ellipsis,) + \
                                      (slice(m0, m1),)
                    vc[cbslc] = v[cbslc]
                    c0 = c1  # Update initial index for start of next block
            else:
                m1 = m0 + dsz[mb][-1]
                # Construct slice corresponding to cropped part of
                # current block of filters in output array and set from
                # input array
                mbslc = tuple([slice(0, x) for x in dsz[mb][0:-1]]
                              ) + (Ellipsis,) + (slice(m0, m1),)
                vc[mbslc] = v[mbslc]
            m0 = m1  # Update initial index for start of next block
        return vc
    else:
        # Single scale dictionary specification
        axnslc = tuple([slice(0, x) for x in dsz[0:dimN]])
        return v[axnslc]



def Pcn(x, dsz, Nv, dimN=2, dimC=1, crp=False, zm=False):
    """Constraint set projection for convolutional dictionary update
    problem.

    Parameters
    ----------
    x  : array_like
      Input array
    dsz : tuple
      Filter support size(s), specified using the same format as the `dsz`
      parameter of :func:`bcrop`
    Nv : tuple
      Sizes of problem spatial indices
    dimN : int, optional (default 2)
      Number of problem spatial indices
    dimC : int, optional (default 1)
      Number of problem channel indices
    crp : bool, optional (default False)
      Flag indicating whether the result should be cropped to the support
      of the largest filter in the dictionary.
    zm : bool, optional (default False)
      Flag indicating whether the projection function should include
      filter mean subtraction

    Returns
    -------
    y : ndarray
      Projection of input onto constraint set
    """

    if crp:
        def zpadfn(x):
            return x
    else:
        def zpadfn(x):
            return zpad(x, Nv)

    if zm:
        def zmeanfn(x):
            return zeromean(x, dsz, dimN)
    else:
        def zmeanfn(x):
            return x

    return normalise(zmeanfn(zpadfn(bcrop(x, dsz, dimN))), dimN + dimC)



def getPcn(dsz, Nv, dimN=2, dimC=1, crp=False, zm=False):
    """Construct the constraint set projection function for convolutional
    dictionary update problem.

    Parameters
    ----------
    dsz : tuple
      Filter support size(s), specified using the same format as the `dsz`
      parameter of :func:`bcrop`
    Nv : tuple
      Sizes of problem spatial indices
    dimN : int, optional (default 2)
      Number of problem spatial indices
    dimC : int, optional (default 1)
      Number of problem channel indices
    crp : bool, optional (default False)
      Flag indicating whether the result should be cropped to the support
      of the largest filter in the dictionary.
    zm : bool, optional (default False)
      Flag indicating whether the projection function should include
      filter mean subtraction

    Returns
    -------
    fn : function
      Constraint set projection function
    """

    fncdict = {(False, False): _Pcn,
               (False, True): _Pcn_zm,
               (True, False): _Pcn_crp,
               (True, True): _Pcn_zm_crp}
    fnc = fncdict[(crp, zm)]
    return functools.partial(fnc, dsz=dsz, Nv=Nv, dimN=dimN, dimC=dimC)



def _Pcn(x, dsz, Nv, dimN=2, dimC=1):
    """Dictionary support projection and normalisation.

    Projection onto dictionary update constraint set: support
    projection and normalisation. The result has the full spatial
    dimensions of the input.

    Parameters
    ----------
    x  : array_like
       Input array
    dsz : tuple
      Filter support size(s), specified using the same format as the
      `dsz` parameter of :func:`bcrop`
    Nv : tuple
      Sizes of problem spatial indices
    dimN : int, optional (default 2)
      Number of problem spatial indices
    dimC : int, optional (default 1)
      Number of problem channel indices

    Returns
    -------
    y : ndarray
      Projection of input onto constraint set
    """

    return normalise(zpad(bcrop(x, dsz, dimN), Nv), dimN + dimC)



def _Pcn_zm(x, dsz, Nv, dimN=2, dimC=1):
    """Dictionary support projection, mean subtraction, and normalisation.

    Projection onto dictionary update constraint set: support projection,
    mean subtraction, and normalisation. The result has the full spatial
    dimensions of the input.

    Parameters
    ----------
    x  : array_like
       Input array
    dsz : tuple
      Filter support size(s), specified using the same format as the
      `dsz` parameter of :func:`bcrop`
    Nv : tuple
      Sizes of problem spatial indices
    dimN : int, optional (default 2)
      Number of problem spatial indices
    dimC : int, optional (default 1)
      Number of problem channel indices

    Returns
    -------
    y : ndarray
      Projection of input onto constraint set
    """

    return normalise(zeromean(zpad(bcrop(x, dsz, dimN), Nv), dsz), dimN + dimC)



def _Pcn_crp(x, dsz, Nv, dimN=2, dimC=1):
    """Dictionary support projection and normalisation (cropped).

    Projection onto dictionary update constraint set: support
    projection and normalisation. The result is cropped to the
    support of the largest filter in the dictionary.

    Parameters
    ----------
    x  : array_like
       Input array
    dsz : tuple
      Filter support size(s), specified using the same format as the
      `dsz` parameter of :func:`bcrop`
    Nv : tuple
      Sizes of problem spatial indices
    dimN : int, optional (default 2)
      Number of problem spatial indices
    dimC : int, optional (default 1)
      Number of problem channel indices

    Returns
    -------
    y : ndarray
      Projection of input onto constraint set
    """

    return normalise(bcrop(x, dsz, dimN), dimN + dimC)



def _Pcn_zm_crp(x, dsz, Nv, dimN=2, dimC=1):
    """Dictionary support projection, mean subtraction, and normalisation
    (cropped).

    Projection onto dictionary update constraint set: support
    projection, mean subtraction, and normalisation. The result is
    cropped to the support of the largest filter in the dictionary.

    Parameters
    ----------
    x  : array_like
       Input array
    dsz : tuple
      Filter support size(s), specified using the same format as the
      `dsz` parameter of :func:`bcrop`.
    Nv : tuple
      Sizes of problem spatial indices
    dimN : int, optional (default 2)
      Number of problem spatial indices
    dimC : int, optional (default 1)
      Number of problem channel indices

    Returns
    -------
    y : ndarray
      Projection of input onto constraint set
    """

    return normalise(zeromean(bcrop(x, dsz, dimN), dsz, dimN), dimN + dimC)
