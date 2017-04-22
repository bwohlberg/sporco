# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""ADMM algorithm for the CCMOD problem"""

from __future__ import division
from __future__ import absolute_import
from builtins import range

import copy
import pprint
import numpy as np
from scipy import linalg

from sporco.admm import admm
import sporco.linalg as sl

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class DictionarySize(object):
    """Compute dictionary size parameters from a dictionary size
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
                        mxsz = np.maximum(mxsz, dsz[m][c][0:dimN])
                    self.nflt += dsz[m][0][-1]
                else:
                    # Combined channel specification
                    mxsz = np.maximum(mxsz, dsz[m][0:dimN])
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





class ConvRepIndexing(object):
    """Manage the inference of problem dimensions and the roles of
    :class:`numpy.ndarray` indices for convolutional representations
    as in :class:`.ConvBPDN` and related classes.
    """

    def __init__(self, dsz, S, dimK=None, dimN=2):
        """Initialise a ConvRepIndexing object representing dimensions of S
        (input signal), D (dictionary), and X (coefficient array) in a
        convolutional representation. These dimensions are inferred
        from the input `dsz` and `S` as well as from parameters `dimN`
        and `dimK`. Management and inferrence of these problem
        dimensions is not entirely straightforward because
        :class:`.ConvCnstrMOD` and related classes make use
        *internally* of S, D, and X arrays with a standard layout
        (described below), but *input* `S` and `dsz` are allowed to
        deviate from this layout for the convenience of the user. Note
        that S, D, and X refers to the names of signal, dictionary,
        and coefficient map arrays in :class:`.ConvBPDN`; the
        corresponding variable names in :class:`.ConvCnstrMOD` are S,
        X, and A.

        The most fundamental parameter is `dimN`, which specifies the
        dimensionality of the spatial/temporal samples being
        represented (e.g. `dimN` = 2 for representations of 2D
        images). This should be common to *input* `S` and `dsz`, and is also
        common to *internal* S, D, and X. The remaining dimensions of
        input `S` can correspond to multiple channels (e.g. for RGB
        images) and/or multiple signals (e.g. the array contains
        multiple independent images). If input `S` contains two
        additional dimensions (in addition to the `dimN` spatial
        dimensions), then those are considered to correspond, in
        order, to channel and signal indices. If there is only a
        single additional dimension, then determination whether it
        represents a channel or signal index is more complicated. The
        rule for making this determination is as follows:

        * if `dimK` is set to 0 or 1 instead of the default ``None``, then
          that value is taken as the number of signal indices in input `S`
          and any remaining indices are taken as channel indices (i.e. if
          `dimK` = 0 then dimC = 1 and if `dimK` = 1 then dimC = 0).
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

        * Nv = [N0, N1, ...] and N = N0 x N1 x ... are the vector of sizes
          of the spatial/temporal indices and the total number of
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
            self.K = S.shape[self.dimN+self.dimC]
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





class ConvCnstrMOD(admm.ADMMEqual):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ConvCnstrMOD
       :parts: 2

    |

    ADMM algorithm for Convolutional Constrained MOD problem
    :cite:`wohlberg-2016-efficient` :cite:`wohlberg-2016-convolutional`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_k \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \right\|_2^2 \quad \text{such that} \quad
       \mathbf{d}_m \in C

    via the ADMM problem

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_k \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \right\|_2^2 + \sum_m \iota_C(\mathbf{g}_m) \quad
       \text{such that} \quad \mathbf{d}_m = \mathbf{g}_m \;\;,

    where :math:`\iota_C(\cdot)` is the indicator function of feasible
    set :math:`C` consisting of filters with unit norm and constrained
    support. Multi-channel problems with input image channels
    :math:`\mathbf{s}_{c,k}` are also supported, either as

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_c \sum_k \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{c,k,m} -
       \mathbf{s}_{c,k} \right\|_2^2 \quad \text{such that} \quad
       \mathbf{d}_m \in C

    with single-channel dictionary filters :math:`\mathbf{d}_m` and
    multi-channel coefficient maps :math:`\mathbf{x}_{c,k,m}`, or

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_c \sum_k \left\| \sum_m \mathbf{d}_{c,m} * \mathbf{x}_{k,m} -
       \mathbf{s}_{c,k} \right\|_2^2 \quad \text{such that} \quad
       \mathbf{d}_{c,m} \in C

    with multi-channel dictionary filters :math:`\mathbf{d}_{c,m}` and
    single-channel coefficient maps :math:`\mathbf{x}_{k,m}`. In this
    latter case, normalisation of filters :math:`\mathbf{d}_{c,m}` is
    performed jointly over index :math:`c` for each filter :math:`m`.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``DFid`` : Value of data fidelity term :math:`(1/2) \sum_k \|
       \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} - \mathbf{s}_k \|_2^2`

       ``Cnstr`` : Constraint violation measure

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Relative residual of X step solver

       ``XSlvCGIt`` : CG iterations used in X step solver

       ``Time`` : Cumulative run time
    """



    class Options(admm.ADMMEqual.Options):
        r"""ConvCnstrMOD algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMMEqual.Options`, together with
        additional options:

          ``AuxVarObj`` : Flag indicating whether the objective
          function should be evaluated using variable X (``False``) or
          Y (``True``) as its argument. Setting this flag to ``True``
          often gives a better estimate of the objective function, but
          at additional computational cost.

          ``LinSolveCheck`` : If ``True``, compute relative residual
          of X step solver.

          ``ZeroMean`` : Flag indicating whether the solution
          dictionary :math:`\{\mathbf{d}_m\}` should have zero-mean
          components.

        ``LinSolve`` : Select linear solver for x step. Options are
        ``SM`` (Sherman-Morrison) or ``CG`` (Conjugate Gradient).

        ``CG`` : CG solver options

          ``MaxIter`` : Maximum iterations

          ``StopTol`` : Stopping tolerance
        """

        defaults = copy.deepcopy(admm.ADMMEqual.Options.defaults)
        defaults.update({'AuxVarObj' : False, 'ReturnX' : False,
                        'RelaxParam' : 1.8, 'ZeroMean' : False,
                        'LinSolve' : 'SM', 'LinSolveCheck' : False,
                        'CG' : {'MaxIter' : 1000, 'StopTol' : 1e-3}})
        defaults['AutoRho'].update({'Enabled' : True, 'Period' : 1,
                                    'AutoScaling' : True, 'Scaling' : 1000,
                                    'RsdlRatio' : 1.2})


        def __init__(self, opt=None):
            """Initialise ConvCnstrMOD algorithm options object."""

            if opt is None:
                opt = {}
            admm.ADMMEqual.Options.__init__(self, opt)

            if self['AutoRho', 'RsdlTarget'] is None:
                self['AutoRho', 'RsdlTarget'] = 1.0



        def __setitem__(self, key, value):
            """Set options 'fEvalX' and 'gEvalY' appropriately when option
            'AuxVarObj' is set.
            """

            admm.ADMMEqual.Options.__setitem__(self, key, value)

            if key == 'AuxVarObj':
                if value is True:
                    self['fEvalX'] = False
                    self['gEvalY'] = True
                else:
                    self['fEvalX'] = True
                    self['gEvalY'] = False



    itstat_fields_objfn = ('DFid', 'Cnstr')
    itstat_fields_extra = ('XSlvRelRes', 'CGIt')
    hdrtxt_objfn = ('DFid', 'Cnstr')
    hdrval_objfun = {'DFid' : 'DFid', 'Cnstr' : 'Cnstr'}



    def __init__(self, A, S, dsz, opt=None, dimK=1, dimN=2):
        """Initialise a ConvCnstrMOD object with problem parameters.

        This class supports an arbitrary number of spatial dimensions,
        `dimN`, with a default of 2. The input coefficient map array `A`
        (usually labelled X, but renamed here to avoid confusion with
        the X and Y variables in the ADMM base class) is expected to
        be in standard form as computed by the ConvBPDN class.

        The input signal set `S` is either `dimN` dimensional (no
        channels, only one signal), `dimN` +1 dimensional (either
        multiple channels or multiple signals), or `dimN` +2 dimensional
        (multiple channels and multiple signals). Parameter `dimK`, with
        a default value of 1, indicates the number of multiple-signal
        dimensions in `S`:

        ::

          Default dimK = 1, i.e. assume input S is of form
            S(N0,  N1,   C,   K)  or  S(N0,  N1,   K)
          If dimK = 0 then input S is of form
            S(N0,  N1,   C,   K)  or  S(N0,  N1,   C)

        The internal data layout for S, D (X here), and X (A here) is:
        ::

          dim<0> - dim<Nds-1> : Spatial dimensions, product of N0,N1,... is N
          dim<Nds>            : C number of channels in S and D
          dim<Nds+1>          : K number of signals in S
          dim<Nds+2>          : M number of filters in D

            sptl.      chn  sig  flt
          S(N0,  N1,   C,   K,   1)
          D(N0,  N1,   C,   1,   M)   (X here)
          X(N0,  N1,   1,   K,   M)   (A here)

        The `dsz` parameter indicates the desired filter supports in the
        output dictionary, since this cannot be inferred from the
        input variables. The format is the same as the `dsz` parameter
        of :func:`bcrop`.

        Parameters
        ----------
        A : array_like
          Coefficient map array
        S : array_like
          Signal array
        dsz : tuple
          Filter support size(s)
        opt : ccmod.Options object
          Algorithm options
        dimK : int, optional (default 1)
          Number of dimensions for multiple signals in input S
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvCnstrMOD.Options()

        # Infer problem dimensions and set relevant attributes of self
        self.cri = ConvRepIndexing(dsz, S, dimK=dimK, dimN=dimN)

        # Call parent class __init__
        super(ConvCnstrMOD, self).__init__(self.cri.shpD, S.dtype, opt)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=self.cri.K, dtype=self.dtype)

        # Reshape S to standard layout (A, i.e. X in cbpdn, is assumed
        # to be taken from cbpdn, and therefore already in standard
        # form). If the dictionary has a single channel but the input
        # (and therefore also the coefficient map array) has multiple
        # channels, the channel index and multiple image index have
        # the same behaviour in the dictionary update equation: the
        # simplest way to handle this is to just reshape so that the
        # channels also appear on the multiple image index.
        if self.cri.Cd == 1 and self.cri.C > 1:
            self.S = S.reshape(self.cri.Nv + (1,) +
                        (self.cri.C*self.cri.K,) + (1,))
        else:
            self.S = S.reshape(self.cri.shpS)
        self.S = np.asarray(self.S, dtype=self.dtype)

        # Compute signal S in DFT domain
        self.Sf = sl.rfftn(self.S, None, self.cri.axisN)

        # Create constraint set projection function
        self.Pcn = getPcn(opt['ZeroMean'], dsz, self.cri.Nv, self.cri.dimN)

        # Create byte aligned arrays for FFT calls
        self.YU = sl.pyfftw_empty_aligned(self.Y.shape, dtype=self.dtype)
        xfshp = list(self.Y.shape)
        xfshp[dimN-1] = xfshp[dimN-1]//2 + 1
        self.Xf = sl.pyfftw_empty_aligned(xfshp,
                            dtype=sl.complex_dtype(self.dtype))

        if A is not None:
            self.setcoef(A)



    def uinit(self, ushape):
        """Return initialiser for working variable U"""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            return self.Y



    def setcoef(self, A):
        """Set coefficient array."""

        # If the dictionary has a single channel but the input (and
        # therefore also the coefficient map array) has multiple
        # channels, the channel index and multiple image index have
        # the same behaviour in the dictionary update equation: the
        # simplest way to handle this is to just reshape so that the
        # channels also appear on the multiple image index.
        if self.cri.Cd == 1 and self.cri.C > 1:
            A = A.reshape(self.cri.Nv + (1,) + (self.cri.Cx*self.cri.K,) +
                          (self.cri.M,))
        self.A = np.asarray(A, dtype=self.dtype)

        self.Af = sl.rfftn(self.A, self.cri.Nv, self.cri.axisN)
        # Compute X^H S
        self.ASf = sl.inner(np.conj(self.Af), self.Sf, self.cri.axisK)



    def getdict(self):
        """Get final dictionary."""

        return bcrop(self.Y, self.cri.dsz)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{x}`."""

        self.cgit = None

        self.YU[:] = self.Y - self.U

        b = self.ASf + self.rho*sl.rfftn(self.YU, None, self.cri.axisN)
        if self.opt['LinSolve'] == 'SM':
            self.Xf[:] = sl.solvemdbi_ism(self.Af, self.rho, b, self.cri.axisM,
                                self.cri.axisK)
        else:
            self.Xf[:], cgit = sl.solvemdbi_cg(self.Af, self.rho, b,
                                self.cri.axisM, self.cri.axisK,
                                self.opt['CG', 'StopTol'],
                                self.opt['CG', 'MaxIter'], self.Xf)
            self.cgit = cgit

        self.X = sl.irfftn(self.Xf, self.cri.Nv, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            Aop = lambda x: np.sum(self.Af * x, axis=self.cri.axisM,
                                   keepdims=True)
            AHop = lambda x: np.sum(np.conj(self.Af) * x, axis=self.cri.axisK,
                                    keepdims=True)
            ax = AHop(Aop(self.Xf)) + self.rho*self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y = self.Pcn(self.AX + self.U)



    def obfn_fvarf(self):
        """Variable to be evaluated in computing data fidelity term,
        depending on 'fEvalX' option value.
        """

        return self.Xf if self.opt['fEvalX'] else \
            sl.rfftn(self.Y, None, self.cri.axisN)



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_dfd()
        cns = self.obfn_cns()
        return (dfd, cns)



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| \sum_m \mathbf{d}_m *
        \mathbf{x}_m - \mathbf{s} \|_2^2`.
        """

        Ef = sl.inner(self.Af, self.obfn_fvarf(), axis=self.cri.axisM) - self.Sf
        return sl.rfl2norm2(Ef, self.S.shape, axis=self.cri.axisN) / 2.0



    def obfn_cns(self):
        r"""Compute constraint violation measure :math:`\| P(\mathbf{y}) -
        \mathbf{y}\|_2`.
        """

        return linalg.norm((self.Pcn(self.obfn_gvar()) - self.obfn_gvar()))



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs, self.cgit)





def stdformD(D, Cd, M, dimN=2):
    """Reshape dictionary array (X here, D in cbpdn module) to internal
    standard form.

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



def getPcn0(zm, dsz, dimN=2, dimC=1):
    """Construct constraint set projection function without support
    projection. The `dsz` parameter specifies the support sizes of each
    filter using the same format as the `dsz` parameter of :func:`bcrop`.

    Parameters
    ----------
    zm : bool
      Flag indicating whether the projection function should include
      filter mean subtraction
    dsz : tuple
      Filter support size(s)
    dimN : int, optional (default 2)
      Number of problem spatial indices
    dimC : int, optional (default 1)
      Number of problem channel indices

    Returns
    -------
    fn : function
      Constraint set projection function
    """

    if zm:
        return lambda x: normalise(zeromean(bcrop(x, dsz), dsz, dimN),
                                   dimN+dimC)
    else:
        return lambda x: normalise(bcrop(x, dsz, dimN), dimN+dimC)



def getPcn(zm, dsz, Nv, dimN=2, dimC=1):
    """Construct the constraint set projection function utilised by
    ystep. The `dsz` parameter specifies the support sizes of each
    filter using the same format as the `dsz` parameter of :func:`bcrop`.

    Parameters
    ----------
    zm : bool
      Flag indicating whether the projection function should include
      filter mean subtraction
    dsz : tuple
      Filter support size(s)
    Nv : tuple
      Sizes of problem spatial indices
    dimN : int, optional (default 2)
      Number of problem spatial indices
    dimC : int, optional (default 1)
      Number of problem channel indices

    Returns
    -------
    fn : function
      Constraint set projection function
    """

    if zm:
        return lambda x: normalise(zeromean(zpad(bcrop(x, dsz, dimN), Nv),
                                            dsz), dimN+dimC)
    else:
        return lambda x: normalise(zpad(bcrop(x, dsz, dimN), Nv), dimN+dimC)



def zeromean(v, dsz, dimN=2):
    """Subtract mean value from each filter in the input array v. The `dsz`
    parameter specifies the support sizes of each filter using the
    same format as the `dsz` parameter of :func:`bcrop`. Support sizes
    must be taken into account to ensure that the mean values are
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
                c0 = 0  # Init. idx. of current channel-block of equi-sized flt.
                for cb in range(0, len(dsz[mb])):
                    c1 = c0 + dsz[mb][cb][-2]
                    # Construct slice corresponding to cropped part of
                    # current block of filters in output array and set from
                    # input array
                    cbslc = tuple([slice(0, x) for x in dsz[mb][cb][0:dimN]]) \
                            + (slice(c0, c1),) + (Ellipsis,) + (slice(m0, m1),)
                    vz[cbslc] -= np.mean(v[cbslc], axisN)
                    c0 = c1  # Update initial index for start of next block
            else:
                m1 = m0 + dsz[mb][-1]
                # Construct slice corresponding to cropped part of
                # current block of filters in output array and set from
                # input array
                mbslc = tuple([slice(0, x) for x in dsz[mb][0:-1]]) + \
                        (Ellipsis,) + (slice(m0, m1),)
                vz[mbslc] -= np.mean(v[mbslc], axisN)
            m0 = m1  # Update initial index for start of next block
    else:
        # Single scale dictionary specification
        axisN = tuple(range(0, dimN))
        axnslc = tuple([slice(0, x) for x in dsz[0:dimN]])
        vz[axnslc] -= np.mean(v[axnslc], axisN)

    return vz



def normalise(v, dimN=2):
    r"""Normalise vectors, corresponding to slices along specified number
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
    vn = np.sqrt(np.sum(v**2, axisN, keepdims=True))
    vn[vn == 0] = 1.0
    return np.asarray(v / vn, dtype=v.dtype)



def zpad(v, Nv):
    """Zero-pad initial axes of array to specified size. Padding is
    applied to the right, top, etc. of the array indices.

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
    """Crop specified number of initial spatial dimensions of dictionary
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
                    maxsz = np.maximum(maxsz, dsz[mb][cb][0:dimN])
            else:
                maxsz = np.maximum(maxsz, dsz[mb][0:dimN])
        # Init. cropped array
        vc = np.zeros(tuple(maxsz) + v.shape[dimN:], dtype=v.dtype)
        m0 = 0  # Initial index of current block of equi-sized filters
        # Iterate over distinct filter sizes
        for mb in range(0, len(dsz)):
            # Determine end index of current block of filters
            if isinstance(dsz[mb][0], tuple):
                m1 = m0 + dsz[mb][0][-1]
                c0 = 0  # Init. idx. of current channel-block of equi-sized flt.
                for cb in range(0, len(dsz[mb])):
                    c1 = c0 + dsz[mb][cb][-2]
                    # Construct slice corresponding to cropped part of
                    # current block of filters in output array and set from
                    # input array
                    cbslc = tuple([slice(0, x) for x in dsz[mb][cb][0:dimN]]) \
                            + (slice(c0, c1),) + (Ellipsis,) + (slice(m0, m1),)
                    vc[cbslc] = v[cbslc]
                    c0 = c1  # Update initial index for start of next block
            else:
                m1 = m0 + dsz[mb][-1]
                # Construct slice corresponding to cropped part of
                # current block of filters in output array and set from
                # input array
                mbslc = tuple([slice(0, x) for x in dsz[mb][0:-1]]) + \
                        (Ellipsis,) + (slice(m0, m1),)
                vc[mbslc] = v[mbslc]
            m0 = m1  # Update initial index for start of next block
        return vc
    else:
        # Single scale dictionary specification
        axnslc = tuple([slice(0, x) for x in dsz[0:dimN]])
        return v[axnslc]
