# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithm for the Convolutional BPDN problem"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from builtins import range

import copy
from types import MethodType
import pprint
import numpy as np
from scipy import linalg

from sporco.admm import admm
import sporco.linalg as sl
from sporco.util import u


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class ConvRepIndexing(object):
    """Manage the inference of problem dimensions and the roles of
    :class:`numpy.ndarray` indices for convolutional representations
    as in :class:`.ConvBPDN` and related classes.
    """

    def __init__(self, D, S, dimK=None, dimN=2):
        """Initialise a ConvRepIndexing object representing dimensions of S
        (input signal), D (dictionary), and X (coefficient array) in a
        convolutional representation. These dimensions are inferred
        from the input `D` and `S` as well as from parameters `dimN` and
        `dimK`. Management and inferrence of these problem dimensions
        is not entirely straightforward because :class:`.ConvBPDN` and
        related classes make use *internally* of S, D, and X arrays
        with a standard layout (described below), but *input* `S` and `D`
        are allowed to deviate from this layout for the convenience of
        the user.

        The most fundamental parameter is `dimN`, which specifies the
        dimensionality of the spatial/temporal samples being
        represented (e.g. `dimN` = 2 for representations of 2D
        images). This should be common to *input* S and D, and is also
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
        * if `dimK` is ``None`` then the number of channel dimensions is
          determined from the number of dimensions in the input dictionary `D`.
          Input `D` should have at least `dimN` + 1 dimensions, with the
          final dimension indexing dictionary filters. If it has exactly
          `dimN` + 1 dimensions then it is a single-channel dictionary,
          and input `S` is also assumed to be single-channel, with the
          additional index in `S` assigned as a signal index (i.e. dimK = 1).
          Conversely, if input `D` has `dimN` + 2 dimensions it is a
          multi-channel dictionary, and the additional index in `S` is
          assigned as a channel index (i.e. dimC = 1).

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
            self.K = S.shape[self.dimN+self.dimC]
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





class GenericConvBPDN(admm.ADMMEqual):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: GenericConvBPDN
       :parts: 2

    |

    Base class for ADMM algorithm for solving variants of the
    Convolutional BPDN (CBPDN) :cite:`wohlberg-2016-efficient` problem.

    The generic problem form is

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + \lambda f( \{ \mathbf{x}_m \} )

    for input image :math:`\mathbf{s}`, dictionary filters
    :math:`\mathbf{d}_m`, and coefficient maps :math:`\mathbf{x}_m`,
    and where :math:`f(\cdot)` is a penalty term or the indicator
    function of a constraint. It is solved via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + f ( \{ \mathbf{y}_m \} )
       \quad \text{such that} \quad \mathbf{x}_m = \mathbf{y}_m \;\;.

    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| \sum_m
       \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``Reg`` : Value of regularisation term

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Relative residual of X step solver

       ``Time`` : Cumulative run time
    """


    class Options(admm.ADMMEqual.Options):
        """ConvBPDN algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMMEqual.Options`, together with
        additional options:

          ``AuxVarObj`` : Flag indicating whether the objective
          function should be evaluated using variable X (``False``) or
          Y (``True``) as its argument. Setting this flag to ``True``
          often gives a better estimate of the objective function, but
          at additional computational cost.

          ``LinSolveCheck`` : Flag indicating whether to compute
          relative residual of X step solver.

          ``HighMemSolve`` : Flag indicating whether to use a slightly
          faster algorithm at the expense of higher memory usage.

          ``NonNegCoef`` : Flag indicating whether to force solution to
          be non-negative.

          ``NoBndryCross`` : Flag indicating whether all solution
          coefficients corresponding to filters crossing the image
          boundary should be forced to zero.
        """

        defaults = copy.deepcopy(admm.ADMMEqual.Options.defaults)
        defaults.update({'AuxVarObj' : False, 'ReturnX' : False,
                         'HighMemSolve' : False, 'LinSolveCheck' : False,
                         'RelaxParam' : 1.8, 'NonNegCoef' : False,
                         'NoBndryCross' : False})
        defaults['AutoRho'].update({'Enabled' : True, 'Period' : 1,
                                    'AutoScaling' : True, 'Scaling' : 1000.0,
                                    'RsdlRatio' : 1.2})


        def __init__(self, opt=None):
            """Initialise GenericConvBPDN algorithm options object."""

            if opt is None:
                opt = {}
            admm.ADMMEqual.Options.__init__(self, opt)



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



    itstat_fields_objfn = ('ObjFun', 'DFid', 'Reg')
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('Fnc', 'DFid', 'Reg')
    hdrval_objfun = {'Fnc' : 'ObjFun', 'DFid' : 'DFid', 'Reg' : 'Reg'}



    def __init__(self, D, S, opt=None, dimK=None, dimN=2):
        """
        Initialise a GenericConvBPDN object with problem parameters.

        This class supports an arbitrary number of spatial dimensions,
        `dimN`, with a default of 2. The input dictionary `D` is either
        `dimN` + 1 dimensional, in which case each spatial component
        (image in the default case) is assumed to consist of a single
        channel, or `dimN` + 2 dimensional, in which case the final
        dimension is assumed to contain the channels (e.g. colour
        channels in the case of images). The input signal set `S` is
        either `dimN` dimensional (no channels, only one signal), `dimN` + 1
        dimensional (either multiple channels or multiple signals), or
        `dimN` + 2 dimensional (multiple channels and multiple signals).
        Determination of problem dimensions is handled by
        :class:`ConvRepIndexing`.


        Parameters
        ----------
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        opt : :class:`GenericConvBPDN.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = GenericConvBPDN.Options()

        # Infer problem dimensions and set relevant attributes of self
        if not hasattr(self, 'cri'):
            self.cri = ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Call parent class __init__
        super(GenericConvBPDN, self).__init__(self.cri.shpX, S.dtype, opt)

        # Reshape D and S to standard layout
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # Compute signal in DFT domain
        self.Sf = sl.rfftn(self.S, None, self.cri.axisN)

        # Initialise byte-aligned arrays for pyfftw
        self.YU = sl.pyfftw_empty_aligned(self.Y.shape, dtype=self.dtype)
        xfshp = list(self.Y.shape)
        xfshp[dimN-1] = xfshp[dimN-1]//2 + 1
        self.Xf = sl.pyfftw_empty_aligned(xfshp,
                                          dtype=sl.complex_dtype(self.dtype))

        self.setdict()



    def dualinit(self):
        """Initial value for dual variable U if initial Y is provided.
        This base-class method is intended to be overridden, but an
        implementation is provided so that overriding it is not
        essential.
        """

        return np.zeros(self.cri.shpX, self.dtype)



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = sl.rfftn(self.D, self.cri.Nv, self.cri.axisN)
        # Compute D^H S
        self.DSf = np.conj(self.Df) * self.Sf
        if self.cri.Cd > 1:
            self.DSf = np.sum(self.DSf, axis=self.cri.axisC, keepdims=True)
        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbi_sm_c(self.Df, np.conj(self.Df), self.rho,
                                      self.cri.axisM)
        else:
            self.c = None



    def getcoef(self):
        """Get final coefficient array."""

        return self.Y



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`."""

        self.YU[:] = self.Y - self.U

        b = self.DSf + self.rho*sl.rfftn(self.YU, None, self.cri.axisN)
        if self.cri.Cd == 1:
            self.Xf[:] = sl.solvedbi_sm(self.Df, self.rho, b, self.c,
                                        self.cri.axisM)
        else:
            self.Xf[:] = sl.solvemdbi_ism(self.Df, self.rho, b, self.cri.axisM,
                                          self.cri.axisC)

        self.X = sl.irfftn(self.Xf, self.cri.Nv, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            Dop = lambda x: sl.inner(self.Df, x, axis=self.cri.axisM)
            if self.cri.Cd == 1:
                DHop = lambda x: np.conj(self.Df) * x
            else:
                DHop = lambda x: sl.inner(np.conj(self.Df), x,
                                          axis=self.cri.axisC)
            ax = DHop(Dop(self.Xf)) + self.rho*self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{y}`.
        If this method is not overridden, the problem is solved without
        any regularisation other than the option enforcement of
        non-negativity of the solution and filter boundary crossing
        supression. When it is overridden, it should be explicitly
        called at the end of the overriding method.
        """

        if self.opt['NonNegCoef']:
            self.Y[self.Y < 0.0] = 0.0
        if self.opt['NoBndryCross']:
            for n in range(0, self.cri.dimN):
                self.Y[(slice(None),)*n +
                       (slice(1-self.D.shape[n], None),)] = 0.0



    def obfn_fvarf(self):
        """Variable to be evaluated in computing data fidelity term,
        depending on ``fEvalX`` option value.
        """

        return self.Xf if self.opt['fEvalX'] else \
            sl.rfftn(self.Y, None, self.cri.axisN)



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_dfd()
        reg = self.obfn_reg()
        obj = dfd + reg[0]
        return (obj, dfd) + reg[1:]



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| \sum_m \mathbf{d}_m *
        \mathbf{x}_m - \mathbf{s} \|_2^2`.
        """

        Ef = sl.inner(self.Df, self.obfn_fvarf(), axis=self.cri.axisM) - \
          self.Sf
        return sl.rfl2norm2(Ef, self.S.shape, axis=self.cri.axisN)/2.0



    def obfn_reg(self):
        """Compute regularisation term(s) and contribution to objective
        function.
        """

        raise NotImplementedError()



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs,)



    def rhochange(self):
        """Updated cached c array when rho changes."""

        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbi_sm_c(self.Df, np.conj(self.Df), self.rho,
                                      self.cri.axisM)



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            X = self.Y
        Xf = sl.rfftn(X, None, self.cri.axisN)
        Sf = np.sum(self.Df * Xf, axis=self.cri.axisM)
        return sl.irfftn(Sf, self.cri.Nv, self.cri.axisN)





class ConvBPDN(GenericConvBPDN):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ConvBPDN
       :parts: 2

    |

    ADMM algorithm for the Convolutional BPDN (CBPDN)
    :cite:`wohlberg-2014-efficient` :cite:`wohlberg-2016-efficient`
    :cite:`wohlberg-2016-convolutional` problem.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1

    for input image :math:`\mathbf{s}`, dictionary filters
    :math:`\mathbf{d}_m`, and coefficient maps :math:`\mathbf{x}_m`,
    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + \lambda \sum_m \| \mathbf{y}_m \|_1
       \quad \text{such that} \quad \mathbf{x}_m = \mathbf{y}_m \;\;.

    Multi-image and multi-channel problems are also supported. The
    multi-image problem is

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_k \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \right\|_2^2 + \lambda \sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1

    with input images :math:`\mathbf{s}_k` and coefficient maps
    :math:`\mathbf{x}_{k,m}`, and the multi-channel problem with input
    image channels :math:`\mathbf{s}_c` is either

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_c \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{c,m} -
       \mathbf{s}_c \right\|_2^2 +
       \lambda \sum_c \sum_m \| \mathbf{x}_{c,m} \|_1

    with single-channel dictionary filters :math:`\mathbf{d}_m` and
    multi-channel coefficient maps :math:`\mathbf{x}_{c,m}`, or

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_c \left\| \sum_m \mathbf{d}_{c,m} * \mathbf{x}_m -
       \mathbf{s}_c \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1

    with multi-channel dictionary filters :math:`\mathbf{d}_{c,m}` and
    single-channel coefficient maps :math:`\mathbf{x}_m`.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| \sum_m
       \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_m \|
       \mathbf{x}_m \|_1`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Relative residual of X step solver

       ``Time`` : Cumulative run time
    """


    class Options(GenericConvBPDN.Options):
        r"""ConvBPDN algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMMEqual.Options`, together with
        additional options:

          ``L1Weight`` : An array of weights for the :math:`\ell_1`
          norm. The array shape must be such that the array is
          compatible for multiplication with the X/Y variables. If this
          option is defined, the regularization term is :math:`\lambda
          \sum_m \| \mathbf{w}_m \odot \mathbf{x}_m \|_1` where
          :math:`\mathbf{w}_m` denotes slices of the weighting array on
          the filter index axis.
        """

        defaults = copy.deepcopy(GenericConvBPDN.Options.defaults)
        defaults.update({'L1Weight' : 1.0})


        def __init__(self, opt=None):
            """Initialise ConvBPDN algorithm options object."""

            if opt is None:
                opt = {}
            GenericConvBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc' : 'ObjFun', 'DFid' : 'DFid', u('Regℓ1') : 'RegL1'}



    def __init__(self, D, S, lmbda=None, opt=None, dimK=None, dimN=2):
        """
        Initialise a ConvBPDN object with problem parameters.

        This class supports an arbitrary number of spatial dimensions,
        `dimN`, with a default of 2. The input dictionary `D` is either
        `dimN` + 1 dimensional, in which case each spatial component
        (image in the default case) is assumed to consist of a single
        channel, or `dimN` + 2 dimensional, in which case the final
        dimension is assumed to contain the channels (e.g. colour
        channels in the case of images). The input signal set `S` is
        either `dimN` dimensional (no channels, only one signal), `dimN` + 1
        dimensional (either multiple channels or multiple signals), or
        `dimN` + 2 dimensional (multiple channels and multiple signals).
        Determination of problem dimensions is handled by
        :class:`ConvRepIndexing`.


        Parameters
        ----------
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        lmbda : float
          Regularisation parameter
        opt : :class:`ConvBPDN.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvBPDN.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        # Set default lambda value if not specified
        if lmbda is None:
            cri = ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)
            Df = sl.rfftn(D.reshape(cri.shpD), cri.Nv, axes=cri.axisN)
            Sf = sl.rfftn(S.reshape(cri.shpS), axes=cri.axisN)
            b = np.conj(Df) * Sf
            lmbda = 0.1*abs(b).max()

        # Set l1 term scaling and weight array
        self.lmbda = self.dtype.type(lmbda)
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(50.0*self.lmbda + 1.0),
                      dtype=self.dtype)

        # Set rho_xi attribute (see Sec. VI.C of wohlberg-2015-adaptive)
        if self.lmbda != 0.0:
            rho_xi = (1.0 + (18.3)**(np.log10(self.lmbda) + 1.0))
        else:
            rho_xi = 1.0
        self.set_attr('rho_xi', opt['AutoRho', 'RsdlTarget'], dval=rho_xi,
                      dtype=self.dtype)

        # Call parent class __init__
        super(ConvBPDN, self).__init__(D, S, opt, dimK, dimN)



    def uinit(self, ushape):
        """Return initialiser for working variable U"""

        if  self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            return (self.lmbda/self.rho)*np.sign(self.Y)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`."""

        self.Y = sl.shrink1(self.AX + self.U,
                            (self.lmbda/self.rho) * self.wl1)
        super(ConvBPDN, self).ystep()



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        return (self.lmbda*rl1, rl1)





class ConvBPDNJoint(ConvBPDN):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ConvBPDNJoint
       :parts: 2

    |

    ADMM algorithm for Convolutional BPDN with joint sparsity via an
    :math:`\ell_{2,1}` norm term :cite:`wohlberg-2016-convolutional`
    (the :math:`\ell_2` norms are computed over the channel index).

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_c \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{c,m} -
       \mathbf{s}_c \right\|_2^2 + \lambda \sum_c \sum_m
       \| \mathbf{x}_{c,m} \|_1 + \mu \| \{ \mathbf{x}_{c,m} \} \|_{2,1}

    with input images :math:`\mathbf{s}_c`, dictionary filters
    :math:`\mathbf{d}_m`, and coefficient maps
    :math:`\mathbf{x}_{c,m}`, via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \sum_c \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{c,m} -
       \mathbf{s}_c \right\|_2^2 + \lambda \sum_c \sum_m
       \| \mathbf{y}_{c,m} \|_1 + \mu \| \{ \mathbf{y}_{c,m} \} \|_{2,1}
       \quad \text{such that} \quad \mathbf{x}_{c,m} = \mathbf{y}_{c,m} \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term :math:`(1/2) \sum_c
       \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} - \mathbf{s}_c
       \right\|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_c \sum_m
       \| \mathbf{x}_{c,m} \|_1`

       ``RegL21`` : Value of regularisation term :math:`\| \{
       \mathbf{x}_{c,m} \} \|_{2,1}`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual Residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegL21')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), u('Regℓ2,1'))
    hdrval_objfun = {'Fnc' : 'ObjFun', 'DFid' : 'DFid',
                     u('Regℓ1') : 'RegL1', u('Regℓ2,1') : 'RegL21'}



    def __init__(self, D, S, lmbda=None, mu=0.0, opt=None, dimK=None, dimN=2):
        """
        Initialise a ConvBPDNJoint object with problem parameters.

        Parameters
        ----------
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        lmbda : float
          Regularisation parameter (l1)
        mu : float
          Regularisation parameter (l2,1)
        opt : :class:`ConvBPDN.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        if opt is None:
            opt = ConvBPDN.Options()
        super(ConvBPDNJoint, self).__init__(D, S, lmbda, opt, dimK=dimK,
                                            dimN=dimN)
        self.mu = self.dtype.type(mu)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y = sl.shrink12(self.AX + self.U, (self.lmbda/self.rho)*self.wl1,
                             self.mu/self.rho, axis=self.cri.axisC)
        GenericConvBPDN.ystep(self)



    def obfn_reg(self):
        r"""Compute regularisation terms and contribution to objective
        function. Regularisation terms are :math:`\| Y \|_1` and
        :math:`\| Y \|_{2,1}`.
        """

        rl1 = linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rl21 = np.sum(np.sqrt(np.sum(self.obfn_gvar()**2, axis=self.cri.axisC)))
        return (self.lmbda*rl1 + self.mu*rl21, rl1, rl21)





class ConvElasticNet(ConvBPDN):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ConvElasticNet
       :parts: 2

    |

    ADMM algorithm for a convolutional form of the elastic net problem
    :cite:`zou-2005-regularization`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
       (\mu/2) \sum_m \| \mathbf{x}_m \|_2^2

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + \lambda \sum_m \| \mathbf{y}_m \|_1
       + (\mu/2) \sum_m \| \mathbf{x}_m \|_2^2
       \quad \text{such that} \quad \mathbf{x}_m = \mathbf{y}_m \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| \sum_m
       \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_m \|
       \mathbf{x}_m \|_1`

       ``RegL2`` : Value of regularisation term :math:`(1/2) \sum_m \|
       \mathbf{x}_m \|_2^2`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Relative residual of X step solver

       ``Time`` : Cumulative run time
    """



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegL2')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), u('Regℓ2'))
    hdrval_objfun = {'Fnc' : 'ObjFun', 'DFid' : 'DFid',
                     u('Regℓ1') : 'RegL1', u('Regℓ2') : 'RegL2'}



    def __init__(self, D, S, lmbda=None, mu=0.0, opt=None, dimK=None, dimN=2):
        """
        Initialise a ConvElasticNet object with problem parameters.

        Parameters
        ----------
        D : array_like
          Dictionary matrix
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter (l1)
        mu : float
          Regularisation parameter (l2)
        opt : :class:`ConvBPDN.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        if opt is None:
            opt = ConvBPDN.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        self.mu = self.dtype.type(mu)

        super(ConvElasticNet, self).__init__(D, S, lmbda, opt, dimK=dimK,
                                             dimN=dimN)



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = sl.rfftn(self.D, self.cri.Nv, self.cri.axisN)
        # Compute D^H S
        self.DSf = np.conj(self.Df) * self.Sf
        if self.cri.Cd > 1:
            self.DSf = np.sum(self.DSf, axis=self.cri.axisC, keepdims=True)
        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbi_sm_c(self.Df, np.conj(self.Df),
                                      self.mu + self.rho, self.cri.axisM)
        else:
            self.c = None



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.YU[:] = self.Y - self.U

        b = self.DSf + self.rho*sl.rfftn(self.YU, None, self.cri.axisN)
        if self.cri.Cd == 1:
            self.Xf[:] = sl.solvedbi_sm(self.Df, self.mu + self.rho,
                                        b, self.c, self.cri.axisM)
        else:
            self.Xf[:] = sl.solvemdbi_ism(self.Df, self.mu + self.rho, b,
                                          self.cri.axisM, self.cri.axisC)

        self.X = sl.irfftn(self.Xf, None, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            Dop = lambda x: sl.inner(self.Df, x, axis=self.cri.axisM)
            if self.cri.Cd == 1:
                DHop = lambda x: np.conj(self.Df) * x
            else:
                DHop = lambda x: sl.inner(np.conj(self.Df), x,
                                          axis=self.cri.axisC)
            ax = DHop(Dop(self.Xf)) + (self.mu + self.rho)*self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rl2 = 0.5*linalg.norm(self.obfn_gvar())**2
        return (self.lmbda*rl1 + self.mu*rl2, rl1, rl2)





class ConvBPDNGradReg(ConvBPDN):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ConvBPDNGradReg
       :parts: 2

    |

    ADMM algorithm for an extension of Convolutional BPDN including a
    term penalising the gradient of the coefficient maps
    :cite:`wohlberg-2016-convolutional2`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
       (\mu/2) \sum_i \sum_m \| G_i \mathbf{x}_m \|_2^2 \;\;,

    where :math:`G_i` is an operator computing the derivative along index
    :math:`i`, via the ADMM problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 + \lambda \sum_m \| \mathbf{y}_m \|_1 +
       (\mu/2) \sum_i \sum_m \| G_i \mathbf{x}_m \|_2^2
       \quad \text{such that} \quad \mathbf{x}_m = \mathbf{y}_m \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| \sum_m
       \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_m \|
       \mathbf{x}_m \|_1`

       ``RegGrad`` : Value of regularisation term :math:`(1/2) \sum_i
       \sum_m \| G_i \mathbf{x}_m \|_2^2`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Relative residual of X step solver

       ``Time`` : Cumulative run time
    """


    class Options(ConvBPDN.Options):
        r"""ConvBPDNGradReg algorithm options

        Options include all of those defined in
        :class:`sporco.admm.cbpdn.ConvBPDN.Options`, together with
        additional options:

          ``GradWeight`` : An array of weights :math:`w_m` for the term
          penalising the gradient of the coefficient maps. If this
          option is defined, the regularization term is :math:`\sum_i
          \sum_m w_m \| G_i \mathbf{x}_m \|_2^2` where :math:`w_m` is
          the weight for filter index :math:`m`. The array should be an
          :math:`M`-vector where :math:`M` is the number of filters in
          the dictionary.
        """

        defaults = copy.deepcopy(ConvBPDN.Options.defaults)
        defaults.update({'GradWeight' : 1.0})


        def __init__(self, opt=None):
            """Initialise ConvBPDNGradReg algorithm options object"""

            if opt is None:
                opt = {}
            ConvBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegGrad')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), u('Regℓ2∇'))
    hdrval_objfun = {'Fnc' : 'ObjFun', 'DFid' : 'DFid',
                     u('Regℓ1') : 'RegL1', u('Regℓ2∇') : 'RegGrad'}



    def __init__(self, D, S, lmbda=None, mu=0.0, opt=None, dimK=None, dimN=2):
        """
        Initialise a ConvBPDNGradReg object with problem parameters.

        Parameters
        ----------
        D : array_like
          Dictionary matrix
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter (l1)
        mu : float
          Regularisation parameter (gradient)
        opt : :class:`ConvBPDNGradReg.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        if opt is None:
            opt = ConvBPDNGradReg.Options()

        self.cri = ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        self.mu = self.dtype.type(mu)
        if hasattr(opt['GradWeight'], 'ndim'):
            self.Wgrd = np.asarray(opt['GradWeight'].reshape((1,)*(dimN+2) +
                                   opt['GradWeight'].shape), dtype=self.dtype)
        else:
            self.Wgrd = np.asarray(opt['GradWeight'], dtype=self.dtype)

        self.Gf, GHGf = sl.GradientFilters(self.cri.dimN+3, self.cri.axisN,
                                           self.cri.Nv, dtype=self.dtype)
        self.GHGf = self.Wgrd * GHGf

        super(ConvBPDNGradReg, self).__init__(D, S, lmbda, opt, dimK=dimK,
                                              dimN=dimN)



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = sl.rfftn(self.D, self.cri.Nv, self.cri.axisN)
        # Compute D^H S
        self.DSf = np.conj(self.Df) * self.Sf
        if self.cri.Cd > 1:
            self.DSf = np.sum(self.DSf, axis=self.cri.axisC, keepdims=True)
        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbd_sm_c(self.Df, np.conj(self.Df),
                        self.mu*self.GHGf + self.rho, self.cri.axisM)
        else:
            self.c = None



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.YU[:] = self.Y - self.U

        b = self.DSf + self.rho*sl.rfftn(self.YU, None, self.cri.axisN)
        if self.cri.Cd == 1:
            self.Xf[:] = sl.solvedbd_sm(self.Df, self.mu*self.GHGf + self.rho,
                                        b, self.c, self.cri.axisM)
        else:
            self.Xf[:] = sl.solvemdbi_ism(self.Df, self.mu*self.GHGf +
                                          self.rho, b, self.cri.axisM,
                                          self.cri.axisC)

        self.X = sl.irfftn(self.Xf, None, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            Dop = lambda x: sl.inner(self.Df, x, axis=self.cri.axisM)
            if self.cri.Cd == 1:
                DHop = lambda x: np.conj(self.Df) * x
            else:
                DHop = lambda x: sl.inner(np.conj(self.Df), x,
                                       axis=self.cri.axisC)
            ax = DHop(Dop(self.Xf)) + (self.mu*self.GHGf + self.rho)*self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        fvf = self.obfn_fvarf()
        rl1 = linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rgr = sl.rfl2norm2(np.sqrt(self.GHGf*np.conj(fvf)*fvf), self.cri.Nv,
                           self.cri.axisN)/2.0
        return (self.lmbda*rl1 + self.mu*rgr, rl1, rgr)





class ConvTwoBlockCnstrnt(admm.ADMMTwoBlockCnstrnt):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ConvTwoBlockCnstrnt
       :parts: 2

    |

    Base class for ADMM algorithms for problems of the form

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       g_0(D \mathbf{x} - \mathbf{s}) + g_1(\mathbf{x}) \;\;,

    where :math:`D \mathbf{x} = \sum_m \mathbf{d}_m * \mathbf{x}_m`,
    via an ADMM problem of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
       g_0(\mathbf{y}_0) + g_1(\mathbf{y}_1) \;\text{such that}\;
       \left( \begin{array}{c} D \\ I \end{array} \right) \mathbf{x}
       - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1 \end{array}
       \right) = \left( \begin{array}{c} \mathbf{s} \\
       \mathbf{0} \end{array} \right) \;\;.

    In this case the ADMM constraint is :math:`A\mathbf{x} + B\mathbf{y} =
    \mathbf{c}` where

    .. math::
       A = \left( \begin{array}{c} D \\ I \end{array} \right)
       \qquad B = -I \qquad \mathbf{y} = \left( \begin{array}{c}
       \mathbf{y}_0 \\ \mathbf{y}_1 \end{array} \right) \qquad
       \mathbf{c} = \left( \begin{array}{c} \mathbf{s} \\
       \mathbf{0} \end{array} \right) \;\;.

    This class specialises class :class:`.ADMMTwoBlockCnstrnt`, but remains
    a base class for other classes that specialise to specific optimisation
    problems.
    """

    class Options(admm.ADMMTwoBlockCnstrnt.Options):
        """ConvTwoBlockCnstrnt algorithm options

        Options include all of those defined in
        :class:`.ADMMTwoBlockCnstrnt.Options`, together with
        additional options:

          ``LinSolveCheck`` : Flag indicating whether to compute
          relative residual of X step solver.

          ``HighMemSolve`` : Flag indicating whether to use a slightly
          faster algorithm at the expense of higher memory usage.

          ``NonNegCoef`` : Flag indicating whether to force solution
          to be non-negative.

          ``NoBndryCross`` : Flag indicating whether all solution
          coefficients corresponding to filters crossing the image
          boundary should be forced to zero.
        """

        defaults = copy.deepcopy(admm.ADMMEqual.Options.defaults)
        defaults.update({'AuxVarObj' : False, 'HighMemSolve' : False,
                         'LinSolveCheck' : False, 'NonNegCoef' : False,
                         'NoBndryCross' : False, 'RelaxParam' : 1.8,
                         'rho' : 1.0, 'ReturnVar' : 'Y1'})


        def __init__(self, opt=None):
            """Initialise ConvTwoBlockCnstrnt algorithm options object."""

            if opt is None:
                opt = {}
            admm.ADMMTwoBlockCnstrnt.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'G0Val', 'G1Val')
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('Fnc', 'g0', 'g1')
    hdrval_objfun = {'Fnc' : 'ObjFun', 'g0' : 'G0Val', 'g1' : 'G1Val'}



    def __init__(self, D, S, opt=None, dimK=None, dimN=2):
        """
        Initialise a ConvTwoBlockCnstrnt object with problem size and options.

        Parameters
        ----------
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        opt : :class:`ConvTwoBlockCnstrnt.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        # Infer problem dimensions and set relevant attributes of self
        self.cri = ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Call parent class __init__
        Nx = self.cri.M * self.cri.N * self.cri.K
        shpY = list(self.cri.shpX)
        shpY[self.cri.axisM] += self.cri.C
        super(ConvTwoBlockCnstrnt, self).__init__(Nx, shpY, self.cri.axisM,
                                                  self.cri.C, S.dtype, opt)

        # Reshape D and S to standard layout
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # Initialise byte-aligned arrays for pyfftw
        self.YU = sl.pyfftw_empty_aligned(self.Y.shape, dtype=self.dtype)
        xfshp = list(self.cri.shpX)
        xfshp[dimN-1] = xfshp[dimN-1]//2 + 1
        self.Xf = sl.pyfftw_empty_aligned(xfshp,
                                          dtype=sl.complex_dtype(self.dtype))

        self.setdict()



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = sl.rfftn(self.D, self.cri.Nv, self.cri.axisN)
        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbi_sm_c(self.Df, np.conj(self.Df), 1.0,
                                      self.cri.axisM)
        else:
            self.c = None



    def getcoef(self):
        """Get final coefficient array."""

        return self.getmin()



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.YU[:] = self.Y - self.U
        self.block_sep0(self.YU)[:] += self.S
        YUf = sl.rfftn(self.YU, None, self.cri.axisN)
        if self.cri.Cd == 1:
            b = np.conj(self.Df) * self.block_sep0(YUf) + self.block_sep1(YUf)
        else:
            b = sl.inner(np.conj(self.Df), self.block_sep0(YUf),
                         axis=self.cri.axisC) + self.block_sep1(YUf)

        if self.cri.Cd == 1:
            self.Xf[:] = sl.solvedbi_sm(self.Df, 1.0, b, self.c,
                                        self.cri.axisM)
        else:
            self.Xf[:] = sl.solvemdbi_ism(self.Df, 1.0, b, self.cri.axisM,
                                          self.cri.axisC)

        self.X = sl.irfftn(self.Xf, self.cri.Nv, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            Dop = lambda x: sl.inner(self.Df, x, axis=self.cri.axisM)
            if self.cri.Cd == 1:
                DHop = lambda x: np.conj(self.Df) * x
            else:
                DHop = lambda x: sl.inner(np.conj(self.Df), x,
                                          axis=self.cri.axisC)
            ax = DHop(Dop(self.Xf)) + self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        if self.opt['NonNegCoef'] or self.opt['NoBndryCross']:
            Y1 = self.block_sep1(self.Y)
            if self.opt['NonNegCoef']:
                Y1[Y1 < 0.0] = 0.0
            if self.opt['NoBndryCross']:
                for n in range(0, self.cri.dimN):
                    Y1[(slice(None),)*n +
                       (slice(1-self.D.shape[n], None),)] = 0.0
            self.block_sep1(self.Y)[:] = Y1



    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        self.AXnr = self.cnst_A(self.X, self.Xf)
        if self.rlx == 1.0:
            self.AX = self.AXnr
        else:
            if not hasattr(self, 'c0'):
                self.c0 = self.cnst_c0()
            if not hasattr(self, 'c1'):
                self.c1 = self.cnst_c1()
            alpha = self.rlx
            self.AX = alpha*self.AXnr + \
                      (1-alpha)*self.block_cat(self.var_y0() + self.c0,
                                               self.var_y1() + self.c1)



    def block_sep0(self, Y):
        r"""Separate variable into component corresponding to
        :math:`\mathbf{y}_0` in :math:`\mathbf{y}\;\;`. The method
        from parent class :class:`.ADMMTwoBlockCnstrnt` is overridden
        here to allow swapping of C (channel) and M (filter) axes in
        block 0 so that it can be concatenated on axis M with block
        1. This is necessary because block 0 has the dimensions of S
        (N x C x K x 1) while block 1 has the dimensions of X (N x 1 x
        K x M).
        """

        return np.swapaxes(Y[(slice(None),)*self.blkaxis +
                    (slice(0, self.blkidx),)], self.cri.axisC, self.cri.axisM)



    def block_cat(self, Y0, Y1):
        r"""Concatenate components corresponding to :math:`\mathbf{y}_0` and
        :math:`\mathbf{y}_1` to form :math:`\mathbf{y}\;\;`.
        The method from parent class :class:`.ADMMTwoBlockCnstrnt` is
        overridden here to allow swapping of C (channel) and M
        (filter) axes in block 0 so that it can be concatenated on
        axis M with block 1. This is necessary because block 0 has the
        dimensions of S (N x C x K x 1) while block 1 has the
        dimensions of X (N x 1 x K x M).
        """

        return np.concatenate((np.swapaxes(Y0, self.cri.axisC,
                              self.cri.axisM), Y1), axis=self.blkaxis)



    def cnst_A(self, X, Xf=None):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint.
        """

        return self.block_cat(self.cnst_A0(X, Xf), self.cnst_A1(X))



    def obfn_g0var(self):
        """Variable to be evaluated in computing
        :meth:`.ADMMTwoBlockCnstrnt.obfn_g0`, depending on the ``AuxVarObj``
        option value.
        """

        return self.var_y0() if self.opt['AuxVarObj'] else \
            self.cnst_A0(None, self.Xf) - self.cnst_c0()



    def cnst_A0(self, X, Xf=None):
        r"""Compute :math:`A_0 \mathbf{x}` component of ADMM problem
        constraint.
        """

        # This calculation involves non-negligible computational cost
        # when Xf is None (i.e. the function is not being applied to
        # self.X).
        if Xf is None:
            Xf = sl.rfftn(X, None, self.cri.axisN)
        return sl.irfftn(sl.inner(self.Df, Xf, axis=self.cri.axisM),
                                  self.cri.Nv, self.cri.axisN)



    def cnst_A0T(self, Y0):
        r"""Compute :math:`A_0^T \mathbf{y}_0` component of
        :math:`A^T \mathbf{y}` (see :meth:`.ADMMTwoBlockCnstrnt.cnst_AT`).
        """

        # This calculation involves non-negligible computational cost. It
        # should be possible to disable relevant diagnostic information
        # (dual residual) to avoid this cost.
        Y0f = sl.rfftn(Y0, None, self.cri.axisN)
        return sl.irfftn(sl.inner(np.conj(self.Df), Y0f, axis=self.cri.axisC),
                                  self.cri.Nv, self.cri.axisN)



    def cnst_c0(self):
        r"""Compute constant component :math:`\mathbf{c}_0` of
        :math:`\mathbf{c}` in the ADMM problem constraint.
        """

        return self.S



    def eval_objfn(self):
        """Compute components of regularisation function as well as total
        contribution to objective function.
        """

        g0v = self.obfn_g0(self.obfn_g0var())
        g1v = self.obfn_g1(self.obfn_g1var())
        obj = g0v + g1v
        return (obj, g0v, g1v)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs,)



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            Xf = self.Xf
        else:
            Xf = sl.rfftn(X, None, self.cri.axisN)

        Sf = np.sum(self.Df * Xf, axis=self.cri.axisM)
        return sl.irfftn(Sf, self.cri.Nv, self.cri.axisN)



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        return self.rho*linalg.norm(self.cnst_AT(self.U))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho*linalg.norm(U)





class ConvBPDNMaskDcpl(ConvTwoBlockCnstrnt):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ConvBPDNMaskDcpl
       :parts: 2

    |

    ADMM algorithm for Convolutional BPDN with Mask Decoupling
    :cite:`heide-2015-fast` :cite:`wohlberg-2016-boundary`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s}\right) \right\|_2^2 + \lambda \sum_m
       \| \mathbf{x}_m \|_1 \;\;,

    where :math:`W` is a mask array, via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
       (1/2) \| W \mathbf{y}_0 \|_2^2 + \lambda \| \mathbf{y}_1 \|_1
       \;\text{such that}\;
       \left( \begin{array}{c} D \\ I \end{array} \right) \mathbf{x}
       - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1 \end{array}
       \right) = \left( \begin{array}{c} \mathbf{s} \\
       \mathbf{0} \end{array} \right) \;\;,

    where :math:`D \mathbf{x} = \sum_m \mathbf{d}_m * \mathbf{x}_m`.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| W
       (\sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}) \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_m \|
       \mathbf{x}_m \|_1`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Relative residual of X step solver

       ``Time`` : Cumulative run time
    """

    class Options(ConvTwoBlockCnstrnt.Options):
        r"""ConvBPDNMaskDcpl algorithm options

        Options include all of those defined in
        :class:`sporco.admm.cbpdn.ConvTwoBlockCnstrnt.Options`, together with
        additional options:

          ``L1Weight`` : An array of weights for the :math:`\ell_1`
          norm. The array shape must be such that the array is
          compatible for multiplication with the X/Y variables. If this
          option is defined, the regularization term is :math:`\lambda
          \sum_m \| \mathbf{w}_m \odot \mathbf{x}_m \|_1` where
          :math:`\mathbf{w}_m` denotes slices of the weighting array on
          the filter index axis.
        """

        defaults = copy.deepcopy(ConvTwoBlockCnstrnt.Options.defaults)
        defaults.update({'L1Weight' : 1.0})


        def __init__(self, opt=None):
            """Initialise ConvTwoBlockCnstrnt algorithm options object"""

            if opt is None:
                opt = {}
            ConvTwoBlockCnstrnt.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc' : 'ObjFun', 'DFid' : 'DFid', u('Regℓ1') : 'RegL1'}



    def __init__(self, D, S, lmbda, W=None, opt=None, dimK=None, dimN=2):
        """Initialise a ConvBPDNMaskDcpl object with problem parameters.

        Parameters
        ----------
        D : array_like
          Dictionary matrix
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        W : array_like
          Mask array. The array shape must be such that the array is
          compatible for multiplication with the *internal* shape of
          input array S (see :class:`.ConvRepIndexing` for a discussion
          of the distinction between *external* and *internal* data
          layouts).
        opt : :class:`ConvBPDNMaskDcpl.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        if opt is None:
            opt = ConvBPDNMaskDcpl.Options()

        super(ConvBPDNMaskDcpl, self).__init__(D, S, opt, dimK=dimK, dimN=dimN)

        self.lmbda = self.dtype.type(lmbda)
        if W is None:
            W = np.array([1.0], dtype=self.dtype)
        self.W = np.asarray(sl.atleast_nd(self.S.ndim, W), dtype=self.dtype)
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)



    def uinit(self, ushape):
        """Return initialiser for working variable U"""

        if  self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            Ub0 = (self.W**2) * self.block_sep0(self.Y) / self.rho
            Ub1 = (self.lmbda/self.rho) * np.sign(self.block_sep1(self.Y))
            return self.block_cat(Ub0, Ub1)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        AXU = self.AX + self.U
        Y0 = (self.rho*(self.block_sep0(AXU) - self.S)) / (self.W**2 + self.rho)
        Y1 = sl.shrink1(self.block_sep1(AXU), (self.lmbda/self.rho)*self.wl1)
        self.Y = self.block_cat(Y0, Y1)

        super(ConvBPDNMaskDcpl, self).ystep()



    def obfn_g0(self, Y0):
        r"""Compute :math:`g_0(\mathbf{y}_0)` component of ADMM objective
        function.
        """

        return (linalg.norm(self.W * self.obfn_g0var())**2) / 2.0



    def obfn_g1(self, Y1):
        r"""Compute :math:`g_1(\mathbf{y_1})` component of ADMM objective
        function.
        """

        return linalg.norm((self.wl1*self.obfn_g1var()).ravel(), 1)





class AddMaskSim(object):
    """Boundary masking for convolutional representations using the
    Additive Mask Simulation (AMS) technique described in
    :cite:`wohlberg-2016-boundary`. Implemented as a wrapper about a
    cbpdn.ConvBPDN or derived object (or any other object with sufficiently
    similar interface and internals).
    """

    def __init__(self, cbpdnclass, D, S, W, *args, **kwargs):
        """Initialise an AddMaskSim object with problem parameters.

        Parameters
        ----------
        cbpdnclass : class name
          Type of internal cbpdn object (e.g. cbpdn.ConvBPDN) to be
          constructed
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        W : array_like
          Mask array. The array shape must be such that the array is
          compatible for multiplication with the *internal* shape of
          input array S (see :class:`.ConvRepIndexing` for a discussion
          of the distinction between *external* and *internal* data
          layouts).
        *args
          Variable length list of arguments for constructor of internal
          cbpdn object
        **kwargs
          Keyword arguments for constructor of internal cbpdn object
        """

        # Number of channel dimensions
        if 'dimK' in kwargs:
            dimK = kwargs['dimK']
        else:
            dimK = None

        # Number of spatial dimensions
        if 'dimN' in kwargs:
            dimN = kwargs['dimN']
        else:
            dimN = 2

        # Infer problem dimensions
        self.cri = ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Construct impulse filter (or filters for the multi-channel
        # case) and append to dictionary
        if self.cri.Cd == 1:
            self.imp = np.zeros(D.shape[0:dimN] + (1,))
            self.imp[(0,)*dimN] = 1.0
        else:
            self.imp = np.zeros(D.shape[0:dimN] + (self.cri.Cd,)*2)
            for c in range(0, self.cri.Cd):
                self.imp[(0,)*dimN+(c, c,)] = 1.0
        Di = np.concatenate((D, self.imp), axis=D.ndim-1)

        # Construct inner cbpdn object
        self.cbpdn = cbpdnclass(Di, S, *args, **kwargs)

        # Mask matrix
        self.W = np.asarray(sl.atleast_nd(self.cri.dimN+3, W),
                            dtype=self.cbpdn.dtype)
        # If Cd > 1 (i.e. a multi-channel dictionary) and mask has a
        # non-singleton channel dimension, swap that axis onto the
        # dictionary filter index dimension (where the
        # multiple-channel impulse filters are located)
        if self.cri.Cd > 1 and self.W.shape[self.cri.dimN] > 1:
            self.W = np.swapaxes(self.W, self.cri.axisC, self.cri.axisM)

        # Record ystep method of inner cbpdn object
        self.inner_ystep = self.cbpdn.ystep
        # Replace ystep method of inner cbpdn object with outer ystep
        self.cbpdn.ystep = MethodType(AddMaskSim.ystep, self)

        # Record obfn_gvar method of inner cbpdn object
        self.inner_obfn_gvar = self.cbpdn.obfn_gvar
        # Replace obfn_gvar method of inner cbpdn object with outer obfn_gvar
        self.cbpdn.obfn_gvar = MethodType(AddMaskSim.obfn_gvar, self)



    def ystep(self):
        """This method is inserted into the inner cbpdn object,
        replacing its own ystep method, thereby providing a hook for
        applying the additional steps necessary for the AMS method.
        """

        # Extract AMS part of ystep argument so that it is not
        # affected by the main part of the ystep
        amidx = self.index_addmsk()
        Yi = self.cbpdn.AX[amidx] + self.cbpdn.U[amidx]
        # Perform main part of ystep from inner cbpdn object
        self.inner_ystep()
        # Apply mask to AMS component and insert into Y from inner
        # cbpdn object
        Yi[np.where(self.W.astype(np.bool))] = 0.0
        self.cbpdn.Y[amidx] = Yi



    def obfn_gvar(self):
        """This method is inserted into the inner cbpdn object,
        replacing its own obfn_gvar method, thereby providing a hook for
        applying the additional steps necessary for the AMS method.
        """

        # Get inner cbpdn object gvar
        gv = self.inner_obfn_gvar().copy()
        # Set slice corresponding to the coefficient map of the final
        # filter (the impulse inserted for the AMS method) to zero so
        # that it does not affect the results (e.g. l1 norm) computed
        # from this variable by the inner cbpdn object
        gv[..., -1] = 0

        return gv



    def solve(self):
        """Call the solve method if the inner cbpdn object and strip
        the AMS component from the returned result.
        """

        # Call solve method of inner cbpdn object
        Xi = self.cbpdn.solve()
        # Copy attributes from inner cbpdn object
        self.timer = self.cbpdn.timer
        self.itstat = self.cbpdn.itstat
        # Return result of inner cbpdn object with AMS component removed
        return Xi[self.index_primary()]



    def setdict(self, D=None):
        """Set dictionary array."""

        Di = np.concatenate((D, sl.atleast_nd(D.ndim, self.imp)),
                            axis=D.ndim-1)
        self.cbpdn.setdict(Di)



    def getcoef(self):
        """Get result of inner cbpdn object with AMS component removed."""

        return self.cbpdn.getcoef()[self.index_primary()]



    def index_primary(self):
        """Return an index expression appropriate for extracting the primary
        (inner) component of the main variables X, Y, etc.
        """

        return np.s_[..., 0:-self.cri.Cd]



    def index_addmsk(self):
        """Return an index expression appropriate for extracting the
        additive mask (outer) component of the main variables X, Y, etc."""

        return np.s_[..., -self.cri.Cd:]



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        # If coefficient array not specified, use non-AMS part of Y from inner
        # cbpdn object
        if X is None:
            X = self.cbpdn.Y[self.index_primary()]
        # FFT of coefficient array
        Xf = sl.rfftn(X, None, self.cri.axisN)
        # Multiply in frequency domain with non-impulse component of
        # dictionary
        Sf = np.sum(self.cbpdn.Df[..., 0:-self.cri.Cd] * Xf,
                    axis=self.cri.axisM)
        # Transform to spatial domain and return result
        return sl.irfftn(Sf, self.cri.Nv, self.cri.axisN)



    def getitstat(self):
        """Get iteration stats from inner cbpdn object."""

        return self.cbpdn.getitstat()
