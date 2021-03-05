# -*- coding: utf-8 -*-
# Copyright (C) 2015-2021 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithm for the Convolutional BPDN problem"""

from __future__ import division, absolute_import, print_function
from builtins import range

import copy
from types import MethodType
import numpy as np

from sporco.admm import admm
import sporco.cnvrep as cr
import sporco.linalg as sl
import sporco.prox as sp
from sporco.util import u
from sporco.fft import (empty_aligned, real_dtype, empty_aligned_func,
                        fftn_func, ifftn_func, fl2norm2_func)
from sporco.signal import gradient_filters


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class GenericConvBPDN(admm.ADMMEqual):
    r"""
    Base class for ADMM algorithm for solving variants of the
    Convolutional BPDN (CBPDN) :cite:`wohlberg-2014-efficient`
    :cite:`wohlberg-2016-efficient` :cite:`wohlberg-2016-convolutional`
    problem.

    |

    .. inheritance-diagram:: GenericConvBPDN
       :parts: 2

    |

    The generic problem form is

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + g( \{ \mathbf{x}_m \} )

    for input image :math:`\mathbf{s}`, dictionary filters
    :math:`\mathbf{d}_m`, and coefficient maps :math:`\mathbf{x}_m`,
    and where :math:`g(\cdot)` is a penalty term or the indicator
    function of a constraint. It is solved via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + g( \{ \mathbf{y}_m \} )
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
        """GenericConvBPDN algorithm options

        Options include all of those defined in
        :class:`.admm.ADMMEqual.Options`, together with additional options:

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
        # Warning: although __setitem__ below takes care of setting
        # 'fEvalX' and 'gEvalY' from the value of 'AuxVarObj', this
        # cannot be relied upon for initialisation since the order of
        # initialisation of the dictionary keys is not deterministic;
        # if 'AuxVarObj' is initialised first, the other two keys are
        # correctly set, but this setting is overwritten when 'fEvalX'
        # and 'gEvalY' are themselves initialised
        defaults.update({'AuxVarObj': False, 'fEvalX': True,
                         'gEvalY': False, 'ReturnX': False,
                         'HighMemSolve': False, 'LinSolveCheck': False,
                         'RelaxParam': 1.8, 'NonNegCoef': False,
                         'NoBndryCross': False})
        defaults['AutoRho'].update({'Enabled': True, 'Period': 1,
                                    'AutoScaling': True, 'Scaling': 1000.0,
                                    'RsdlRatio': 1.2})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              GenericConvBPDN algorithm options
            """

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
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', 'Reg': 'Reg'}



    def __init__(self, D, S, opt=None, dimK=None, dimN=2):
        """
        This class supports an arbitrary number of spatial dimensions,
        `dimN`, with a default of 2. The input dictionary `D` is either
        `dimN` + 1 dimensional, in which case each spatial component
        (image in the default case) is assumed to consist of a single
        channel, or `dimN` + 2 dimensional, in which case the final
        dimension is assumed to contain the channels (e.g. colour
        channels in the case of images). The input signal set `S` is
        either `dimN` dimensional (no channels, only one signal),
        `dimN` + 1 dimensional (either multiple channels or multiple
        signals), or `dimN` + 2 dimensional (multiple channels and
        multiple signals). Determination of problem dimensions is
        handled by :class:`.cnvrep.CSC_ConvRepIndexing`.


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

        # Set flag indicating whether problem involves real or complex
        # values, and get appropriate versions of functions from fft
        # module
        self.real_dtype = np.isrealobj(D) and np.isrealobj(S)
        self.empty_aligned = empty_aligned_func(self.real_dtype)
        self.fftn = fftn_func(self.real_dtype)
        self.ifftn = ifftn_func(self.real_dtype)
        self.fl2norm2 = fl2norm2_func(self.real_dtype)

        # Infer problem dimensions and set relevant attributes of self
        if not hasattr(self, 'cri'):
            self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Call parent class __init__
        super(GenericConvBPDN, self).__init__(self.cri.shpX, S.dtype, opt)

        # Reshape D and S to standard layout
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # Compute signal in DFT domain
        self.Sf = self.fftn(self.S, None, self.cri.axisN)

        # Initialise byte-aligned arrays for pyfftw
        self.YU = empty_aligned(self.Y.shape, dtype=self.dtype)
        self.Xf = self.empty_aligned(self.Y.shape, self.cri.axisN,
                                     self.dtype)

        self.setdict()



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = self.fftn(self.D, self.cri.Nv, self.cri.axisN)
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

        return self.getmin()



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`."""

        self.YU[:] = self.Y - self.U

        b = self.DSf + self.rho * self.fftn(self.YU, None, self.cri.axisN)
        if self.cri.Cd == 1:
            self.Xf[:] = sl.solvedbi_sm(self.Df, self.rho, b, self.c,
                                        self.cri.axisM)
        else:
            self.Xf[:] = sl.solvemdbi_ism(self.Df, self.rho, b, self.cri.axisM,
                                          self.cri.axisC)

        self.X = self.ifftn(self.Xf, self.cri.Nv, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            Dop = lambda x: sl.inner(self.Df, x, axis=self.cri.axisM)
            if self.cri.Cd == 1:
                DHop = lambda x: np.conj(self.Df) * x
            else:
                DHop = lambda x: sl.inner(np.conj(self.Df), x,
                                          axis=self.cri.axisC)
            ax = DHop(Dop(self.Xf)) + self.rho * self.Xf
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
                self.Y[(slice(None),) * n +
                       (slice(1 - self.D.shape[n], None),)] = 0.0



    def obfn_fvarf(self):
        """Variable to be evaluated in computing data fidelity term,
        depending on ``fEvalX`` option value.
        """

        return self.Xf if self.opt['fEvalX'] else \
            self.fftn(self.Y, None, self.cri.axisN)



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
        return self.fl2norm2(Ef, self.S.shape, axis=self.cri.axisN) / 2.0



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
        Xf = self.fftn(X, None, self.cri.axisN)
        Sf = np.sum(self.Df * Xf, axis=self.cri.axisM)
        return self.ifftn(Sf, self.cri.Nv, self.cri.axisN)





class ConvBPDN(GenericConvBPDN):
    r"""
    ADMM algorithm for the Convolutional BPDN (CBPDN)
    :cite:`wohlberg-2014-efficient` :cite:`wohlberg-2016-efficient`
    :cite:`wohlberg-2016-convolutional` problem.

    |

    .. inheritance-diagram:: ConvBPDN
       :parts: 2

    |

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

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
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
        :class:`.admm.ADMMEqual.Options`, together with additional options:

          ``L1Weight`` : An array of weights for the :math:`\ell_1`
          norm. The array shape must be such that the array is
          compatible for multiplication with the `X`/`Y` variables (see
          :func:`.cnvrep.l1Wshape` for more details). If this
          option is defined, the regularization term is :math:`\lambda
          \sum_m \| \mathbf{w}_m \odot \mathbf{x}_m \|_1` where
          :math:`\mathbf{w}_m` denotes slices of the weighting array on
          the filter index axis.
        """

        defaults = copy.deepcopy(GenericConvBPDN.Options.defaults)
        defaults.update({'L1Weight': 1.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDN algorithm options
            """

            if opt is None:
                opt = {}
            GenericConvBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1'}



    def __init__(self, D, S, lmbda=None, opt=None, dimK=None, dimN=2):
        """
        This class supports an arbitrary number of spatial dimensions,
        `dimN`, with a default of 2. The input dictionary `D` is either
        `dimN` + 1 dimensional, in which case each spatial component
        (image in the default case) is assumed to consist of a single
        channel, or `dimN` + 2 dimensional, in which case the final
        dimension is assumed to contain the channels (e.g. colour
        channels in the case of images). The input signal set `S` is
        either `dimN` dimensional (no channels, only one signal), `dimN`
        + 1 dimensional (either multiple channels or multiple signals),
        or `dimN` + 2 dimensional (multiple channels and multiple
        signals). Determination of problem dimensions is handled by
        :class:`.cnvrep.CSC_ConvRepIndexing`.


        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdn_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdn_init.svg

        |


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

        # Call parent class __init__
        super(ConvBPDN, self).__init__(D, S, opt, dimK, dimN)

        # Set default lambda value if not specified
        if lmbda is None:
            cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)
            Df = self.fftn(D.reshape(cri.shpD), cri.Nv, axes=cri.axisN)
            Sf = self.fftn(S.reshape(cri.shpS), axes=cri.axisN)
            b = np.conj(Df) * Sf
            lmbda = 0.1 * abs(b).max()

        # Set l1 term scaling
        self.lmbda = real_dtype(self.dtype).type(lmbda)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(50.0 * self.lmbda + 1.0),
                      dtype=real_dtype(self.dtype), reset=True)

        # Set rho_xi attribute (see Sec. VI.C of wohlberg-2015-adaptive)
        if self.lmbda != 0.0:
            rho_xi = float((1.0 + (18.3)**(np.log10(self.lmbda) + 1.0)))
        else:
            rho_xi = 1.0
        self.set_attr('rho_xi', opt['AutoRho', 'RsdlTarget'], dval=rho_xi,
                      dtype=real_dtype(self.dtype), reset=True)

        # Set l1 term weight array
        self.wl1 = np.asarray(opt['L1Weight'], dtype=real_dtype(self.dtype))
        self.wl1 = self.wl1.reshape(cr.l1Wshape(self.wl1, self.cri))



    def uinit(self, ushape):
        """Return initialiser for working variable U"""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            return (self.lmbda/self.rho)*np.sign(self.Y)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`."""

        self.Y = sp.prox_l1(self.AX + self.U,
                            (self.lmbda / self.rho) * self.wl1)
        super(ConvBPDN, self).ystep()



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = np.linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        return (self.lmbda*rl1, rl1)





class ConvBPDNJoint(ConvBPDN):
    r"""
    ADMM algorithm for Convolutional BPDN with joint sparsity via an
    :math:`\ell_{2,1}` norm term :cite:`wohlberg-2016-convolutional`
    (the :math:`\ell_2` norms are computed over the channel index).

    |

    .. inheritance-diagram:: ConvBPDNJoint
       :parts: 2

    |

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

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
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


    class Options(ConvBPDN.Options):
        r"""ConvBPDNJoint algorithm options

        Options include all of those defined in :class:`ConvBPDN.Options`,
        together with additional options:

          ``L21Weight`` : An array of weights for the :math:`\ell_{2,1}`
          norm. The array shape must be such that the array is
          compatible for multiplication with the X/Y variables *after*
          the sum over ``axisC`` performed during the computation of the
          :math:`\ell_{2,1}` norm. If this option is defined, the
          regularization term is :math:`\mu \sum_i w_i \sqrt{ \sum_c
          \mathbf{x}_{i,c}^2 }` where :math:`w_i` are the elements of the
          weight array, subscript :math:`c` indexes the channel axis and
          subscript :math:`i` indexes all other axes.
        """

        defaults = copy.deepcopy(ConvBPDN.Options.defaults)
        defaults.update({'L21Weight': 1.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDNJoint algorithm options
            """

            if opt is None:
                opt = {}
            ConvBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegL21')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), u('Regℓ2,1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid',
                     u('Regℓ1'): 'RegL1', u('Regℓ2,1'): 'RegL21'}



    def __init__(self, D, S, lmbda=None, mu=0.0, opt=None, dimK=None, dimN=2):
        """
        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdnjnt_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdnjnt_init.svg

        |


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
        opt : :class:`ConvBPDNJoint.Options` object
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
        self.wl21 = np.asarray(opt['L21Weight'], dtype=self.dtype)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y = sp.prox_sl1l2(self.AX + self.U,
                               (self.lmbda / self.rho) * self.wl1,
                               (self.mu / self.rho) * self.wl21,
                               axis=self.cri.axisC)
        GenericConvBPDN.ystep(self)



    def obfn_reg(self):
        r"""Compute regularisation terms and contribution to objective
        function. Regularisation terms are :math:`\| Y \|_1` and
        :math:`\| Y \|_{2,1}`.
        """

        rl1 = np.linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rl21 = np.sum(self.wl21 * np.sqrt(np.sum(self.obfn_gvar()**2,
                                                 axis=self.cri.axisC)))
        return (self.lmbda*rl1 + self.mu*rl21, rl1, rl21)





class ConvElasticNet(ConvBPDN):
    r"""
    ADMM algorithm for a convolutional form of the elastic net problem
    :cite:`zou-2005-regularization`.

    |

    .. inheritance-diagram:: ConvElasticNet
       :parts: 2

    |

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

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
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
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid',
                     u('Regℓ1'): 'RegL1', u('Regℓ2'): 'RegL2'}



    def __init__(self, D, S, lmbda=None, mu=0.0, opt=None, dimK=None, dimN=2):
        """
        |

        **Call graph**

        .. image:: ../_static/jonga/celnet_init.svg
           :width: 20%
           :target: ../_static/jonga/celnet_init.svg

        |


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
        self.Df = self.fftn(self.D, self.cri.Nv, self.cri.axisN)
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

        b = self.DSf + self.rho*self.fftn(self.YU, None, self.cri.axisN)
        if self.cri.Cd == 1:
            self.Xf[:] = sl.solvedbi_sm(self.Df, self.mu + self.rho,
                                        b, self.c, self.cri.axisM)
        else:
            self.Xf[:] = sl.solvemdbi_ism(self.Df, self.mu + self.rho, b,
                                          self.cri.axisM, self.cri.axisC)

        self.X = self.ifftn(self.Xf, self.cri.Nv, self.cri.axisN)

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

        rl1 = np.linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rl2 = 0.5*np.linalg.norm(self.obfn_gvar())**2
        return (self.lmbda*rl1 + self.mu*rl2, rl1, rl2)





class ConvBPDNGradReg(ConvBPDN):
    r"""
    ADMM algorithm for an extension of Convolutional BPDN including a
    term penalising the gradient of the coefficient maps
    :cite:`wohlberg-2016-convolutional2`.

    |

    .. inheritance-diagram:: ConvBPDNGradReg
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
       (\mu/2) \sum_i \sum_m \| G_i \mathbf{x}_m \|_2^2 \;\;,

    where :math:`G_i` is an operator computing the derivative along index
    :math:`i`, via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 + \lambda \sum_m \| \mathbf{y}_m \|_1 +
       (\mu/2) \sum_i \sum_m \| G_i \mathbf{x}_m \|_2^2
       \quad \text{such that} \quad \mathbf{x}_m = \mathbf{y}_m \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
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

        Options include all of those defined in :class:`ConvBPDN.Options`,
        together with additional options:

          ``GradWeight`` : An array of weights :math:`w_m` for the term
          penalising the gradient of the coefficient maps. If this
          option is defined, the gradient regularization term is
          :math:`\sum_i \sum_m w_m \| G_i \mathbf{x}_m \|_2^2` where
          :math:`w_m` is the weight for filter index :math:`m`. The array
          should be an :math:`M`-vector where :math:`M` is the number of
          filters in the dictionary.
        """

        defaults = copy.deepcopy(ConvBPDN.Options.defaults)
        defaults.update({'GradWeight': 1.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDNGradReg algorithm options
            """

            if opt is None:
                opt = {}
            ConvBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegGrad')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), u('Regℓ2∇'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid',
                     u('Regℓ1'): 'RegL1', u('Regℓ2∇'): 'RegGrad'}



    def __init__(self, D, S, lmbda=None, mu=0.0, opt=None, dimK=None, dimN=2):
        """
        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdngrd_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdngrd_init.svg

        |


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

        self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        self.mu = self.dtype.type(mu)
        if hasattr(opt['GradWeight'], 'ndim'):
            self.Wgrd = np.asarray(opt['GradWeight'].reshape((1,)*(dimN+2) +
                                   opt['GradWeight'].shape), dtype=self.dtype)
        else:
            self.Wgrd = np.asarray(opt['GradWeight'], dtype=self.dtype)

        self.Gf, GHGf = gradient_filters(self.cri.dimN+3, self.cri.axisN,
                                         self.cri.Nv, dtype=self.dtype)
        self.GHGf = self.Wgrd * GHGf

        super(ConvBPDNGradReg, self).__init__(D, S, lmbda, opt, dimK=dimK,
                                              dimN=dimN)



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = self.fftn(self.D, self.cri.Nv, self.cri.axisN)
        # Compute D^H S
        self.DSf = np.conj(self.Df) * self.Sf
        if self.cri.Cd > 1:
            self.DSf = np.sum(self.DSf, axis=self.cri.axisC, keepdims=True)
        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbd_sm_c(
                self.Df, np.conj(self.Df), self.mu*self.GHGf + self.rho,
                self.cri.axisM)
        else:
            self.c = None



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.YU[:] = self.Y - self.U

        b = self.DSf + self.rho*self.fftn(self.YU, None, self.cri.axisN)
        if self.cri.Cd == 1:
            self.Xf[:] = sl.solvedbd_sm(self.Df, self.mu*self.GHGf + self.rho,
                                        b, self.c, self.cri.axisM)
        else:
            self.Xf[:] = sl.solvemdbi_ism(self.Df, self.mu*self.GHGf +
                                          self.rho, b, self.cri.axisM,
                                          self.cri.axisC)

        self.X = self.ifftn(self.Xf, self.cri.Nv, self.cri.axisN)

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
        rl1 = np.linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rgr = self.fl2norm2(np.sqrt(self.GHGf*np.conj(fvf)*fvf), self.cri.Nv,
                            self.cri.axisN)/2.0
        return (self.lmbda*rl1 + self.mu*rgr, rl1, rgr)





class ConvBPDNProjL1(GenericConvBPDN):
    r"""
    ADMM algorithm for a ConvBPDN variant with projection onto the
    :math:`\ell_1` ball instead of an :math:`\ell_1` penalty.

    |

    .. inheritance-diagram:: ConvBPDNProjL1
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 \; \text{such that} \; \sum_m \| \mathbf{x}_m \|_1
       \leq \gamma

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + \iota_{C(\gamma)}
       (\{\mathbf{y}_m\}) \quad \text{such that} \quad \mathbf{x}_m =
       \mathbf{y}_m \;\;,

    where :math:`\iota_{C(\gamma)}(\cdot)` is the indicator function
    of the :math:`\ell_1` ball of radius :math:`\gamma` about the origin.
    The algorithm is very similar to that for the CBPDN problem (see
    :class:`ConvBPDN`), the only difference being in the replacement in the
    :math:`\mathbf{y}` step of the proximal operator of the :math:`\ell_1`
    norm with the projection operator of the :math:`\ell_1` norm.
    In particular, the :math:`\mathbf{x}` step uses the solver from
    :cite:`wohlberg-2014-efficient` for single-channel dictionaries, and the
    solver from :cite:`wohlberg-2016-convolutional` for multi-channel
    dictionaries.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``Cnstr`` : Constraint violation measure

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
        """ConvBPDNProjL1 algorithm options

        Options are the same as those defined in
        :class:`.GenericConvBPDN.Options`.
        """

        defaults = copy.deepcopy(GenericConvBPDN.Options.defaults)
        defaults['AutoRho'].update({'RsdlTarget': 1.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDNProjL1 algorithm options
            """

            if opt is None:
                opt = {}
            GenericConvBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'Cnstr')
    hdrtxt_objfn = ('Fnc', 'Cnstr')
    hdrval_objfun = {'Fnc': 'ObjFun', 'Cnstr': 'Cnstr'}



    def __init__(self, D, S, gamma, opt=None, dimK=None, dimN=2):
        """
        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdnprjl1_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdnprjl1_init.svg

        |


        Parameters
        ----------
        D : array_like
          Dictionary matrix
        S : array_like
          Signal vector or matrix
        gamma : float
          Constraint parameter
        opt : :class:`ConvBPDNProjL1.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        # Set default options if necessary
        if opt is None:
            opt = ConvBPDNProjL1.Options()

        super(ConvBPDNProjL1, self).__init__(D, S, opt, dimK=dimK, dimN=dimN)

        self.gamma = self.dtype.type(gamma)



    def uinit(self, ushape):
        """Return initialiser for working variable U."""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            # NB: still needs to be worked out.
            return np.zeros(ushape, dtype=self.dtype)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y = sp.proj_l1(self.AX + self.U, self.gamma,
                            axis=self.cri.axisN + (self.cri.axisC,
                                                   self.cri.axisM))
        super(ConvBPDNProjL1, self).ystep()



    def eval_objfn(self):
        """Compute components of regularisation function as well as total
        objective function.
        """

        dfd = self.obfn_dfd()
        prj = sp.proj_l1(self.obfn_gvar(), self.gamma,
                         axis=self.cri.axisN + (self.cri.axisC,
                                                self.cri.axisM))
        cns = np.linalg.norm(prj - self.obfn_gvar())
        return (dfd, cns)





class ConvTwoBlockCnstrnt(admm.ADMMTwoBlockCnstrnt):
    r"""
    Base class for ADMM algorithms for problems of the form

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       g_0(D \mathbf{x} - \mathbf{s}) + g_1(\mathbf{x}) \;\;,

    where :math:`D \mathbf{x} = \sum_m \mathbf{d}_m * \mathbf{x}_m`.

    |

    .. inheritance-diagram:: ConvTwoBlockCnstrnt
       :parts: 2

    |

    The problem is solved via an ADMM problem of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
       g_0(\mathbf{y}_0) + g_1(\mathbf{y}_1) \;\text{such that}\;
       \left( \begin{array}{c} D \\ I \end{array} \right) \mathbf{x}
       - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1 \end{array}
       \right) = \left( \begin{array}{c} \mathbf{s} \\
       \mathbf{0} \end{array} \right) \;\;.

    In this case the ADMM constraint is :math:`A\mathbf{x} + B\mathbf{y}
    = \mathbf{c}` where

    .. math::
       A = \left( \begin{array}{c} D \\ I \end{array} \right)
       \qquad B = -I \qquad \mathbf{y} = \left( \begin{array}{c}
       \mathbf{y}_0 \\ \mathbf{y}_1 \end{array} \right) \qquad
       \mathbf{c} = \left( \begin{array}{c} \mathbf{s} \\
       \mathbf{0} \end{array} \right) \;\;.

    |

    The implementation of this class is substantially complicated by the
    support of multi-channel signals. In the following, the number of
    channels in the signal and dictionary are denoted by ``C`` and ``Cd``
    respectively, the number of signals and the number of filters are
    denoted by ``K`` and ``M`` respectively, ``D``, ``X``, and ``S`` denote
    the dictionary, coefficient map, and signal arrays respectively, and
    ``Y0`` and ``Y1`` denote blocks 0 and 1 of the auxiliary (split)
    variable of the ADMM problem. We need to consider three different cases:

      1. Single channel signal and dictionary (``C`` = ``Cd`` = 1)
      2. Multi-channel signal, single channel dictionary (``C`` > 1,
         ``Cd`` = 1)
      3. Multi-channel signal and dictionary (``C`` = ``Cd`` > 1)


    The final three (non-spatial) dimensions of the main variables in each
    of these cases are as in the following table:

      ======   ==================   =====================   ==================
      Var.     ``C`` = ``Cd`` = 1   ``C`` > 1, ``Cd`` = 1   ``C`` = ``Cd`` > 1
      ======   ==================   =====================   ==================
      ``D``    1 x 1 x ``M``        1 x 1 x ``M``           ``Cd`` x 1 x ``M``
      ``X``    1 x ``K`` x ``M``    ``C`` x ``K`` x ``M``   1 x ``K`` x ``M``
      ``S``    1 x ``K`` x 1        ``C`` x ``K`` x 1       ``C`` x ``K`` x 1
      ``Y0``   1 x ``K`` x 1        ``C`` x ``K`` x 1       ``C`` x ``K`` x 1
      ``Y1``   1 x ``K`` x ``M``    ``C`` x ``K`` x ``M``   1 x ``K`` x ``M``
      ======   ==================   =====================   ==================

    In order to combine the block components ``Y0`` and ``Y1`` of
    variable ``Y`` into a single array, we need to be able to
    concatenate the two component arrays on one of the axes. The final
    ``M`` axis is suitable in the first two cases, but it is not
    possible to concatenate ``Y0`` and ``Y1`` on the final axis in
    case 3. The solution is that, in case 3, the the ``C`` and ``M``
    axes of ``Y0`` are swapped before concatenating, as well as after
    extracting the ``Y0`` component from the concatenated ``Y``
    variable (see :meth:`.block_sep0` and :meth:`block_cat`).

    |

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
        defaults.update({'AuxVarObj': False, 'fEvalX': True,
                         'gEvalY': False, 'HighMemSolve': False,
                         'LinSolveCheck': False, 'NonNegCoef': False,
                         'NoBndryCross': False, 'RelaxParam': 1.8,
                         'rho': 1.0, 'ReturnVar': 'Y1'})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvTwoBlockCnstrnt algorithm options
            """

            if opt is None:
                opt = {}
            admm.ADMMTwoBlockCnstrnt.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'G0Val', 'G1Val')
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('Fnc', 'g0', 'g1')
    hdrval_objfun = {'Fnc': 'ObjFun', 'g0': 'G0Val', 'g1': 'G1Val'}



    def __init__(self, D, S, opt=None, dimK=None, dimN=2):
        """
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

        # Set flag indicating whether problem involves real or complex
        # values, and get appropriate versions of functions from fft
        # module
        self.real_dtype = np.isrealobj(D) and np.isrealobj(S)
        self.empty_aligned = empty_aligned_func(self.real_dtype)
        self.fftn = fftn_func(self.real_dtype)
        self.ifftn = ifftn_func(self.real_dtype)
        self.fl2norm2 = fl2norm2_func(self.real_dtype)

        # Infer problem dimensions and set relevant attributes of self
        self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Determine whether axis swapping on Y block 0 is necessary
        self.y0swapaxes = bool(self.cri.C > 1 and self.cri.Cd > 1)

        # Call parent class __init__
        Nx = self.cri.M * self.cri.N * self.cri.K
        shpY = list(self.cri.shpX)
        if self.y0swapaxes:
            shpY[self.cri.axisC] = 1
        shpY[self.cri.axisM] += self.cri.Cd
        super(ConvTwoBlockCnstrnt, self).__init__(Nx, shpY, self.cri.axisM,
                                                  self.cri.Cd, S.dtype, opt)

        # Reshape D and S to standard layout
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # Initialise byte-aligned arrays for pyfftw
        self.YU = empty_aligned(self.Y.shape, dtype=self.dtype)
        self.Xf = self.empty_aligned(self.cri.shpX, self.cri.axisN,
                                     self.dtype)

        self.setdict()



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = self.fftn(self.D, self.cri.Nv, self.cri.axisN)
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
        YUf = self.fftn(self.YU, None, self.cri.axisN)
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

        self.X = self.ifftn(self.Xf, self.cri.Nv, self.cri.axisN)

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
            self.AX = alpha*self.AXnr + (1-alpha)*self.block_cat(
                self.var_y0() + self.c0, self.var_y1() + self.c1)



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
        if self.y0swapaxes:
            return np.swapaxes(Y[(slice(None),)*self.blkaxis +
                                 (slice(0, self.blkidx),)],
                               self.cri.axisC, self.cri.axisM)
        else:
            return super(ConvTwoBlockCnstrnt, self).block_sep0(Y)



    def block_cat(self, Y0, Y1):
        r"""Concatenate components corresponding to :math:`\mathbf{y}_0`
        and :math:`\mathbf{y}_1` to form :math:`\mathbf{y}\;\;`.
        The method from parent class :class:`.ADMMTwoBlockCnstrnt` is
        overridden here to allow swapping of C (channel) and M
        (filter) axes in block 0 so that it can be concatenated on
        axis M with block 1. This is necessary because block 0 has the
        dimensions of S (N x C x K x 1) while block 1 has the
        dimensions of X (N x 1 x K x M).
        """

        if self.y0swapaxes:
            return np.concatenate((np.swapaxes(Y0, self.cri.axisC,
                                  self.cri.axisM), Y1), axis=self.blkaxis)
        else:
            return super(ConvTwoBlockCnstrnt, self).block_cat(Y0, Y1)



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
            Xf = self.fftn(X, None, self.cri.axisN)
        return self.ifftn(sl.inner(self.Df, Xf, axis=self.cri.axisM),
                          self.cri.Nv, self.cri.axisN)



    def cnst_A0T(self, Y0):
        r"""Compute :math:`A_0^T \mathbf{y}_0` component of
        :math:`A^T \mathbf{y}` (see :meth:`.ADMMTwoBlockCnstrnt.cnst_AT`).
        """

        # This calculation involves non-negligible computational cost. It
        # should be possible to disable relevant diagnostic information
        # (dual residual) to avoid this cost.
        Y0f = self.fftn(Y0, None, self.cri.axisN)
        if self.cri.Cd == 1:
            return self.ifftn(np.conj(self.Df) * Y0f, self.cri.Nv,
                              self.cri.axisN)
        else:
            return self.ifftn(sl.inner(
                np.conj(self.Df), Y0f, axis=self.cri.axisC),
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
            Xf = self.fftn(X, None, self.cri.axisN)

        Sf = np.sum(self.Df * Xf, axis=self.cri.axisM)
        return self.ifftn(Sf, self.cri.Nv, self.cri.axisN)



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        return self.rho * self.cnst_AT(self.U)



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho * np.linalg.norm(U)





class ConvMinL1InL2Ball(ConvTwoBlockCnstrnt):
    r"""
    ADMM algorithm for the problem with an :math:`\ell_1` objective and
    an :math:`\ell_2` constraint, following the general approach proposed
    in :cite:`afonso-2011-augmented`.

    |

    .. inheritance-diagram:: ConvMinL1InL2Ball
       :parts: 2

    |

    The :math:`\mathbf{y}` step is essentially the same as that of
    :class:`.admm.bpdn.MinL1InL2Ball` (with the trivial difference of a
    swap between the roles of :math:`\mathbf{y}_0` and
    :math:`\mathbf{y}_1`). The :math:`\mathbf{x}` step uses the solver
    from :cite:`wohlberg-2014-efficient` for single-channel
    dictionaries, and the solver from
    :cite:`wohlberg-2016-convolutional` for multi-channel dictionaries.

    Solve the Single Measurement Vector (SMV) problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \sum_m \| \mathbf{x}_m \|_1 \;
       \text{such that} \;  \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m
       - \mathbf{s} \right\|_2 \leq \epsilon

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
       \| \mathbf{y}_1 \|_1 + \iota_{C(\epsilon)}(\mathbf{y}_0)
       \;\text{such that}\;
       \left( \begin{array}{c} D \\ I \end{array} \right) \mathbf{x}
       - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1
       \end{array} \right) = \left( \begin{array}{c} \mathbf{s} \\
       \mathbf{0} \end{array} \right) \;\;,

    where :math:`\iota_{C(\epsilon)}(\cdot)` is the indicator
    function of the :math:`\ell_2` ball of radius :math:`\epsilon`
    about the origin, and :math:`D \mathbf{x} = \sum_m
    \mathbf{d}_m * \mathbf{x}_m`. The Multiple Measurement Vector
    (MMV) problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \sum_k \sum_m \| \mathbf{x}_{k,m} \|_1
       \; \text{such that} \; \left\|  \sum_m \mathbf{d}_m *
       \mathbf{x}_{k,m} - \mathbf{s}_k \right\|_2 \leq \epsilon \;\;\;
       \forall k \;\;,

    is also supported.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value :math:`\| \mathbf{x} \|_1`

       ``Cnstr`` : Constraint violation measure

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """


    class Options(ConvTwoBlockCnstrnt.Options):
        r"""ConvMinL1InL2Ball algorithm options

        Options include all of those defined in
        :class:`ConvTwoBlockCnstrnt.Options`, together with additional
        options:

          ``L1Weight`` : An array of weights for the :math:`\ell_1`
          norm. The array shape must be such that the array is
          compatible for multiplication with the `X`/`Y` variables (see
          :func:`.cnvrep.l1Wshape` for more details). If this
          option is defined, the objective function is :math:`\lambda \|
          \mathbf{w} \odot \mathbf{x} \|_1` where :math:`\mathbf{w}`
          denotes the weighting array.

          ``NonNegCoef`` : If ``True``, force solution to be non-negative.
        """

        defaults = copy.deepcopy(ConvTwoBlockCnstrnt.Options.defaults)
        defaults.update({'AuxVarObj': False, 'fEvalX': True,
                         'gEvalY': False, 'RelaxParam': 1.8,
                         'L1Weight': 1.0, 'NonNegCoef': False,
                         'ReturnVar': 'Y1'})
        defaults['AutoRho'].update({'Enabled': True, 'Period': 10,
                                    'AutoScaling': True, 'Scaling': 1000.0,
                                    'RsdlRatio': 1.2, 'RsdlTarget': 1.0})

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvMinL1InL2Ball algorithm options
            """

            if opt is None:
                opt = {}
            ConvTwoBlockCnstrnt.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'Cnstr')
    hdrtxt_objfn = ('Fnc', 'Cnstr')
    hdrval_objfun = {'Fnc': 'ObjFun', 'Cnstr': 'Cnstr'}



    def __init__(self, D, S, epsilon, opt=None, dimK=None, dimN=2):
        r"""

        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdnml1l2_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdnml1l2_init.svg

        |


        Parameters
        ----------
        D : array_like
          Dictionary matrix
        S : array_like
          Signal vector or matrix
        epsilon : float
          :math:`\ell_2` ball radius
        opt : :class:`ConvMinL1InL2Ball.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        if opt is None:
            opt = ConvMinL1InL2Ball.Options()

        self.S = S
        super(ConvMinL1InL2Ball, self).__init__(D, S, opt, dimK=dimK,
                                                dimN=dimN)

        # Set l1 term weight array
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)
        self.wl1 = self.wl1.reshape(cr.l1Wshape(self.wl1, self.cri))

        # Record epsilon value
        self.epsilon = self.dtype.type(epsilon)



    def uinit(self, ushape):
        """Return initialiser for working variable U."""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            U0 = np.sign(self.block_sep0(self.Y)) / self.rho
            U1 = self.block_sep1(self.Y) - sl.atleast_nd(self.cri.dimN+3,
                                                         self.S)
            return self.block_cat(U0, U1)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        AXU = self.AX + self.U
        Y0 = sp.proj_l2(self.block_sep0(AXU) - self.S, self.epsilon,
                            axis=self.cri.axisN)
        Y1 = sp.prox_l1(self.block_sep1(AXU), self.wl1 / self.rho)
        self.Y = self.block_cat(Y0, Y1)

        super(ConvMinL1InL2Ball, self).ystep()



    def obfn_g0(self, Y0):
        r"""Compute :math:`g_0(\mathbf{y}_0)` component of ADMM objective
        function.
        """

        return np.linalg.norm(sp.proj_l2(Y0, self.epsilon,
                                         axis=self.cri.axisN) - Y0)



    def obfn_g1(self, Y1):
        r"""Compute :math:`g_1(\mathbf{y_1})` component of ADMM objective
        function.
        """

        return np.linalg.norm((self.wl1 * Y1).ravel(), 1)



    def eval_objfn(self):
        """Compute components of regularisation function as well as total
        contribution to objective function.
        """

        g0v = self.obfn_g0(self.obfn_g0var())
        g1v = self.obfn_g1(self.obfn_g1var())
        return (g1v, g0v)





class ConvBPDNMaskDcpl(ConvTwoBlockCnstrnt):
    r"""
    ADMM algorithm for Convolutional BPDN with Mask Decoupling
    :cite:`heide-2015-fast`.

    |

    .. inheritance-diagram:: ConvBPDNMaskDcpl
       :parts: 2

    |

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

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
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
        :class:`ConvTwoBlockCnstrnt.Options`, together with additional
        options:

          ``L1Weight`` : An array of weights for the :math:`\ell_1`
          norm. The array shape must be such that the array is
          compatible for multiplication with the `X` variable (see
          :func:`.cnvrep.l1Wshape` for more details). If this
          option is defined, the regularization term is :math:`\lambda
          \sum_m \| \mathbf{w}_m \odot \mathbf{x}_m \|_1` where
          :math:`\mathbf{w}_m` denotes slices of the weighting array on
          the filter index axis.
        """

        defaults = copy.deepcopy(ConvTwoBlockCnstrnt.Options.defaults)
        defaults.update({'L1Weight': 1.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDNMaskDcpl algorithm options
            """

            if opt is None:
                opt = {}
            ConvTwoBlockCnstrnt.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1'}



    def __init__(self, D, S, lmbda, W=None, opt=None, dimK=None, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdnmd_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdnmd_init.svg

        |


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
          input array S (see :class:`.cnvrep.CSC_ConvRepIndexing` for a
          discussion of the distinction between *external* and *internal*
          data layouts) after reshaping to the shape determined by
          :func:`.cnvrep.mskWshape`.
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
        self.W = np.asarray(W.reshape(cr.mskWshape(W, self.cri)),
                            dtype=self.dtype)
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)
        self.wl1 = self.wl1.reshape(cr.l1Wshape(self.wl1, self.cri))



    def uinit(self, ushape):
        """Return initialiser for working variable U."""

        if self.opt['Y0'] is None:
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
        Y0 = (self.rho*(self.block_sep0(AXU) - self.S)) / \
             (self.W**2 + self.rho)
        Y1 = sp.prox_l1(self.block_sep1(AXU),
                        (self.lmbda / self.rho) * self.wl1)
        self.Y = self.block_cat(Y0, Y1)

        super(ConvBPDNMaskDcpl, self).ystep()



    def eval_objfn(self):
        """Compute components of regularisation function as well as total
        contribution to objective function.
        """

        g0v = self.obfn_g0(self.obfn_g0var())
        g1v = self.obfn_g1(self.obfn_g1var())
        obj = g0v + self.lmbda*g1v
        return (obj, g0v, g1v)



    def obfn_g0(self, Y0):
        r"""Compute :math:`g_0(\mathbf{y}_0)` component of ADMM objective
        function.
        """

        return (np.linalg.norm(self.W * Y0)**2) / 2.0



    def obfn_g1(self, Y1):
        r"""Compute :math:`g_1(\mathbf{y_1})` component of ADMM objective
        function.
        """

        return np.linalg.norm((self.wl1 * Y1).ravel(), 1)





class AddMaskSim(object):
    """Boundary masking for convolutional representations using the
    Additive Mask Simulation (AMS) technique described in
    :cite:`wohlberg-2016-boundary`. Implemented as a wrapper about a
    cbpdn.ConvBPDN or derived object (or any other object with
    sufficiently similar interface and internals). The wrapper is largely
    transparent, but must be taken into account when setting some of the
    options for the inner object, e.g. the shape of the ``L1Weight``
    option array must take into account the extra dictionary atom appended
    by the wrapper.
    """

    def __init__(self, cbpdnclass, D, S, W, *args, **kwargs):
        """
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
          compatible for multiplication with input array S (see
          :func:`.cnvrep.mskWshape` for more details).
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
        self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Construct impulse filter (or filters for the multi-channel
        # case) and append to dictionary
        if self.cri.Cd == 1:
            self.imp = np.zeros(D.shape[0:dimN] + (1,))
            self.imp[(0,)*dimN] = 1.0
        else:
            self.imp = np.zeros(D.shape[0:dimN] + (self.cri.Cd,)*2)
            for c in range(0, self.cri.Cd):
                self.imp[(0,)*dimN + (c, c,)] = 1.0
        Di = np.concatenate((D, self.imp), axis=D.ndim-1)

        # Construct inner cbpdn object
        self.cbpdn = cbpdnclass(Di, S, *args, **kwargs)

        # Required because dictlrn.DictLearn assumes that all valid
        # xstep objects have an IterationStats attribute
        self.IterationStats = self.cbpdn.IterationStats

        # Mask matrix
        self.W = np.asarray(W.reshape(cr.mskWshape(W, self.cri)),
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
        Yi[np.where(self.W.astype(bool))] = 0.0
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
        gv[..., -self.cri.Cd:] = 0

        return gv



    def solve(self):
        """Call the solve method of the inner cbpdn object and strip
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
        Xf = self.cbpdn.fftn(X, None, self.cri.axisN)
        # Multiply in frequency domain with non-impulse component of
        # dictionary
        Sf = np.sum(self.cbpdn.Df[..., 0:-self.cri.Cd] * Xf,
                    axis=self.cri.axisM)
        # Transform to spatial domain and return result
        return self.cbpdn.ifftn(Sf, self.cri.Nv, self.cri.axisN)



    def getitstat(self):
        """Get iteration stats from inner cbpdn object."""

        return self.cbpdn.getitstat()





class ConvL1L1Grd(ConvBPDNMaskDcpl):
    r"""
    ADMM algorithm for a Convolutional Sparse Coding problem with
    an :math:`\ell_1` data fidelity term and both :math:`\ell_1`
    and :math:`\ell_2` of gradient regularisation terms
    :cite:`wohlberg-2016-convolutional2`.

    |

    .. inheritance-diagram:: ConvL1L1Grd
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       \left\|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s}\right) \right\|_1 + \lambda \sum_m
       \| \mathbf{x}_m \|_1 + (\mu/2) \sum_i \sum_m
       \| G_i \mathbf{x}_m \|_2^2\;\;,

    where :math:`W` is a mask array and :math:`G_i` is an operator
    computing the derivative along index :math:`i`, via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
       \| W \mathbf{y}_0 \|_1 + \lambda \| \mathbf{y}_1 \|_1
       + (\mu/2) \sum_i \| \Gamma_i \mathbf{x} \|_2^2
       \;\text{such that}\; \left( \begin{array}{c} D \\ I \end{array}
       \right) \mathbf{x} - \left( \begin{array}{c} \mathbf{y}_0 \\
       \mathbf{y}_1 \end{array} \right) = \left( \begin{array}{c}
       \mathbf{s} \\ \mathbf{0} \end{array} \right) \;\;,

    where :math:`D \mathbf{x} = \sum_m \mathbf{d}_m * \mathbf{x}_m` and

    .. math::
       \Gamma_i = \left( \begin{array}{ccc} G_i & 0 & \ldots \\
       0 & G_i & \ldots \\ \vdots & \vdots & \ddots \end{array}
       \right) \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| W
       (\sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}) \|_1`

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

    class Options(ConvBPDNMaskDcpl.Options):
        r"""ConvL1L1Grd algorithm options

        Options include all of those defined in
        :class:`ConvBPDNMaskDcpl.Options`, together with additional
        options:

          ``GradWeight`` : An array of weights :math:`w_m` for the term
          penalising the gradient of the coefficient maps. If this
          option is defined, the gradient regularization term is
          :math:`\sum_i \sum_m w_m \| G_i \mathbf{x}_m \|_2^2` where
          :math:`w_m` is the weight for filter index :math:`m`. The array
          should be an :math:`M`-vector where :math:`M` is the number of
          filters in the dictionary.
        """

        defaults = copy.deepcopy(ConvBPDNMaskDcpl.Options.defaults)
        defaults.update({'GradWeight': 1.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvL1L1Grd algorithm options
            """

            if opt is None:
                opt = {}
            ConvBPDNMaskDcpl.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegGrad')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), u('Regℓ2∇'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid',
                     u('Regℓ1'): 'RegL1', u('Regℓ2∇'): 'RegGrad'}



    def __init__(self, D, S, lmbda, mu, W=None, opt=None, dimK=None, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/cl1l1grd_init.svg
           :width: 20%
           :target: ../_static/jonga/cl1l1grd_init.svg

        |


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
        W : array_like
          Mask array. The array shape must be such that the array is
          compatible for multiplication with input array S (see
          :func:`.cnvrep.mskWshape` for more details).
        opt : :class:`ConvL1L1Grd.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        if opt is None:
            opt = ConvL1L1Grd.Options()

        super(ConvL1L1Grd, self).__init__(D, S, lmbda, W, opt, dimK=dimK,
                                          dimN=dimN)

        self.mu = self.dtype.type(mu)
        if hasattr(opt['GradWeight'], 'ndim'):
            self.Wgrd = np.asarray(opt['GradWeight'].reshape((1,)*(dimN+2) +
                                   opt['GradWeight'].shape), dtype=self.dtype)
        else:
            self.Wgrd = np.asarray(opt['GradWeight'], dtype=self.dtype)

        self.Gf, GHGf = gradient_filters(self.cri.dimN+3, self.cri.axisN,
                                         self.cri.Nv, dtype=self.dtype)
        self.GHGf = self.Wgrd * GHGf



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = self.fftn(self.D, self.cri.Nv, self.cri.axisN)
        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbd_sm_c(
                self.Df, np.conj(self.Df),
                (self.mu / self.rho) * self.GHGf + 1.0, self.cri.axisM)
        else:
            self.c = None



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.YU[:] = self.Y - self.U
        self.block_sep0(self.YU)[:] += self.S
        YUf = self.fftn(self.YU, None, self.cri.axisN)
        if self.cri.Cd == 1:
            b = np.conj(self.Df) * self.block_sep0(YUf) + self.block_sep1(YUf)
        else:
            b = sl.inner(np.conj(self.Df), self.block_sep0(YUf),
                         axis=self.cri.axisC) + self.block_sep1(YUf)

        if self.cri.Cd == 1:
            self.Xf[:] = sl.solvedbd_sm(
                self.Df, (self.mu / self.rho) * self.GHGf + 1.0, b,
                self.c, self.cri.axisM)
        else:
            self.Xf[:] = sl.solvemdbi_ism(
                self.Df, (self.mu / self.rho) * self.GHGf + 1.0, b,
                self.cri.axisM, self.cri.axisC)

        self.X = self.ifftn(self.Xf, self.cri.Nv, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            Dop = lambda x: sl.inner(self.Df, x, axis=self.cri.axisM)
            if self.cri.Cd == 1:
                DHop = lambda x: np.conj(self.Df) * x
            else:
                DHop = lambda x: sl.inner(np.conj(self.Df), x,
                                          axis=self.cri.axisC)
            ax = DHop(Dop(self.Xf)) + ((self.mu / self.rho) * self.GHGf
                                       + 1.0) * self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        AXU = self.AX + self.U
        Y0 = sp.prox_l1(self.block_sep0(AXU) - self.S, (1.0/self.rho)*self.W)
        Y1 = sp.prox_l1(self.block_sep1(AXU), (self.lmbda/self.rho)*self.wl1)
        self.Y = self.block_cat(Y0, Y1)

        super(ConvBPDNMaskDcpl, self).ystep()



    def eval_objfn(self):
        """Compute components of regularisation function as well as total
        contribution to objective function.
        """

        g0v = self.obfn_g0(self.obfn_g0var())
        g1v = self.obfn_g1(self.obfn_g1var())
        rgr = self.fl2norm2(np.sqrt(self.GHGf * np.conj(self.Xf) * self.Xf),
                            self.cri.Nv, self.cri.axisN)/2.0
        obj = g0v + self.lmbda*g1v + self.mu*rgr
        return (obj, g0v, g1v, rgr)



    def obfn_g0(self, Y0):
        r"""Compute :math:`g_0(\mathbf{y}_0)` component of ADMM objective
        function.
        """

        return np.sum(np.abs(self.W * self.obfn_g0var()))



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        return self.rho * self.cnst_AT(Yprev - Y)



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho * np.linalg.norm(self.cnst_AT(U))



    def rhochange(self):
        """Updated cached c array when rho changes."""

        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbd_sm_c(
                self.Df, np.conj(self.Df),
                (self.mu / self.rho) * self.GHGf + 1.0, self.cri.axisM)





class MultiDictConvBPDN(object):
    r"""Solve a convolutional sparse coding problem fitting a single set of
    coefficient maps to multiple dictionaries and signals, e.g.

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| D_0 \mathbf{x} - \mathbf{s}_0 \right\|_2^2 +
       (1/2) \left\| D_1 \mathbf{x} - \mathbf{s}_1 \right\|_2^2 +
       \lambda \| \mathbf{x} \|_1 \;\;,

    for input images :math:`\mathbf{s}_0`, :math:`\mathbf{s}_1`,
    dictionaries :math:`D_0` and :math:`D_0`, and coefficient map set
    :math:`\mathbf{x}`, where :math:`D_0 \mathbf{x} = \sum_m
    \mathbf{d}_{0,m} \mathbf{x}_m` and :math:`D_1 \mathbf{x} = \sum_m
    \mathbf{d}_{1,m} \mathbf{x}_m`.

    Implemented as a wrapper about a :class:`ConvBPDN` or derived object
    (or any other object with sufficiently similar interface and
    internals).
    """

    def __init__(self, cbpdnclass, D, S, *args, **kwargs):
        """
        Parameters
        ----------
        cbpdnclass : class name
          Type of internal cbpdn object (e.g. cbpdn.ConvBPDN) to be
          constructed
        D : tuple of array_like
          Set of dictionary arrays
        S : tuple of array_like
          Set of signal arrays
        *args
          Variable length list of arguments for constructor of internal
          cbpdn object
        **kwargs
          Keyword arguments for constructor of internal
          cbpdn object
        """

        # Number of spatial dimensions
        if 'dimN' in kwargs:
            dimN = kwargs['dimN']
        else:
            dimN = 2

        # Number of channel dimensions in D[0] (should be the same for all)
        dimC = D[0].ndim - dimN - 1

        # Number of filters in D[0] (should be the same for all)
        M = D[0].shape[-1]

        # Determine spatial and channel sizes of members of dictionary set
        if dimC == 0:
            chn = [1,] * len(D)
        else:
            chn = [D[b].shape[dimN] for b in range(0, len(D))]
        C = int(np.sum(np.asarray(chn)))
        dsz = np.asarray((0,) * dimN)
        for b in range(0, len(D)):
            dsz = np.maximum(dsz, np.asarray(D[b].shape[0:dimN]))

        # Construct single dictionary array with multiple dictionaries
        # stacked on the channel index
        Dm = np.zeros(tuple(dsz.tolist()) + (C,) + (M,))
        chncs = np.cumsum(np.asarray([0,] + chn))
        slc0 = (slice(None),)*dimN + (np.newaxis,)*(1-dimC)
        for b in range(0, len(D)):
            slc1 = tuple([slice(0, n) for n in D[b].shape[0:dimN]] +
                         [slice(chncs[b], chncs[b+1])])
            Dm[slc1] = D[b][slc0]

        # Construct single signal array (this is simpler since all
        # members of the signal set are assumed to be of the same
        # size)
        Sm = np.concatenate([S[b][slc0] for b in range(0, len(S))], dimN+dimC)

        # Construct inner cbpdn object
        self.cbpdn = cbpdnclass(Dm, Sm, *args, **kwargs)

        # Record some problem parameters
        self.dimN = dimN
        self.chn = chn
        self.chncs = chncs
        self.C = C



    def solve(self):
        """Call the solve method of the inner cbpdn object and return the
        result.
        """

        # Call solve method of inner cbpdn object
        Xi = self.cbpdn.solve()
        # Copy attributes from inner cbpdn object
        self.timer = self.cbpdn.timer
        self.itstat = self.cbpdn.itstat
        # Return result of inner cbpdn object
        return Xi



    def getcoef(self):
        """Call the getcoef method if the inner cbpdn object."""

        return self.cbpdn.getcoef()



    def getitstat(self):
        """Get iteration stats from inner cbpdn object."""

        return self.cbpdn.getitstat()



    def reconstruct(self, b, X=None):
        """Reconstruct representation of signal b in signal set."""

        if X is None:
            X = self.getcoef()
        Xf = rfftn(X, None, self.cbpdn.cri.axisN)
        slc = (slice(None),)*self.dimN + \
              (slice(self.chncs[b], self.chncs[b+1]),)
        Sf = np.sum(self.cbpdn.Df[slc] * Xf, axis=self.cbpdn.cri.axisM)
        return self.cbpdn.ifftn(Sf, self.cbpdn.cri.Nv, self.cbpdn.cri.axisN)
