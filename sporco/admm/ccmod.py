# -*- coding: utf-8 -*-
# Copyright (C) 2015-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""ADMM algorithms for the Convolutional Constrained MOD problem"""

from __future__ import division, absolute_import
from builtins import range

import copy
import numpy as np

from sporco.admm import admm
import sporco.cnvrep as cr
import sporco.linalg as sl
from sporco.common import _fix_dynamic_class_lookup
from sporco.fft import (empty_aligned, real_dtype, empty_aligned_func,
                        fftn_func, ifftn_func, fl2norm2_func)

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class ConvCnstrMODBase(admm.ADMMEqual):
    r"""
    Base class for the ADMM algorithms for Convolutional Constrained MOD
    problem :cite:`wohlberg-2016-efficient`, including support for
    multi-channel signals/images :cite:`wohlberg-2016-convolutional`.

    |

    .. inheritance-diagram:: ConvCnstrMODBase
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_k \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \right\|_2^2 \quad \text{such that} \quad
       \mathbf{d}_m \in C \;\; \forall m

    where :math:`C` is the feasible set consisting of filters with unit
    norm and constrained support, via the ADMM problem

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_k \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \right\|_2^2 + \sum_m \iota_C(\mathbf{g}_m) \quad
       \text{such that} \quad \mathbf{d}_m = \mathbf{g}_m \;\;,

    where :math:`\iota_C(\cdot)` is the indicator function of feasible
    set :math:`C`. Multi-channel problems with input image channels
    :math:`\mathbf{s}_{c,k}` are also supported, either as

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_c \sum_k \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{c,k,m} -
       \mathbf{s}_{c,k} \right\|_2^2 \quad \text{such that} \quad
       \mathbf{d}_m \in C \;\; \forall m

    with single-channel dictionary filters :math:`\mathbf{d}_m` and
    multi-channel coefficient maps :math:`\mathbf{x}_{c,k,m}`, or

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_c \sum_k \left\| \sum_m \mathbf{d}_{c,m} *
       \mathbf{x}_{k,m} - \mathbf{s}_{c,k} \right\|_2^2 \quad
       \text{such that} \quad \mathbf{d}_{c,m} \in C \;\; \forall c, m

    with multi-channel dictionary filters :math:`\mathbf{d}_{c,m}` and
    single-channel coefficient maps :math:`\mathbf{x}_{k,m}`. In this
    latter case, normalisation of filters :math:`\mathbf{d}_{c,m}` is
    performed jointly over index :math:`c` for each filter :math:`m`.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
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

       ``Time`` : Cumulative run time
    """



    class Options(admm.ADMMEqual.Options):
        r"""ConvCnstrMODBase algorithm options

        Options include all of those defined in
        :class:`.admm.ADMMEqual.Options`, together with additional options:

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
                         'RelaxParam': 1.8, 'ZeroMean': False,
                         'LinSolveCheck': False})
        defaults['AutoRho'].update({'Enabled': True, 'Period': 1,
                                    'AutoScaling': True, 'Scaling': 1000,
                                    'RsdlRatio': 1.2})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvCnstrMODBase algorithm options
            """

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
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('DFid', 'Cnstr')
    hdrval_objfun = {'DFid': 'DFid', 'Cnstr': 'Cnstr'}



    def __init__(self, Z, S, dsz, opt=None, dimK=1, dimN=2):
        """
        This class supports an arbitrary number of spatial dimensions,
        `dimN`, with a default of 2. The input coefficient map array `Z`
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

        The internal data layout for S, D (X here), and X (Z here) is:
        ::

          dim<0> - dim<Nds-1> : Spatial dimensions, product of N0,N1,... is N
          dim<Nds>            : C number of channels in S and D
          dim<Nds+1>          : K number of signals in S
          dim<Nds+2>          : M number of filters in D

            sptl.      chn  sig  flt
          S(N0,  N1,   C,   K,   1)
          D(N0,  N1,   C,   1,   M)   (X here)
          X(N0,  N1,   1,   K,   M)   (Z here)

        The `dsz` parameter indicates the desired filter supports in the
        output dictionary, since this cannot be inferred from the
        input variables. The format is the same as the `dsz` parameter
        of :func:`.cnvrep.bcrop`.

        Parameters
        ----------
        Z : array_like
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
            opt = ConvCnstrMODBase.Options()

        # Set flag indicating whether problem involves real or complex
        # values, and get appropriate versions of functions from fft
        # module
        self.real_dtype = np.isrealobj(Z) and np.isrealobj(S)
        self.empty_aligned = empty_aligned_func(self.real_dtype)
        self.fftn = fftn_func(self.real_dtype)
        self.ifftn = ifftn_func(self.real_dtype)
        self.fl2norm2 = fl2norm2_func(self.real_dtype)

        # Infer problem dimensions and set relevant attributes of self
        self.cri = cr.CDU_ConvRepIndexing(dsz, S, dimK=dimK, dimN=dimN)

        # Call parent class __init__
        super(ConvCnstrMODBase, self).__init__(self.cri.shpD, S.dtype, opt)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=self.cri.K,
                      dtype=real_dtype(self.dtype))

        # Reshape S to standard layout (Z, i.e. X in cbpdn, is assumed
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
        self.Sf = self.fftn(self.S, None, self.cri.axisN)

        # Create constraint set projection function
        self.Pcn = cr.getPcn(dsz, self.cri.Nv, self.cri.dimN, self.cri.dimCd,
                             zm=opt['ZeroMean'])

        # Create byte aligned arrays for FFT calls
        self.YU = empty_aligned(self.Y.shape, dtype=self.dtype)
        self.Xf = self.empty_aligned(self.Y.shape, self.cri.axisN, self.dtype)

        if Z is not None:
            self.setcoef(Z)



    def uinit(self, ushape):
        """Return initialiser for working variable U"""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            return self.Y



    def setcoef(self, Z):
        """Set coefficient array."""

        # If the dictionary has a single channel but the input (and
        # therefore also the coefficient map array) has multiple
        # channels, the channel index and multiple image index have
        # the same behaviour in the dictionary update equation: the
        # simplest way to handle this is to just reshape so that the
        # channels also appear on the multiple image index.
        if self.cri.Cd == 1 and self.cri.C > 1:
            Z = Z.reshape(self.cri.Nv + (1,) + (self.cri.Cx*self.cri.K,) +
                          (self.cri.M,))
        self.Z = np.asarray(Z, dtype=self.dtype)

        self.Zf = self.fftn(self.Z, self.cri.Nv, self.cri.axisN)
        # Compute X^H S
        self.ZSf = sl.inner(np.conj(self.Zf), self.Sf, self.cri.axisK)



    def getdict(self, crop=True):
        """Get final dictionary. If ``crop`` is ``True``, apply
        :func:`.cnvrep.bcrop` to returned array.
        """

        D = self.Y
        if crop:
            D = cr.bcrop(D, self.cri.dsz, self.cri.dimN)
        return D



    def xstep_check(self, b):
        r"""Check the minimisation of the Augmented Lagrangian with
        respect to :math:`\mathbf{x}` by method `xstep` defined in
        derived classes. This method should be called at the end of any
        `xstep` method.
        """

        if self.opt['LinSolveCheck']:
            Zop = lambda x: sl.inner(self.Zf, x, axis=self.cri.axisM)
            ZHop = lambda x: sl.inner(np.conj(self.Zf), x,
                                      axis=self.cri.axisK)
            ax = ZHop(Zop(self.Xf)) + self.rho*self.Xf
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
            self.fftn(self.Y, None, self.cri.axisN)



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

        Ef = sl.inner(self.Zf, self.obfn_fvarf(), axis=self.cri.axisM) - \
          self.Sf
        return self.fl2norm2(Ef, self.S.shape, axis=self.cri.axisN) / 2.0



    def obfn_cns(self):
        r"""Compute constraint violation measure :math:`\| P(\mathbf{y}) -
        \mathbf{y}\|_2`.
        """

        return np.linalg.norm((self.Pcn(self.obfn_gvar()) - self.obfn_gvar()))



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs,)



    def reconstruct(self, D=None):
        """Reconstruct representation."""

        if D is None:
            Df = self.Xf
        else:
            Df = self.fftn(D, None, self.cri.axisN)

        Sf = np.sum(self.Zf * Df, axis=self.cri.axisM)
        return self.ifftn(Sf, self.cri.Nv, self.cri.axisN)





class ConvCnstrMOD_IterSM(ConvCnstrMODBase):
    r"""
    ADMM algorithm for Convolutional Constrained MOD problem with the
    :math:`\mathbf{x}` step solved via iterated application of the
    Sherman-Morrison equation :cite:`wohlberg-2016-efficient`.

    |

    .. inheritance-diagram:: ConvCnstrMOD_IterSM
       :parts: 2

    |

    Multi-channel signals/images are supported
    :cite:`wohlberg-2016-convolutional`. See :class:`.ConvCnstrMODBase`
    for interface details.
    """


    class Options(ConvCnstrMODBase.Options):
        """ConvCnstrMOD_IterSM algorithm options

        Options are the same as those defined in
        :class:`.ConvCnstrMODBase.Options`.
        """

        defaults = copy.deepcopy(ConvCnstrMODBase.Options.defaults)


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvCnstrMOD_IterSM algorithm options
            """

            if opt is None:
                opt = {}
            ConvCnstrMODBase.Options.__init__(self, opt)



    def __init__(self, Z, S, dsz, opt=None, dimK=1, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/ccmodism_init.svg
           :width: 20%
           :target: ../_static/jonga/ccmodism_init.svg
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvCnstrMOD_IterSM.Options()

        super(ConvCnstrMOD_IterSM, self).__init__(Z, S, dsz, opt, dimK, dimN)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{x}`.
        """

        self.YU[:] = self.Y - self.U
        b = self.ZSf + self.rho*self.fftn(self.YU, None, self.cri.axisN)
        self.Xf[:] = sl.solvemdbi_ism(self.Zf, self.rho, b, self.cri.axisM,
                                      self.cri.axisK)
        self.X = self.ifftn(self.Xf, self.cri.Nv, self.cri.axisN)
        self.xstep_check(b)





class ConvCnstrMOD_CG(ConvCnstrMODBase):
    r"""
    ADMM algorithm for the Convolutional Constrained MOD problem with the
    :math:`\mathbf{x}` step solved via Conjugate Gradient (CG)
    :cite:`wohlberg-2016-efficient`.

    |

    .. inheritance-diagram:: ConvCnstrMOD_CG
       :parts: 2

    |

    Multi-channel signals/images are supported
    :cite:`wohlberg-2016-convolutional`. See
    :class:`.ConvCnstrMODBase` for interface details.
    """


    class Options(ConvCnstrMODBase.Options):
        """ConvCnstrMOD_CG algorithm options

        Options include all of those defined in
        :class:`.ConvCnstrMODBase.Options`, together with
        additional options:

          ``CG`` : CG solver options

            ``MaxIter`` : Maximum CG iterations.

            ``StopTol`` : CG stopping tolerance.
        """

        defaults = copy.deepcopy(ConvCnstrMODBase.Options.defaults)
        defaults.update({'CG': {'MaxIter': 1000, 'StopTol': 1e-3}})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvCnstrMOD_CG algorithm options
            """

            if opt is None:
                opt = {}
            ConvCnstrMODBase.Options.__init__(self, opt)



    itstat_fields_extra = ('XSlvRelRes', 'XSlvCGIt')



    def __init__(self, Z, S, dsz, opt=None, dimK=1, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/ccmodcg_init.svg
           :width: 20%
           :target: ../_static/jonga/ccmodcg_init.svg
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvCnstrMOD_CG.Options()

        super(ConvCnstrMOD_CG, self).__init__(Z, S, dsz, opt, dimK, dimN)
        self.Xf[:] = 0.0



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{x}`.
        """

        self.cgit = None
        self.YU[:] = self.Y - self.U
        b = self.ZSf + self.rho*self.fftn(self.YU, None, self.cri.axisN)
        self.Xf[:], cgit = sl.solvemdbi_cg(self.Zf, self.rho, b,
                                           self.cri.axisM, self.cri.axisK,
                                           self.opt['CG', 'StopTol'],
                                           self.opt['CG', 'MaxIter'], self.Xf)
        self.cgit = cgit
        self.X = self.ifftn(self.Xf, self.cri.Nv, self.cri.axisN)
        self.xstep_check(b)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs, self.cgit)





class ConvCnstrMOD_Consensus(admm.ADMMConsensus):
    r"""
    ADMM algorithm for the Convolutional Constrained MOD problem
    with the :math:`\mathbf{x}` step solved via an ADMM consensus problem
    :cite:`boyd-2010-distributed` (Ch. 7), :cite:`sorel-2016-fast`.

    |

    .. inheritance-diagram:: ConvCnstrMOD_Consensus
       :parts: 2

    |

    Multi-channel signals/images are supported
    :cite:`wohlberg-2016-convolutional`. See :class:`.ConvCnstrMODBase`
    for interface details.
    """


    class Options(admm.ADMMConsensus.Options, ConvCnstrMODBase.Options):
        """ConvCnstrMOD_Consensus algorithm options

        Available options are the same as those defined in
        :class:`.ADMMConsensus.Options` and :class:`ConvCnstrMODBase.Options`.
        """

        defaults = copy.deepcopy(ConvCnstrMODBase.Options.defaults)
        defaults.update(admm.ADMMConsensus.Options.defaults)
        defaults.update({'RelaxParam': 1.8})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvCnstrMOD_Consensus algorithm options
            """

            if opt is None:
                opt = {}
            ConvCnstrMODBase.Options.__init__(self, opt)
            admm.ADMMConsensus.Options.__init__(self, opt)



    itstat_fields_objfn = ('DFid', 'Cnstr')
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('DFid', 'Cnstr')
    hdrval_objfun = {'DFid': 'DFid', 'Cnstr': 'Cnstr'}



    def __init__(self, Z, S, dsz, opt=None, dimK=1, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/ccmodcnsns_init.svg
           :width: 20%
           :target: ../_static/jonga/ccmodcnsns_init.svg
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvCnstrMOD_Consensus.Options()

        # Set flag indicating whether problem involves real or complex
        # values, and get appropriate versions of functions from fft
        # module
        self.real_dtype = np.isrealobj(Z) and np.isrealobj(S)
        self.empty_aligned = empty_aligned_func(self.real_dtype)
        self.fftn = fftn_func(self.real_dtype)
        self.ifftn = ifftn_func(self.real_dtype)
        self.fl2norm2 = fl2norm2_func(self.real_dtype)

        # Infer problem dimensions and set relevant attributes of self
        self.cri = cr.CDU_ConvRepIndexing(dsz, S, dimK=dimK, dimN=dimN)

        # Handle possible reshape of channel axis onto multiple image axis
        # (see comment below)
        Nb = self.cri.K if self.cri.C == self.cri.Cd else \
             self.cri.C * self.cri.K
        admm.ADMMConsensus.__init__(self, Nb, self.cri.shpD, S.dtype, opt)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=self.cri.K,
                      dtype=real_dtype(self.dtype))

        # Reshape S to standard layout (Z, i.e. X in cbpdn, is assumed
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
        self.Sf = self.fftn(self.S, None, self.cri.axisN)

        # Create constraint set projection function
        self.Pcn = cr.getPcn(dsz, self.cri.Nv, self.cri.dimN, self.cri.dimCd,
                             zm=opt['ZeroMean'])

        if Z is not None:
            self.setcoef(Z)

        self.X = empty_aligned(self.xshape, dtype=self.dtype)
        # See comment on corresponding test in xstep method
        if self.cri.Cd > 1:
            self.YU = empty_aligned(self.yshape, dtype=self.dtype)
        else:
            self.YU = empty_aligned(self.xshape, dtype=self.dtype)
        self.Xf = self.empty_aligned(self.xshape, self.cri.axisN, self.dtype)



    def uinit(self, ushape):
        """Return initialiser for working variable U."""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            return np.repeat(self.Y[..., np.newaxis],
                             self.Nb, axis=-1)/self.rho



    def setcoef(self, Z):
        """Set coefficient array."""

        # If the dictionary has a single channel but the input (and
        # therefore also the coefficient map array) has multiple
        # channels, the channel index and multiple image index have
        # the same behaviour in the dictionary update equation: the
        # simplest way to handle this is to just reshape so that the
        # channels also appear on the multiple image index.
        if self.cri.Cd == 1 and self.cri.C > 1:
            Z = Z.reshape(self.cri.Nv + (1,) + (self.cri.Cx*self.cri.K,) +
                          (self.cri.M,))
        self.Z = np.asarray(Z, dtype=self.dtype)

        self.Zf = self.fftn(self.Z, self.cri.Nv, self.cri.axisN)
        # Compute X^H S
        self.ZSf = np.conj(self.Zf) * self.Sf



    def swapaxes(self, x):
        """Class :class:`.admm.ADMMConsensus`, from which this class is
        derived, expects the multiple blocks of a consensus problem to
        be stacked on the final axis. For compatibility with this
        requirement, ``axisK`` of the variables used in this algorithm is
        swapped with a new final axis. This method undoes the swap and
        removes the final axis for compatibility with functions that
        expect the variables in standard layout.
        """

        return np.swapaxes(x, self.cri.axisK, -1)[..., 0]



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to block vector
        :math:`\mathbf{x} = \left( \begin{array}{ccc} \mathbf{x}_0^T &
        \mathbf{x}_1^T & \ldots \end{array} \right)^T\;`.
        """

        # This test reflects empirical evidence that two slightly
        # different implementations are faster for single or
        # multi-channel data. This kludge is intended to be temporary.
        if self.cri.Cd > 1:
            for i in range(self.Nb):
                self.xistep(i)
        else:
            self.YU[:] = self.Y[..., np.newaxis] - self.U
            b = np.swapaxes(self.ZSf[..., np.newaxis], self.cri.axisK, -1) \
                + self.rho*self.fftn(self.YU, None, self.cri.axisN)
            for i in range(self.Nb):
                self.Xf[..., i] = sl.solvedbi_sm(
                    self.Zf[..., [i], :], self.rho, b[..., i],
                    axis=self.cri.axisM)
            self.X = self.ifftn(self.Xf, self.cri.Nv, self.cri.axisN)


        if self.opt['LinSolveCheck']:
            ZSfs = np.sum(self.ZSf, axis=self.cri.axisK, keepdims=True)
            YU = np.sum(self.Y[..., np.newaxis] - self.U, axis=-1)
            b = ZSfs + self.rho*self.fftn(YU, None, self.cri.axisN)
            Xf = self.swapaxes(self.Xf)
            Zop = lambda x: sl.inner(self.Zf, x, axis=self.cri.axisM)
            ZHop = lambda x: np.conj(self.Zf) * x
            ax = np.sum(ZHop(Zop(Xf)) + self.rho*Xf, axis=self.cri.axisK,
                        keepdims=True)
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def xistep(self, i):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{x}`
        component :math:`\mathbf{x}_i`.
        """

        self.YU[:] = self.Y - self.U[..., i]
        b = np.take(self.ZSf, [i], axis=self.cri.axisK) + \
            self.rho*self.fftn(self.YU, None, self.cri.axisN)

        self.Xf[..., i] = sl.solvedbi_sm(np.take(
            self.Zf, [i], axis=self.cri.axisK),
                                         self.rho, b, axis=self.cri.axisM)
        self.X[..., i] = self.ifftn(self.Xf[..., i], self.cri.Nv,
                                    self.cri.axisN)



    def prox_g(self, X, rho):
        """Proximal operator of :math:`g`"""

        return self.Pcn(X)



    def getdict(self, crop=True):
        """Get final dictionary. If ``crop`` is ``True``, apply
        :func:`.cnvrep.bcrop` to returned array.
        """

        D = self.Y
        if crop:
            D = cr.bcrop(D, self.cri.dsz, self.cri.dimN)
        return D



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_dfd()
        cns = self.obfn_cns()
        return (dfd, cns)



    def obfn_fvarf(self):
        """Variable to be evaluated in computing data fidelity term,
        depending on 'fEvalX' option value.
        """

        if self.opt['fEvalX']:
            return self.swapaxes(self.Xf)
        else:
            return self.fftn(self.Y, None, self.cri.axisN)



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| \sum_m
        \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`.
        """

        Ef = sl.inner(self.Zf, self.obfn_fvarf(), axis=self.cri.axisM) \
          - self.Sf
        return self.fl2norm2(Ef, self.S.shape, axis=self.cri.axisN) / 2.0



    def obfn_cns(self):
        r"""Compute constraint violation measure :math:`\| P(\mathbf{y})
        - \mathbf{y}\|_2`.
        """

        Y = self.obfn_gvar()
        return np.linalg.norm((self.Pcn(Y) - Y))



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs,)





def ConvCnstrMOD(*args, **kwargs):
    """A wrapper function that dynamically defines a class derived from
    one of the implementations of the Convolutional Constrained MOD
    problems, and returns an object instantiated with the provided
    parameters. The wrapper is designed to allow the appropriate
    object to be created by calling this function using the same
    syntax as would be used if it were a class. The specific
    implementation is selected by use of an additional keyword
    argument 'method'. Valid values are:

    - ``'ism'`` :
      Use the implementation defined in :class:`.ConvCnstrMOD_IterSM`. This
      method works well for a small number of training images, but is very
      slow for larger training sets.
    - ``'cg'`` :
      Use the implementation defined in :class:`.ConvCnstrMOD_CG`. This
      method is slower than ``'ism'`` for small training sets, but has better
      run time scaling as the training set grows.
    - ``'cns'`` :
      Use the implementation defined in :class:`.ConvCnstrMOD_Consensus`.
      This method is the best choice for large training sets.

    The default value is ``'cns'``.
    """

    # Extract method selection argument or set default
    if 'method' in kwargs:
        method = kwargs['method']
        del kwargs['method']
    else:
        method = 'cns'

    # Assign base class depending on method selection argument
    if method == 'ism':
        base = ConvCnstrMOD_IterSM
    elif method == 'cg':
        base = ConvCnstrMOD_CG
    elif method == 'cns':
        base = ConvCnstrMOD_Consensus
    else:
        raise ValueError('Unknown ConvCnstrMOD solver method %s' % method)

    # Nested class with dynamically determined inheritance
    class ConvCnstrMOD(base):
        def __init__(self, *args, **kwargs):
            super(ConvCnstrMOD, self).__init__(*args, **kwargs)

    # Allow pickling of objects of type ConvCnstrMOD
    _fix_dynamic_class_lookup(ConvCnstrMOD, method)

    # Return object of the nested class type
    return ConvCnstrMOD(*args, **kwargs)




def ConvCnstrMODOptions(opt=None, method='cns'):
    """A wrapper function that dynamically defines a class derived from
    the Options class associated with one of the implementations of
    the Convolutional Constrained MOD problem, and returns an object
    instantiated with the provided parameters. The wrapper is designed
    to allow the appropriate object to be created by calling this
    function using the same syntax as would be used if it were a
    class. The specific implementation is selected by use of an
    additional keyword argument 'method'. Valid values are as
    specified in the documentation for :func:`ConvCnstrMOD`.
    """

    # Assign base class depending on method selection argument
    if method == 'ism':
        base = ConvCnstrMOD_IterSM.Options
    elif method == 'cg':
        base = ConvCnstrMOD_CG.Options
    elif method == 'cns':
        base = ConvCnstrMOD_Consensus.Options
    else:
        raise ValueError('Unknown ConvCnstrMOD solver method %s' % method)

    # Nested class with dynamically determined inheritance
    class ConvCnstrMODOptions(base):
        def __init__(self, opt):
            super(ConvCnstrMODOptions, self).__init__(opt)

    # Allow pickling of objects of type ConvCnstrMODOptions
    _fix_dynamic_class_lookup(ConvCnstrMODOptions, method)

    # Return object of the nested class type
    return ConvCnstrMODOptions(opt)
