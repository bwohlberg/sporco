#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithm for the Convolutional BPDN problem"""

from __future__ import division
from __future__ import absolute_import
from builtins import range

import numpy as np
from scipy import linalg
import copy
import collections

from sporco.admm import admm
import sporco.linalg as sl

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class ConvRepIndexing(object):
    """Manage the inference of problem dimensions and the roles of ndarray
    indices for convolutional representations as in :class:`.ConvBPDN`
    and related classes.
    """

    def __init__(self, D, S, dimN=2):
        """Initialise a ConvRepIndexing object, inferring the problem
        dimensions from input dictionary and signal arrays D and S
        respectively.

        The internal data layout for S, D, and X is:
        ::

          dim<0> - dim<Nds-1> : Spatial dimensions, product of N0,N1,... is N
          dim<Nds>            : C number of channels in S and D
          dim<Nds+1>          : K number of signals in S
          dim<Nds+2>          : M number of filters in D

            sptl.      chn  sig  flt
          S(N0,  N1,   C,   K,   1)
          D(N0,  N1,   C,   1,   M)
          X(N0,  N1,   1,   K,   M)
        """

        # Numbers of spatial, channel, and signal dimensions in
        # external D and S. These need to be calculated since inputs D
        # and S do not already have the standard data layout above,
        # i.e. singleton dimensions will not be present
        self.dimN = dimN                  # Number of spatial dimensions
        self.dimC = D.ndim-dimN-1         # Number of channel dimensions in D
        self.dimK = S.ndim-dimN-self.dimC # Number of signal dimensions in S

        # Number of channels in external D and S
        if self.dimC == 1:
            self.C = D.shape[dimN]
        else:
            self.C = 1

        # Number of signals in external S
        if self.dimK == 1:
            self.K = S.shape[self.dimN+self.dimC]
        else:
            self.K = 1

        # Number of filters
        self.M = D.shape[self.dimN+self.dimC]
        # Shape of spatial indices and number of spatial samples
        self.Nv = S[(slice(None),)*dimN + (1,)*(self.dimC+self.dimK)].shape
        self.N = np.prod(np.array(self.Nv))

        # Axis indices for each component of X and internal S and D
        self.axisN = tuple(range(0, dimN))
        self.axisC = dimN
        self.axisK = dimN + 1
        self.axisM = dimN + 2

        # Shapes of internal S, D, and X
        self.shpD = D.shape[0:dimN] + (self.C,) + (1,) + (self.M,)
        self.shpS = self.Nv + (self.C,) + (self.K,) + (1,)
        self.shpX = self.Nv + (1,) + (self.K,) + (self.M,)



class ConvBPDN(admm.ADMMEqual):

    """ADMM algorithm for the Convolutional BPDN (CBPDN)
    :cite:`wohlberg-2014-efficient` :cite:`wohlberg-2016-efficient`
    :cite:`wohlberg-2016-convolutional` problem.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \\left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \\right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \\left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \\right\|_2^2 + \lambda \sum_m \| \mathbf{y}_m \|_1
       \quad \\text{such that} \quad \mathbf{x}_m = \mathbf{y}_m \;\;.


    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term \
       :math:`(1/2) \| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term \
       :math:`\sum_m \| \mathbf{x}_m \|_1`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance \
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance \
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

        ``AuxVarObj`` : Flag indicating whether the objective function
        should be evaluated using variable X  (``False``) or Y (``True``)
        as its argument.

        ``LinSolveCheck`` : Flag indicating whether to compute
        relative residual of X step solver.

        ``HighMemSolve`` : Flag indicating whether to use a slightly
        faster algorithm at the expense of higher memory usage.

        ``L1Weight`` : An array of weights for the :math:`\ell^1`
        norm. The array shape must be such that the array is
        compatible for multiplication with the X/Y variables. If this
        option is defined, the regularization term is :math:`\lambda  \sum_m
        \| \mathbf{w}_m \odot \mathbf{x}_m \|_1` where :math:`\mathbf{w}_m`
        denotes slices of the weighting array on the filter index axis.

        ``NonNegCoef`` : Flag indicating whether to force solution to
        be non-negative.

        ``NoBndryCross`` : Flag indicating whether all solution
        coefficients corresponding to filters crossing the image
        boundary should be forced to zero.
        """

        defaults = copy.deepcopy(admm.ADMMEqual.Options.defaults)
        defaults.update({'AuxVarObj' : False,  'ReturnX' : False,
                         'HighMemSolve' : False, 'LinSolveCheck' : False,
                         'RelaxParam' : 1.8, 'L1Weight' : 1.0,
                         'NonNegCoef' : False, 'NoBndryCross' : False})
        defaults['AutoRho'].update({'Enabled' : True, 'Period' : 1,
                                    'AutoScaling' : True, 'Scaling' : 1000.0,
                                    'RsdlRatio' : 1.2})


        def __init__(self, opt=None):
            """Initialise CBPDN algorithm options object"""

            if opt is None:
                opt = {}
            admm.ADMMEqual.Options.__init__(self, opt)

            if self['AuxVarObj']:
                self['fEvalX'] = False
                self['gEvalY'] = True
            else:
                self['fEvalX'] = True
                self['gEvalY'] = False


        def set_lambda(self, lmbda, override=False):
            """Set parameters depending on lambda value"""

            if override or self['rho'] is None:
                self['rho'] = 50.0*lmbda + 1.0
            if override or self['AutoRho','RsdlTarget'] is None:
                self['AutoRho','RsdlTarget'] = 1.0 + \
                  (18.3)**(np.log10(lmbda)+1.0)


    IterationStats = collections.namedtuple('IterationStats',
                ['Iter', 'ObjFun', 'DFid', 'RegL1', 'PrimalRsdl', 'DualRsdl',
                 'EpsPrimal', 'EpsDual', 'Rho', 'XSlvRelRes', 'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    hdrtxt = ['Itn', 'Fnc', 'DFid', 'l1', 'r', 's', 'rho']
    """Display column header text"""
    hdrval = {'Itn' : 'Iter', 'Fnc' : 'ObjFun', 'DFid' : 'DFid',
              'l1' : 'RegL1', 'r' : 'PrimalRsdl', 's' : 'DualRsdl',
              'rho' : 'Rho'}
    """Dictionary mapping display column headers to IterationStats entries"""



    def __init__(self, D, S, lmbda=None, opt=None, dimN=2):
        """
        Initialise a ConvBPDN object with problem parameters.

        This class supports an arbitrary number of spatial dimensions,
        dimN, with a default of 2. The input dictionary D is either
        dimN+1 dimensional, in which case each spatial component
        (image in the default case) is assumed to consist of a single
        channel, or dimN+2 dimensional, in which case the final
        dimension is assumed to contain the channels (e.g. colour
        channels in the case of images). The input signal set S is
        either dimN dimensional (no channels, only one signal), dimN+1
        dimensional (either multiple channels or multiple signals), or
        dimN+2 dimensional (multiple channels and multiple signals).


        The internal data layout for S, D, and X is:
        ::

          dim<0> - dim<Nds-1> : Spatial dimensions, product of N0,N1,... is N
          dim<Nds>            : C number of channels in S and D
          dim<Nds+1>          : K number of signals in S
          dim<Nds+2>          : M number of filters in D

            sptl.      chn  sig  flt
          S(N0,  N1,   C,   K,   1)
          D(N0,  N1,   C,   1,   M)
          X(N0,  N1,   1,   K,   M)


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
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvBPDN.Options()

        # Infer problem dimensions and set relevant attributes of self
        cri = ConvRepIndexing(D, S, dimN)
        for attr in ['dimN', 'dimC', 'dimK', 'C', 'K', 'M', 'Nv', 'N',
                     'axisN', 'axisC', 'axisK', 'axisM']:
            setattr(self, attr, getattr(cri, attr))

        # Call parent class __init__
        Nx = self.M*self.N
        super(ConvBPDN, self).__init__(Nx, opt)

        # Reshape D and S to standard layout
        self.D = D.reshape(cri.shpD)
        self.S = S.reshape(cri.shpS)

        # Compute signal in DFT domain
        self.Sf = sl.rfftn(self.S, None, self.axisN)

        # Set default lambda value if not specified
        if lmbda is None:
            Df = sl.rfftn(self.D, self.Nv, self.axisN)
            b = np.conj(Df) * self.Sf
            self.lmbda = 0.1*abs(b).max()
        else:
            self.lmbda = lmbda

        # Set rho value (computed from lambda if not specified)
        self.opt.set_lambda(self.lmbda)
        self.rho = self.opt['rho']

        # Determine working data type
        self.opt.set_dtype(S.dtype)
        dtype = self.opt['DataType']

        # Initial values for Y
        if self.opt['Y0'] is None:
            self.Y = np.zeros(cri.shpX, dtype)
        else:
            self.Y = self.opt['Y0']
        self.Yprev = self.Y

        # Initial value for U
        if self.opt['U0'] is None:
            if self.opt['Y0'] is None:
                self.U = np.zeros(cri.shpX, dtype)
            else:
                # If Y0 is given, but not U0, then choose the initial
                # U so that the relevant dual optimality criterion
                # (see (3.10) in boyd-2010-distributed) is satisfied.
                self.U = (self.lmbda/self.rho)*np.sign(self.Y)
        else:
            self.U = self.opt['U0']

        # Initialise byte-aligned arrays for pyfftw
        self.YU = sl.pyfftw_empty_aligned(self.Y.shape, dtype=dtype)
        xfshp = list(self.Y.shape)
        xfshp[dimN-1] = xfshp[dimN-1]//2 + 1
        self.Xf = sl.pyfftw_empty_aligned(xfshp, dtype=sl.complex_dtype(dtype))
        self.runtime += self.timer.elapsed()

        self.setdict()



    def setdict(self, D=None):
        """Set dictionary array."""

        self.timer.start()
        if D is not None:
            self.D = D
        self.Df = sl.rfftn(self.D, self.Nv, self.axisN)
        # Compute D^H S
        self.DSf = np.sum(np.conj(self.Df) * self.Sf, axis=self.axisC,
                          keepdims=True)
        if self.opt['HighMemSolve'] and self.C == 1:
            self.c = sl.solvedbi_sm_c(self.Df, np.conj(self.Df), self.rho,
                                      self.axisM)
        else:
            self.c = None
        self.runtime += self.timer.elapsed()




    def xstep(self):
        """Minimise Augmented Lagrangian with respect to x."""

        self.YU[:] = self.Y - self.U

        b = self.DSf + self.rho*sl.rfftn(self.YU, None, self.axisN)
        if self.C == 1:
            self.Xf[:] = sl.solvedbi_sm(self.Df, self.rho, b, self.c,
                                        self.axisM)
        else:
            self.Xf[:] = sl.solvemdbi_ism(self.Df, self.rho, b, self.axisM,
                                          self.axisC)

        self.X = sl.irfftn(self.Xf, self.Nv, self.axisN)

        if self.opt['LinSolveCheck']:
            Dop = lambda x: np.sum(self.Df * x, axis=self.axisM, keepdims=True)
            DHop = lambda x: np.sum(np.conj(self.Df) * x, axis=self.axisC,
                                    keepdims=True)
            ax = DHop(Dop(self.Xf)) + self.rho*self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        """Minimise Augmented Lagrangian with respect to y."""

        self.Y = sl.shrink1(self.AX + self.U,
                            (self.lmbda/self.rho)*self.opt['L1Weight'])
        if self.opt['NonNegCoef']:
            self.Y[self.Y < 0.0] = 0.0
        if self.opt['NoBndryCross']:
            for n in range(0, self.dimN):
                self.Y[(slice(None),)*n +(slice(1-self.D.shape[n],None),)] = 0.0



    def obfn_fvarf(self):
        """Variable to be evaluated in computing data fidelity term,
        depending on ``fEvalX`` option value."""

        return self.Xf if self.opt['fEvalX'] else \
            sl.rfftn(self.Y, None, self.axisN)



    def iteration_stats(self, k, r, s, epri, edua, tk):
        """
        Construct iteration stats record tuple. Data fidelity term is
        :math:`(1/2) \|  \sum_m \mathbf{d}_m * \mathbf{x}_m -
        \mathbf{s} \|_2^2` and regularisation term is
        :math:`\sum_m \| \mathbf{x}_m \|_1`.
        """

        Ef = np.sum(self.Df * self.obfn_fvarf(), axis=self.axisM,
                    keepdims=True) - self.Sf
        dfd = sl.rfl2norm2(Ef, self.S.shape, axis=tuple(range(self.dimN)))/2.0
        reg = linalg.norm((self.opt['L1Weight'] * self.obfn_gvar()).ravel(), 1)
        obj = dfd + self.lmbda*reg
        itst = type(self).IterationStats(k, obj, dfd, reg, r, s,
                                         epri, edua, self.rho, self.xrrs, tk)
        return itst



    def rhochange(self):
        """Updated cached c array when rho changes."""

        if self.opt['HighMemSolve']:
            self.c = self.Df / (np.sum(self.Df * np.conj(self.Df),
                     axis=self.axisM, keepdims=True) + self.rho)



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            X = self.Y
        Xf = sl.rfftn(X, None, self.axisN)
        Sf = np.sum(self.Df * Xf, axis=self.axisM)
        return sl.irfftn(Sf, self.Nv, self.axisN)





class ConvElasticNet(ConvBPDN):
    """ADMM algorithm for a convolutional form of the elastic net problem
    :cite:`zou-2005-regularization`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \\left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \\right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
       (\mu/2) \sum_m \| \mathbf{x}_m \|_2^2

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \\left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \\right\|_2^2 + \lambda \sum_m \| \mathbf{y}_m \|_1
       + (\mu/2) \sum_m \| \mathbf{x}_m \|_2^2
       \quad \\text{such that} \quad \mathbf{x}_m = \mathbf{y}_m \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term \
       :math:`(1/2) \|  \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term \
       :math:`\sum_m \| \mathbf{x}_m \|_1`

       ``RegL2`` : Value of regularisation term \
       :math:`(1/2) \sum_m \| \mathbf{x}_m \|_2^2`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance \
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance \
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Relative residual of X step solver

       ``Time`` : Cumulative run time
    """

    IterationStats = collections.namedtuple('IterationStats',
                ['Iter', 'ObjFun', 'DFid', 'RegL1', 'RegL2',
                 'PrimalRsdl', 'DualRsdl', 'EpsPrimal', 'EpsDual',
                  'Rho', 'XSlvRelRes', 'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    hdrtxt = ['Itn', 'Fnc', 'DFid', 'l1', 'l2', 'r', 's', 'rho']
    """Display column header text"""

    hdrval = {'Itn' : 'Iter', 'Fnc' : 'ObjFun', 'DFid' : 'DFid',
              'l1' : 'RegL1', 'l2' : 'RegL2', 'r' : 'PrimalRsdl',
              's' : 'DualRsdl', 'rho' : 'Rho'}
    """Dictionary mapping display column headers to IterationStats entries"""



    def __init__(self, D, S, lmbda=None, mu=0.0, opt=None, dimN=2):
        """
        Initialise a ConvElasticNet object with problem parameters.

        Parameters
        ----------
        D : array_like, shape (N, M)
          Dictionary matrix
        S : array_like, shape (M, K)
          Signal vector or matrix
        lmbda : float
          Regularisation parameter (l1)
        mu : float
          Regularisation parameter (l2)
        opt : :class:`ConvBPDN.Options` object
          Algorithm options
        """

        if opt is None:
            opt = ConvBPDN.Options()
        self.mu = mu
        super(ConvElasticNet, self).__init__(D, S, lmbda, opt, dimN)




    def setdict(self, D=None):
        """Set dictionary array."""

        self.timer.start()
        if D is not None:
            self.D = D
        self.Df = sl.rfftn(self.D, self.Nv, self.axisN)
        # Compute D^H S
        self.DSf = np.sum(np.conj(self.Df) * self.Sf, axis=self.axisC,
                          keepdims=True)
        if self.opt['HighMemSolve'] and self.C == 1:
            self.c = sl.solvedbi_sm_c(self.Df, np.conj(self.Df),
                                      self.mu + self.rho, self.axisM)
        else:
            self.c = None
        self.runtime += self.timer.elapsed()



    def xstep(self):
        """Minimise Augmented Lagrangian with respect to x."""

        self.YU[:] = self.Y - self.U

        b = self.DSf + self.rho*sl.rfftn(self.YU, None, self.axisN)
        if self.C == 1:
            self.Xf[:] = sl.solvedbi_sm(self.Df, self.mu + self.rho,
                                        b, self.c, self.axisM)
        else:
            self.Xf[:] = sl.solvemdbi_ism(self.Df, self.mu + self.rho, b,
                                          self.axisM, self.axisC)

        self.X = sl.irfftn(self.Xf, None, self.axisN)

        if self.opt['LinSolveCheck']:
            Dop = lambda x: np.sum(self.Df * x, axis=self.axisM, keepdims=True)
            DHop = lambda x: np.sum(np.conj(self.Df) * x, axis=self.axisC,
                                    keepdims=True)
            ax = DHop(Dop(self.Xf)) + (self.mu + self.rho)*self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def iteration_stats(self, k, r, s, epri, edua, tk):
        """
        Construct iteration stats record tuple. Data fidelity term is
        :math:`(1/2) \|  \sum_m \mathbf{d}_m * \mathbf{x}_m -
        \mathbf{s} \|_2^2` and regularisation terms are
        :math:`\sum_m \| \mathbf{x}_m \|_1` and
        :math:`(1/2) \sum_m \| \mathbf{x}_m \|_2^2`
        """

        Ef = np.sum(self.Df * self.obfn_fvarf(), axis=self.axisM,
                    keepdims=True) - self.Sf
        dfd = sl.rfl2norm2(Ef, self.S.shape, axis=tuple(range(self.dimN)))/2.0
        rl1 = linalg.norm((self.opt['L1Weight'] * self.obfn_gvar()).ravel(), 1)
        rl2 = 0.5*linalg.norm(self.obfn_gvar())**2
        obj = dfd + self.lmbda*rl1 + self.mu*rl2
        itst = type(self).IterationStats(k, obj, dfd, rl1, rl2, r, s,
                                         epri, edua, self.rho, self.xrrs, tk)
        return itst
