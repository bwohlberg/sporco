#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithms for Total Variation (TV) optimisation with
an :math:`\ell^2` data fidelity term"""

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


class TVL2Denoise(admm.ADMM):
    """ADMM algorithm for :math:`\ell^2`-TV denoising problem
    :cite:`rudin-1992-nonlinear`, :cite:`goldstein-2009-split`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| W_{\mathrm{df}}(\mathbf{x} - \mathbf{s}) \|_2^2 +
             \lambda \\left\| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 + 
             (G_c \mathbf{x})^2} \\right\|_1

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_r,\mathbf{y}_c} \;
       (1/2) \| W_{\mathrm{df}}(\mathbf{x} - \mathbf{s}) \|_2^2 +
             \lambda \\left\| W_{\mathrm{tv}} \sqrt{(\mathbf{y}_r)^2 + 
             (\mathbf{y}_c)^2} \\right\|_1 \;\\text{such that}\;
       \\left( \\begin{array}{c} G_r \\\\ G_c \\end{array} \\right) \mathbf{x}
       - \\left( \\begin{array}{c} \mathbf{y}_r \\\\ \mathbf{y}_c \\end{array}
       \\right) = \\left( \\begin{array}{c} \mathbf{0} \\\\
       \mathbf{0} \\end{array} \\right) \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term \
       :math:`(1/2) \| W_{\mathrm{df}} (\mathbf{x} - \mathbf{s}) \|_2^2`

       ``RegTV`` : Value of regularisation term \
       :math:`\| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 + 
       (G_c \mathbf{x})^2} \|_1`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance \
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance \
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``GSIter`` : Number of Gauss-Seidel iterations

       ``GSRelRes`` : Relative residual of Gauss-Seidel solution

       ``Time`` : Cumulative run time
    """



    class Options(admm.ADMM.Options):
        """TVL2Denoise algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMM.Options`, together with
        additional options:

        ``gEvalY`` : Flag indicating whether the :math:`g` component of the \
        objective function should be evaluated using variable Y \
        (``True``) or X (``False``) as its argument

        ``MaxGSIter`` : Maximum Gauss-Seidel iterations

        ``GSTol`` : Gauss-Seidel stopping tolerance

        ``DFidWeight`` : Data fidelity weight matrix

        ``TVWeight`` : TV term weight matrix
        """

        defaults = copy.deepcopy(admm.ADMM.Options.defaults)
        defaults.update({'gEvalY' : True, 'RelaxParam' : 1.8,
                         'DFidWeight' : 1.0, 'TVWeight' : 1.0,
                         'GSTol' : 0.0, 'MaxGSIter' : 2
                        })
        defaults['AutoRho'].update({'Enabled' : False, 'Period' : 1,
                                    'AutoScaling' : True, 'Scaling' : 1000.0,
                                    'RsdlRatio' : 1.2})


        def __init__(self, opt=None):
            """Initialise TVL2Denoise algorithm options object."""

            if opt is None:
                opt = {}
            admm.ADMM.Options.__init__(self, opt)


        def set_lambda(self, lmbda, override=False):
            """Set parameters depending on lambda value"""

            if override or self['rho'] is None:
                self['rho'] = 2.0*lmbda + 0.1
            if override or self['AutoRho','RsdlTarget'] is None:
                self['AutoRho','RsdlTarget'] = 1.0




    IterationStats = collections.namedtuple('IterationStats',
                ['Iter', 'ObjFun', 'DFid', 'RegTV', 'PrimalRsdl', 'DualRsdl',
                 'EpsPrimal', 'EpsDual', 'Rho', 'GSIter', 'GSRelRes', 'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    hdrtxt = ['Itn', 'Fnc', 'DFid', 'TV', 'r', 's', 'rho']
    """Display column header text"""
    hdrval = {'Itn' : 'Iter', 'Fnc' : 'ObjFun', 'DFid' : 'DFid',
              'TV' : 'RegTV', 'r' : 'PrimalRsdl', 's' : 'DualRsdl',
              'rho' : 'Rho'}
    """Dictionary mapping display column headers to IterationStats entries"""


    def __init__(self, S, lmbda, opt=None, axes=(0,1)):
        """
        Initialise a TVL2Denoise object with problem parameters.

        Parameters
        ----------
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        opt : TVL2Denoise.Options object
          Algorithm options
        axes : tuple or list
          Axes on which TV regularisation is to be applied
        """

        if opt is None:
            opt = TVL2Denoise.Options()
        Nx = S.size
        Nc = len(axes)*Nx
        super(TVL2Denoise, self).__init__(Nx, Nc, opt)

        self.axes = axes
        self.lmbda = lmbda

        # Set rho value (computed from lambda if not specified)
        self.opt.set_lambda(self.lmbda)
        self.rho = self.opt['rho']

        # Determine working data type
        self.opt.set_dtype(S.dtype)
        self.dtype = self.opt['DataType']

        # Initial values for Y
        if  self.opt['Y0'] is None:
            self.Y = np.zeros(S.shape + (len(axes),), self.dtype)
        else:
            self.Y = self.opt['Y0']
        self.Yprev = self.Y

        # Initial value for U
        if  self.opt['U0'] is None:
            if  self.opt['Y0'] is None:
                self.U = np.zeros(S.shape + (len(axes),), self.dtype)
            else:
                # If Y0 is given, but not U0, then choose the initial
                # U so that the relevant dual optimality criterion
                # (see (3.10) in boyd-2010-distributed) is satisfied.
                Yss = np.sqrt(np.sum(self.Y**2, axis=S.ndim, keepdims=True))
                self.U = (self.lmbda/self.rho)*sl.zquotient(self.Y,Yss)
        else:
            self.U = self.opt['U0']

        self.S = S
        self.Wdf = self.opt['DFidWeight']
        self.Wdf2 = self.Wdf**2
        self.lcw = self.LaplaceCentreWeight()
        self.Wtv = self.opt['TVWeight']
        if hasattr(self.Wtv, 'ndim') and self.Wtv.ndim == S.ndim:
            self.Wtvna = self.Wtv[...,np.newaxis]
        else:
            self.Wtvna = self.Wtv

        # Need to initialise X because of Gauss-Seidel in xstep
        self.X = S

        self.runtime += self.timer.elapsed()




    def xstep(self):
        """Minimise Augmented Lagrangian with respect to x."""

        ngsit = 0
        gsrrs = np.inf
        while gsrrs > self.opt['GSTol'] and ngsit < self.opt['MaxGSIter']:
            self.X = self.GaussSeidelStep(self.S, self.X,
                                          self.cnst_AT(self.Y-self.U),
                                          self.rho, self.lcw, self.Wdf2)
            gsrrs = sl.rrs(
                self.rho*self.cnst_AT(self.cnst_A(self.X)) +
                self.Wdf2*self.X, self.Wdf2*self.S +
                self.rho*self.cnst_AT(self.Y - self.U))
            ngsit += 1

        self.xs = (ngsit, gsrrs)



    def ystep(self):
        """Minimise Augmented Lagrangian with respect to y."""

        self.Y = sl.shrink2(self.AX + self.U, (self.lmbda/self.rho)*self.Wtvna)




    def obfn_gvar(self):
        """Variable to be evaluated in computing regularisation term,
        depending on 'gEvalY' option value."""

        if self.opt['gEvalY']:
            return self.Y
        else:
            return self.cnst_A(self.X)



    def iteration_stats(self, k, r, s, epri, edua, tk):
        """
        Construct iteration stats record tuple. Data fidelity term is
        :math:`(1/2) \| \mathbf{x} - \mathbf{s} \|_2^2` and
        regularisation term is :math:`\| W_{\mathrm{tv}}
        \sqrt{(G_r \mathbf{x})^2 + (G_c \mathbf{x})^2}\|_1`.
        """

        dfd = 0.5*(linalg.norm(self.Wdf * (self.X - self.S))**2)
        reg = np.sum(self.Wtv * np.sqrt(np.sum(self.obfn_gvar()**2,
                                               axis=self.Y.ndim-1)))
        obj = dfd + self.lmbda*reg
        itst = type(self).IterationStats(k, obj, dfd, reg, r, s, epri,
                                edua, self.rho, self.xs[0], self.xs[1], tk)
        return itst



    def cnst_A(self, X):
        """Compute :math:`A \mathbf{x}` component of ADMM problem constraint.
        In this case :math:`A \mathbf{x} = (G_r^T \;\; G_c^T)^T \mathbf{x}`.
        """

        return np.concatenate(
            [sl.Gax(X, ax)[...,np.newaxis] for ax in self.axes],
            axis=X.ndim)



    def cnst_AT(self, X):
        """Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{x} = (G_r^T \;\; G_c^T) \mathbf{x}`."""

        return np.sum(np.concatenate(
            [sl.GTax(X[...,ax], ax)[...,np.newaxis] for ax in self.axes],
            axis=X.ndim-1), axis=X.ndim-1)



    def cnst_B(self, Y):
        """Compute :math:`B \mathbf{y}` component of ADMM problem constraint.
        In this case :math:`B \mathbf{y} = -\mathbf{y}`."""

        return -Y



    def cnst_c(self):
        """Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = \mathbf{0}`."""

        return np.zeros(self.S.shape + (len(self.axes),), self.dtype)



    def LaplaceCentreWeight(self):
        """Centre weighting matrix for TV Laplacian"""

        sz = [1,] * self.S.ndim
        for ax in self.axes:
            sz[ax] = self.S.shape[ax]
        lcw = 2.0*len(self.axes)*np.ones(sz, dtype=np.float32)
        for ax in self.axes:
            lcw[(slice(None),)*ax + ((0,-1),)] -= 1.0
        return lcw



    def GaussSeidelStep(self, S, X, ATYU, rho, lcw, W2):
        """Gauss-Seidel step for linear system in TV problem"""

        Xss = np.zeros_like(S)
        for ax in self.axes:
            Xss += sl.zpad(X[(slice(None),)*ax + (slice(0,-1),)], (1,0), ax)
            Xss += sl.zpad(X[(slice(None),)*ax + (slice(1,None),)], (0,1), ax)
        return (rho*(Xss + ATYU) + W2*S) / (W2 + rho*lcw)





class TVL2Deconv(admm.ADMM):
    """ADMM algorithm for :math:`\ell^2`-TV deconvolution problem.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| \mathbf{h} * \mathbf{x} - \mathbf{s} \|_2^2 +
             \lambda \\left\| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
             (G_c \mathbf{x})^2} \\right\|_1

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_r,\mathbf{y}_c} \;
       (1/2) \| \mathbf{h} * \mathbf{x} - \mathbf{s} \|_2^2 +
             \lambda \\left\| W_{\mathrm{tv}} \sqrt{(\mathbf{y}_r)^2 + 
             (\mathbf{y}_c)^2} \\right\|_1 \;\\text{such that}\;
       \\left( \\begin{array}{c} G_r \\\\ G_c \\end{array} \\right) \mathbf{x}
       - \\left( \\begin{array}{c} \mathbf{y}_r \\\\ \mathbf{y}_c \\end{array}
       \\right) = \\left( \\begin{array}{c} \mathbf{0} \\\\
       \mathbf{0} \\end{array} \\right) \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term \
       :math:`(1/2) \| \mathbf{h} * \mathbf{x} - \mathbf{s} \|_2^2`

       ``RegTV`` : Value of regularisation term \
       :math:`\| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
       (G_c \mathbf{x})^2} \|_1`

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


    class Options(admm.ADMM.Options):
        """TVL2Deconv algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMM.Options`, together with
        additional options:

        ``gEvalY`` : Flag indicating whether the :math:`g` component of the \
        objective function should be evaluated using variable Y \
        (``True``) or X (``False``) as its argument

        ``LinSolveCheck`` : If ``True``, compute relative residual of
        X step solver

        ``TVWeight`` : TV term weight matrix
        """

        defaults = copy.deepcopy(admm.ADMM.Options.defaults)
        defaults.update({'gEvalY' : True, 'RelaxParam' : 1.8,
                         'LinSolveCheck' : False, 'TVWeight' : 1.0})
        defaults['AutoRho'].update({'Enabled' : True, 'Period' : 1,
                                    'AutoScaling' : True, 'Scaling' : 1000.0,
                                    'RsdlRatio' : 1.2})


        def __init__(self, opt=None):
            """Initialise TVL2Deconv algorithm options object."""

            if opt is None:
                opt = {}
            admm.ADMM.Options.__init__(self, opt)



        def set_lambda(self, lmbda, override=False):
            """Set parameters depending on lambda value"""

            if override or self['rho'] is None:
                self['rho'] = 2.0*lmbda + 0.1
            if override or self['AutoRho','RsdlTarget'] is None:
                self['AutoRho','RsdlTarget'] = 1.0




    IterationStats = collections.namedtuple('IterationStats',
                ['Iter', 'ObjFun', 'DFid', 'RegTV', 'PrimalRsdl', 'DualRsdl',
                 'EpsPrimal', 'EpsDual', 'Rho', 'XSlvRelRes', 'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    hdrtxt = ['Itn', 'Fnc', 'DFid', 'TV', 'r', 's', 'rho']
    """Display column header text"""
    hdrval = {'Itn' : 'Iter', 'Fnc' : 'ObjFun', 'DFid' : 'DFid',
              'TV' : 'RegTV', 'r' : 'PrimalRsdl', 's' : 'DualRsdl',
              'rho' : 'Rho'}
    """Dictionary mapping display column headers to IterationStats entries"""


    def __init__(self, A, S, lmbda, opt=None, axes=(0,1)):
        """
        Initialise a TVL2Deconv object with problem parameters.

        Parameters
        ----------
        A : array_like
          Filter kernel (see :math:`\mathbf{h}` above)
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        opt : TVL2Deconv.Options object
          Algorithm options
        axes : tuple or list
          Axes on which TV regularisation is to be applied
        """

        if opt is None:
            opt = TVL2Deconv.Options()
        Nx = S.size
        Nc = len(axes)*Nx
        super(TVL2Deconv, self).__init__(Nx, Nc, opt)

        self.axes = axes
        self.lmbda = lmbda

        # Set rho value (computed from lambda if not specified)
        self.opt.set_lambda(self.lmbda)
        self.rho = self.opt['rho']

        # Determine working data type
        self.opt.set_dtype(S.dtype)
        self.dtype = self.opt['DataType']

        # Initial values for Y
        if  self.opt['Y0'] is None:
            self.Y = np.zeros(S.shape + (len(axes),), self.dtype)
        else:
            self.Y = self.opt['Y0']
        self.Yprev = self.Y

        # Initial value for U
        if  self.opt['U0'] is None:
            if  self.opt['Y0'] is None:
                self.U = np.zeros(S.shape + (len(axes),), self.dtype)
            else:
                # If Y0 is given, but not U0, then choose the initial
                # U so that the relevant dual optimality criterion
                # (see (3.10) in boyd-2010-distributed) is satisfied.
                Yss = np.sqrt(np.sum(self.Y**2, axis=S.ndim, keepdims=True))
                self.U = (self.lmbda/self.rho)*sl.zquotient(self.Y,Yss)
        else:
            self.U = self.opt['U0']

        self.axshp = [S.shape[k] for k in axes]
        self.A = sl.atleast_nd(S.ndim, A.astype(self.dtype))
        self.S = S
        self.Af = sl.rfftn(self.A, self.axshp, axes=axes)
        self.Sf = sl.rfftn(S, axes=axes)
        self.AHAf = np.conj(self.Af)*self.Af
        self.AHSf = np.conj(self.Af)*self.Sf

        self.Wtv = self.opt['TVWeight']
        if hasattr(self.Wtv, 'ndim') and self.Wtv.ndim == S.ndim:
            self.Wtvna = self.Wtv[...,np.newaxis]
        else:
            self.Wtvna = self.Wtv

        g = np.zeros([2 if k in axes else 1 for k in range(S.ndim)] + 
                     [len(axes),], self.dtype)
        for k in axes:
            g[(0,)*k +(slice(None),)+(0,)*(g.ndim-2-k)+(k,)] = [1,-1]
        self.Gf = sl.rfftn(g, self.axshp, axes=axes)
        self.GHGf = np.sum(np.conj(self.Gf)*self.Gf, axis=self.Y.ndim-1)

        self.runtime += self.timer.elapsed()



    def xstep(self):
        """Minimise Augmented Lagrangian with respect to x."""

        b = self.AHSf + self.rho*np.sum(np.conj(self.Gf)*
                    sl.rfftn(self.Y-self.U, axes=self.axes), axis=self.Y.ndim-1)
        self.Xf = b / (self.AHAf + self.rho*self.GHGf)
        self.X = sl.irfftn(self.Xf, None, axes=self.axes)

        if self.opt['LinSolveCheck']:
            ax = (self.AHAf + self.rho*self.GHGf)*self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        """Minimise Augmented Lagrangian with respect to y."""

        self.Y = sl.shrink2(self.AX + self.U, (self.lmbda/self.rho)*self.Wtvna)



    def obfn_gvar(self):
        """Variable to be evaluated in computing regularisation term,
        depending on 'gEvalY' option value."""

        if self.opt['gEvalY']:
            return self.Y
        else:
            return self.cnst_A(None, self.Xf)



    def iteration_stats(self, k, r, s, epri, edua, tk):
        """
        Construct iteration stats record tuple. Data fidelity term is
        :math:`(1/2) \| \mathbf{h} * \mathbf{x} - \mathbf{s} \|_2^2` and
        regularisation term is :math:`\| W_{\mathrm{tv}}
        \sqrt{(G_r \mathbf{x})^2 + (G_c \mathbf{x})^2}\|_1`.
        """

        Ef = self.Af * self.Xf - self.Sf
        dfd = sl.rfl2norm2(Ef, self.S.shape, axis=self.axes) / 2.0
        reg = np.sum(self.Wtv * np.sqrt(np.sum(self.obfn_gvar()**2,
                     axis=self.Y.ndim-1)))
        obj = dfd + self.lmbda*reg
        itst = type(self).IterationStats(k, obj, dfd, reg, r, s, epri,
                                         edua, self.rho, self.xrrs, tk)
        return itst



    def cnst_A(self, X, Xf=None):
        """Compute :math:`A \mathbf{x}` component of ADMM problem constraint.
        In this case :math:`A \mathbf{x} = (G_r^T \;\; G_c^T)^T \mathbf{x}`.
        """

        if Xf is None:
            Xf = sl.rfftn(X, axes=self.axes)
        return sl.irfftn(self.Gf*Xf[...,np.newaxis], None, axes=self.axes)



    def cnst_AT(self, X):
        """Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{x} = (G_r^T \;\; G_c^T) \mathbf{x}`."""

        Xf = sl.rfftn(X, axes=self.axes)
        return np.sum(sl.irfftn(np.conj(self.Gf)*Xf, None, axes=self.axes),
                      axis=self.Y.ndim-1)



    def cnst_B(self, Y):
        """Compute :math:`B \mathbf{y}` component of ADMM problem constraint.
        In this case :math:`B \mathbf{y} = -\mathbf{y}`."""

        return -Y



    def cnst_c(self):
        """Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = \mathbf{0}`."""

        return np.zeros(self.S.shape + (len(self.axes),), self.dtype)
