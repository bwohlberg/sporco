# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Classes for ADMM algorithms for Total Variation (TV) optimisation
with an :math:`\ell_2` data fidelity term"""

from __future__ import division
from __future__ import absolute_import
from builtins import range

import copy
import numpy as np
from scipy import linalg

from sporco.admm import admm
import sporco.linalg as sl

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class TVL2Denoise(admm.ADMM):
    r"""ADMM algorithm for :math:`\ell_2`-TV denoising problem
    :cite:`rudin-1992-nonlinear`, :cite:`goldstein-2009-split`,
    :cite:`blomgren-1998-color`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| W_{\mathrm{df}}(\mathbf{x} - \mathbf{s}) \|_2^2 +
             \lambda \left\| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
             (G_c \mathbf{x})^2} \right\|_1

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_r,\mathbf{y}_c} \;
       (1/2) \| W_{\mathrm{df}}(\mathbf{x} - \mathbf{s}) \|_2^2 +
             \lambda \left\| W_{\mathrm{tv}} \sqrt{(\mathbf{y}_r)^2 +
             (\mathbf{y}_c)^2} \right\|_1 \;\text{such that}\;
       \left( \begin{array}{c} G_r \\ G_c \end{array} \right) \mathbf{x}
       - \left( \begin{array}{c} \mathbf{y}_r \\ \mathbf{y}_c \end{array}
       \right) = \left( \begin{array}{c} \mathbf{0} \\
       \mathbf{0} \end{array} \right) \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \|
       W_{\mathrm{df}} (\mathbf{x} - \mathbf{s}) \|_2^2`

       ``RegTV`` : Value of regularisation term :math:`\|
       W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 + (G_c \mathbf{x})^2}
       \|_1`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
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

          ``gEvalY`` : Flag indicating whether the :math:`g` component
          of the objective function should be evaluated using variable
          Y (``True``) or X (``False``) as its argument.

          ``MaxGSIter`` : Maximum Gauss-Seidel iterations.

          ``GSTol`` : Gauss-Seidel stopping tolerance.

          ``DFidWeight`` : Data fidelity weight matrix.

          ``TVWeight`` : TV term weight matrix.
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

            if self['AutoRho', 'RsdlTarget'] is None:
                self['AutoRho', 'RsdlTarget'] = 1.0



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegTV')
    itstat_fields_extra = ('GSIter', 'GSRelRes')
    hdrtxt_objfn = ('Fnc', 'DFid', 'RegTV')
    hdrval_objfun = {'Fnc' : 'ObjFun', 'DFid' : 'DFid', 'RegTV' : 'RegTV'}



    def __init__(self, S, lmbda, opt=None, axes=(0, 1), caxis=None):
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
        caxis : int or None, optional (default None)
          Axis on which channels of a multi-channel image are stacked.
          If None, TV regularisation is applied indepdendently to each
          channel, otherwise Vector TV :cite:`blomgren-1998-color`
          regularisation is applied jointly to all channels.
        """

        if opt is None:
            opt = TVL2Denoise.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        self.S = np.asarray(S, dtype=self.dtype)
        self.axes = axes
        if caxis is None:
            self.saxes = (-1,)
        else:
            self.saxes = (caxis, -1)
        self.lmbda = self.dtype.type(lmbda)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(2.0*self.lmbda + 0.1),
                      dtype=self.dtype)

        yshape = S.shape + (len(axes),)
        super(TVL2Denoise, self).__init__(S.size, yshape, yshape, S.dtype, opt)

        self.Wdf = np.asarray(self.opt['DFidWeight'], dtype=self.dtype)
        self.Wdf2 = self.Wdf**2
        self.lcw = self.LaplaceCentreWeight()
        self.Wtv = np.asarray(self.opt['TVWeight'], dtype=self.dtype)
        if hasattr(self.Wtv, 'ndim') and self.Wtv.ndim == S.ndim:
            self.Wtvna = self.Wtv[..., np.newaxis]
        else:
            self.Wtvna = self.Wtv

        # Need to initialise X because of Gauss-Seidel in xstep
        self.X = self.S



    def uinit(self, ushape):
        """Return initialiser for working variable U."""

        if  self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            Yss = np.sqrt(np.sum(self.Y**2, axis=self.S.ndim, keepdims=True))
            return (self.lmbda/self.rho)*sl.zdivide(self.Y, Yss)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

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
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y = np.asarray(sl.shrink2(self.AX + self.U,
                    (self.lmbda/self.rho)*self.Wtvna, axis=self.saxes),
                    dtype=self.dtype)



    def obfn_gvar(self):
        """Variable to be evaluated in computing regularisation term,
        depending on 'gEvalY' option value.
        """

        if self.opt['gEvalY']:
            return self.Y
        else:
            return self.cnst_A(self.X)



    def eval_objfn(self):
        r"""Compute components of objective function as well as total
        contribution to objective function. Data fidelity term is
        :math:`(1/2) \| \mathbf{x} - \mathbf{s} \|_2^2` and
        regularisation term is :math:`\| W_{\mathrm{tv}}
        \sqrt{(G_r \mathbf{x})^2 + (G_c \mathbf{x})^2}\|_1`.
        """

        dfd = 0.5*(linalg.norm(self.Wdf * (self.X - self.S))**2)
        reg = np.sum(self.Wtv * np.sqrt(np.sum(self.obfn_gvar()**2,
                                               axis=self.saxes)))
        obj = dfd + self.lmbda*reg
        return (obj, dfd, reg)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xs[0], self.xs[1])



    def cnst_A(self, X):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint.  In this case :math:`A \mathbf{x} = (G_r^T \;\;
        G_c^T)^T \mathbf{x}`.
        """

        return np.concatenate(
            [sl.Gax(X, ax)[..., np.newaxis] for ax in self.axes],
            axis=X.ndim)



    def cnst_AT(self, X):
        r"""Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{x} = (G_r^T \;\; G_c^T) \mathbf{x}`.
        """

        return np.sum(np.concatenate(
            [sl.GTax(X[..., ax], ax)[..., np.newaxis] for ax in self.axes],
            axis=X.ndim-1), axis=X.ndim-1)



    def cnst_B(self, Y):
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem
        constraint.  In this case :math:`B \mathbf{y} = -\mathbf{y}`.
        """

        return -Y



    def cnst_c(self):
        r"""Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = \mathbf{0}`.
        """

        return np.zeros(self.S.shape + (len(self.axes),), self.dtype)



    def LaplaceCentreWeight(self):
        """Centre weighting matrix for TV Laplacian."""

        sz = [1,] * self.S.ndim
        for ax in self.axes:
            sz[ax] = self.S.shape[ax]
        lcw = 2*len(self.axes)*np.ones(sz, dtype=self.dtype)
        for ax in self.axes:
            lcw[(slice(None),)*ax + ((0, -1),)] -= 1.0
        return lcw



    def GaussSeidelStep(self, S, X, ATYU, rho, lcw, W2):
        """Gauss-Seidel step for linear system in TV problem."""

        Xss = np.zeros_like(S, dtype=self.dtype)
        for ax in self.axes:
            Xss += sl.zpad(X[(slice(None),)*ax + (slice(0, -1),)],
                           (1, 0), ax)
            Xss += sl.zpad(X[(slice(None),)*ax + (slice(1, None),)],
                           (0, 1), ax)
        return (rho*(Xss + ATYU) + W2*S) / (W2 + rho*lcw)





class TVL2Deconv(admm.ADMM):
    r"""ADMM algorithm for :math:`\ell_2`-TV deconvolution problem.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| H \mathbf{x} - \mathbf{s} \|_2^2 +
             \lambda \left\| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
             (G_c \mathbf{x})^2} \right\|_1 \;\;,

    where :math:`H` denotes the linear operator corresponding to a
    convolution, via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_r,\mathbf{y}_c} \;
       (1/2) \| H \mathbf{x} - \mathbf{s} \|_2^2 +
             \lambda \left\| W_{\mathrm{tv}} \sqrt{(\mathbf{y}_r)^2 +
             (\mathbf{y}_c)^2} \right\|_1 \;\text{such that}\;
       \left( \begin{array}{c} G_r \\ G_c \end{array} \right) \mathbf{x}
       - \left( \begin{array}{c} \mathbf{y}_r \\ \mathbf{y}_c \end{array}
       \right) = \left( \begin{array}{c} \mathbf{0} \\
       \mathbf{0} \end{array} \right) \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| H
       \mathbf{x} - \mathbf{s} \|_2^2`

       ``RegTV`` : Value of regularisation term :math:`\|
       W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 + (G_c \mathbf{x})^2}
       \|_1`

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


    class Options(admm.ADMM.Options):
        """TVL2Deconv algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMM.Options`, together with
        additional options:

          ``gEvalY`` : Flag indicating whether the :math:`g` component
          of the objective function should be evaluated using variable
          Y (``True``) or X (``False``) as its argument.

          ``LinSolveCheck`` : If ``True``, compute relative residual
          of X step solver.

          ``TVWeight`` : TV term weight matrix.
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

            if self['AutoRho', 'RsdlTarget'] is None:
                self['AutoRho', 'RsdlTarget'] = 1.0



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegTV')
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('Fnc', 'DFid', 'RegTV')
    hdrval_objfun = {'Fnc' : 'ObjFun', 'DFid' : 'DFid', 'RegTV' : 'RegTV'}



    def __init__(self, A, S, lmbda, opt=None, axes=(0, 1), caxis=None):
        """
        Initialise a TVL2Deconv object with problem parameters.

        Parameters
        ----------
        A : array_like
          Filter kernel corresponding to operator :math:`H` above
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        opt : TVL2Deconv.Options object
          Algorithm options
        axes : tuple or list
          Axes on which TV regularisation is to be applied
        caxis : int or None, optional (default None)
          Axis on which channels of a multi-channel image are stacked.
          If None, TV regularisation is applied indepdendently to each
          channel, otherwise Vector TV :cite:`blomgren-1998-color`
          regularisation is applied jointly to all channels.
        """

        if opt is None:
            opt = TVL2Deconv.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        self.S = np.asarray(S, dtype=self.dtype)
        self.axes = axes
        if caxis is None:
            self.saxes = (-1,)
        else:
            self.saxes = (caxis, -1)
        self.lmbda = self.dtype.type(lmbda)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(2.0*self.lmbda + 0.1),
                      dtype=self.dtype)

        yshape = S.shape + (len(axes),)
        super(TVL2Deconv, self).__init__(S.size, yshape, yshape, S.dtype, opt)

        self.axshp = [S.shape[k] for k in axes]
        self.A = sl.atleast_nd(S.ndim, A.astype(self.dtype))
        self.Af = sl.rfftn(self.A, self.axshp, axes=axes)
        self.Sf = sl.rfftn(self.S, axes=axes)
        self.AHAf = np.conj(self.Af)*self.Af
        self.AHSf = np.conj(self.Af)*self.Sf

        self.Wtv = np.asarray(self.opt['TVWeight'], dtype=self.dtype)
        if hasattr(self.Wtv, 'ndim') and self.Wtv.ndim == S.ndim:
            self.Wtvna = self.Wtv[..., np.newaxis]
        else:
            self.Wtvna = self.Wtv

        # Construct gradient operators in frequency domain
        self.Gf, self.GHGf = sl.GradientFilters(S.ndim, axes, self.axshp,
                                                dtype=self.dtype)



    def uinit(self, ushape):
        """Return initialiser for working variable U."""

        if  self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            Yss = np.sqrt(np.sum(self.Y**2, axis=self.S.ndim, keepdims=True))
            return (self.lmbda/self.rho)*sl.zdivide(self.Y, Yss)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        b = self.AHSf + self.rho*np.sum(np.conj(self.Gf)*
                    sl.rfftn(self.Y-self.U, axes=self.axes),
                             axis=self.Y.ndim-1)
        self.Xf = b / (self.AHAf + self.rho*self.GHGf)
        self.X = sl.irfftn(self.Xf, None, axes=self.axes)

        if self.opt['LinSolveCheck']:
            ax = (self.AHAf + self.rho*self.GHGf)*self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y = sl.shrink2(self.AX + self.U, (self.lmbda/self.rho)*self.Wtvna,
                            axis=self.saxes)



    def obfn_gvar(self):
        """Variable to be evaluated in computing regularisation term,
        depending on 'gEvalY' option value.
        """

        if self.opt['gEvalY']:
            return self.Y
        else:
            return self.cnst_A(None, self.Xf)



    def eval_objfn(self):
        r"""Compute components of objective function as well as total
        contribution to objective function. Data fidelity term is
        :math:`(1/2) \| H \mathbf{x} - \mathbf{s} \|_2^2` and
        regularisation term is :math:`\| W_{\mathrm{tv}}
        \sqrt{(G_r \mathbf{x})^2 + (G_c \mathbf{x})^2}\|_1`.
        """

        Ef = self.Af * self.Xf - self.Sf
        dfd = sl.rfl2norm2(Ef, self.S.shape, axis=self.axes) / 2.0
        reg = np.sum(self.Wtv * np.sqrt(np.sum(self.obfn_gvar()**2,
                     axis=self.saxes)))
        obj = dfd + self.lmbda*reg
        return (obj, dfd, reg)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs,)



    def cnst_A(self, X, Xf=None):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint.  In this case :math:`A \mathbf{x} = (G_r^T \;\;
        G_c^T)^T \mathbf{x}`.
        """

        if Xf is None:
            Xf = sl.rfftn(X, axes=self.axes)
        return sl.irfftn(self.Gf*Xf[..., np.newaxis], None, axes=self.axes)



    def cnst_AT(self, X):
        r"""Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{x} = (G_r^T \;\; G_c^T) \mathbf{x}`.
        """

        Xf = sl.rfftn(X, axes=self.axes)
        return np.sum(sl.irfftn(np.conj(self.Gf)*Xf, None, axes=self.axes),
                      axis=self.Y.ndim-1)



    def cnst_B(self, Y):
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem
        constraint.  In this case :math:`B \mathbf{y} = -\mathbf{y}`.
        """

        return -Y



    def cnst_c(self):
        r"""Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = \mathbf{0}`.
        """

        return np.zeros(self.S.shape + (len(self.axes),), self.dtype)
