# -*- coding: utf-8 -*-
# Copyright (C) 2015-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Classes for ADMM algorithms for Total Variation (TV) optimisation
with an :math:`\ell_1` data fidelity term"""

from __future__ import division, absolute_import

import copy
import numpy as np

from sporco.admm import admm
from sporco.array import zpad, atleast_nd, zdivide
from sporco.fft import real_dtype, fftn_func, ifftn_func
from sporco.signal import gradient_filters, grad, gradT
from sporco.linalg import rrs
from sporco.prox import prox_l1, prox_l2


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class TVL1Denoise(admm.ADMM):
    r"""ADMM algorithm for :math:`\ell_1`-TV denoising problem
    :cite:`alliney-1992-digital` :cite:`esser-2010-primal` (Sec. 2.4.4).

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
        \| W_{\mathrm{df}}  (\mathbf{x} - \mathbf{s}) \|_1 +
             \lambda \left\| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
             (G_c \mathbf{x})^2} \right\|_1

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_d,\mathbf{y}_r,\mathbf{y}_c} \;
       (1/2) \| W_{\mathrm{df}} \mathbf{y}_d \|_1 +
             \lambda \left\| W_{\mathrm{tv}} \sqrt{(\mathbf{y}_r)^2 +
             (\mathbf{y}_c)^2} \right\|_1 \;\text{such that}\;
       \left( \begin{array}{c} G_r \\ G_c \\ I \end{array} \right)
       \mathbf{x}  - \left( \begin{array}{c} \mathbf{y}_r \\
       \mathbf{y}_c \\ \mathbf{y}_d \end{array}
       \right) = \left( \begin{array}{c} \mathbf{0} \\ \mathbf{0} \\
       \mathbf{s} \end{array} \right) \;\;,

    where :math:`G_r` and :math:`G_c` are gradient operators along array
    rows and columns respectively, and :math:`W_{\mathrm{df}}` and
    :math:`W_{\mathrm{tv}}` are diagonal weighting matrices.

    While these equations describe the default behaviour of regularisation
    in two dimensions, this class supports an arbitrary number of
    dimensions. For example, for 3D TV regularisation in a 3D array,
    the object should be initialised with parameter `axes` set to
    `(0, 1, 2)`.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`\|
       W_{\mathrm{df}} (\mathbf{x} - \mathbf{s}) \|_1`

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
        """TVL1Denoise algorithm options

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
        defaults.update({'gEvalY': True, 'RelaxParam': 1.8,
                         'DFidWeight': 1.0, 'TVWeight': 1.0,
                         'GSTol': 0.0, 'MaxGSIter': 2
                        })
        defaults['AutoRho'].update({'Enabled': False, 'Period': 1,
                                    'AutoScaling': True, 'Scaling': 1000.0,
                                    'RsdlRatio': 1.2})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              TVL1Denoise algorithm options
            """

            if opt is None:
                opt = {}
            admm.ADMM.Options.__init__(self, opt)

            if self['AutoRho', 'RsdlTarget'] is None:
                self['AutoRho', 'RsdlTarget'] = 1.0



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegTV')
    itstat_fields_extra = ('GSIter', 'GSRelRes')
    hdrtxt_objfn = ('Fnc', 'DFid', 'RegTV')
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', 'RegTV': 'RegTV'}



    def __init__(self, S, lmbda, opt=None, axes=(0, 1), caxis=None):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/tvl1den_init.svg
           :width: 20%
           :target: ../_static/jonga/tvl1den_init.svg

        |


        Parameters
        ----------
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        opt : TVL1Denoise.Options object
          Algorithm options
        axes : tuple, optional (default (0, 1))
          Axes on which TV regularisation is to be applied
        caxis : int or None, optional (default None)
          Axis on which channels of a multi-channel image are stacked.
          If None, TV regularisation is applied indepdendently to each
          channel, otherwise Vector TV :cite:`blomgren-1998-color`
          regularisation is applied jointly to all channels.
        """

        if opt is None:
            opt = TVL1Denoise.Options()

        # Set flag indicating whether problem involves real or complex
        # values
        self.real_dtype = np.isrealobj(S)

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        self.S = np.asarray(S, dtype=self.dtype)
        self.axes = axes
        if caxis is None:
            self.saxes = (-1,)
        else:
            self.saxes = (caxis, -1)
        self.lmbda = real_dtype(self.dtype).type(lmbda)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(2.0*self.lmbda + 0.1),
                      dtype=real_dtype(self.dtype))

        yshape = S.shape + (len(axes)+1,)
        super(TVL1Denoise, self).__init__(S.size, yshape, yshape, S.dtype, opt)

        self.Wdf = np.asarray(self.opt['DFidWeight'],
                              dtype=real_dtype(self.dtype))
        self.lcw = self.LaplaceCentreWeight()
        self.Wtv = np.asarray(self.opt['TVWeight'],
                              dtype=real_dtype(self.dtype))
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
            Yss = np.sqrt(np.sum(self.Y[..., 0:-1]**2, axis=self.S.ndim,
                                 keepdims=True))
            U0 = (self.lmbda/self.rho)*zdivide(self.Y[..., 0:-1], Yss)
            U1 = (1.0 / self.rho)*np.sign(self.Y[..., -1:])
            return np.concatenate((U0, U1), axis=self.S.ndim)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        ngsit = 0
        gsrrs = np.inf
        YU = self.Y - self.U
        SYU = self.S + YU[..., -1]
        YU[..., -1] = 0.0
        ATYU = self.cnst_AT(YU)
        while gsrrs > self.opt['GSTol'] and ngsit < self.opt['MaxGSIter']:
            self.X = self.GaussSeidelStep(
                SYU, self.X, ATYU, 1.0, self.lcw, 1.0)
            gsrrs = rrs(
                self.cnst_AT(self.cnst_A(self.X)),
                self.cnst_AT(self.cnst_c() - self.cnst_B(self.Y) - self.U)
            )
            ngsit += 1

        self.xs = (ngsit, gsrrs)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y[..., 0:-1] = prox_l2(
            self.AX[..., 0:-1] + self.U[..., 0:-1],
            (self.lmbda/self.rho)*self.Wtvna, axis=self.saxes)
        self.Y[..., -1] = prox_l1(
            self.AX[..., -1] + self.U[..., -1] - self.S,
            (1.0/self.rho)*self.Wdf)



    def obfn_gvar(self):
        """Variable to be evaluated in computing regularisation term,
        depending on 'gEvalY' option value.
        """

        if self.opt['gEvalY']:
            return self.Y
        else:
            return self.cnst_A(self.X) - self.cnst_c()



    def eval_objfn(self):
        r"""Compute components of objective function as well as total
        contribution to objective function. Data fidelity term is
        :math:`(1/2) \| \mathbf{x} - \mathbf{s} \|_2^2` and
        regularisation term is :math:`\| W_{\mathrm{tv}}
        \sqrt{(G_r \mathbf{x})^2 + (G_c \mathbf{x})^2}\|_1`.
        """

        if self.real_dtype:
            gvr = self.obfn_gvar()
        else:
            gvr = np.abs(self.obfn_gvar())
        dfd = np.sum(np.abs(self.Wdf * gvr[..., -1]))
        reg = np.sum(self.Wtv * np.sqrt(np.sum(gvr[..., 0:-1]**2,
                                               axis=self.saxes)))
        obj = dfd + self.lmbda*reg
        return (obj, dfd, reg)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xs[0], self.xs[1])



    def cnst_A(self, X):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint. In this case :math:`A \mathbf{x} = (G_r^T \;\; G_c^T
        \;\; I)^T \mathbf{x}`.
        """

        return np.concatenate(
            [grad(X, ax)[..., np.newaxis] for ax in self.axes] +
            [X[..., np.newaxis],], axis=X.ndim)



    def cnst_AT(self, X):
        r"""Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{x} = (G_r^T \;\; G_c^T \;\; I) \mathbf{x}`.
        """

        return np.sum(np.concatenate(
            [gradT(X[..., ax], ax)[..., np.newaxis] for ax in self.axes] +
            [X[..., -1:],], axis=X.ndim-1), axis=X.ndim-1)



    def cnst_B(self, Y):
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem
        constraint.  In this case :math:`B \mathbf{y} = -\mathbf{y}`.
        """

        return -Y



    def cnst_c(self):
        r"""Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = (\mathbf{0} \;\;
        \mathbf{0} \;\; \mathbf{s})`.
        """

        c = np.zeros(self.S.shape + (len(self.axes)+1,), self.dtype)
        c[..., -1] = self.S
        return c



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        return self.rho*np.linalg.norm(self.cnst_AT(self.U))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho*np.linalg.norm(U)



    def LaplaceCentreWeight(self):
        """Centre weighting matrix for TV Laplacian."""

        sz = [1,] * self.S.ndim
        for ax in self.axes:
            sz[ax] = self.S.shape[ax]
        lcw = 2*len(self.axes)*np.ones(sz, dtype=self.dtype)
        for ax in self.axes:
            lcw[(slice(None),)*ax + ([0, -1],)] -= 1.0
        return lcw



    def GaussSeidelStep(self, S, X, ATYU, rho, lcw, W2):
        """Gauss-Seidel step for linear system in TV problem."""

        Xss = np.zeros_like(S, dtype=self.dtype)
        for ax in self.axes:
            Xss += zpad(X[(slice(None),)*ax + (slice(0, -1),)], (1, 0), ax)
            Xss += zpad(X[(slice(None),)*ax + (slice(1, None),)],
                        (0, 1), ax)
        return (rho*(Xss + ATYU) + W2*S) / (W2 + rho*lcw)





class TVL1Deconv(admm.ADMM):
    r"""ADMM algorithm for :math:`\ell_1`-TV deconvolution problem.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       \| W_{\mathrm{df}} (H \mathbf{x} - \mathbf{s}) \|_1 +
       \lambda \left\| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
       (G_c \mathbf{x})^2} \right\|_1 \;\;,

    where :math:`H` denotes the linear operator corresponding to a
    convolution, :math:`G_r` and :math:`G_c` are gradient operators
    along array rows and columns respectively, and
    :math:`W_{\mathrm{df}}` and :math:`W_{\mathrm{tv}}` are diagonal
    weighting matrices, via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_d,\mathbf{y}_r,\mathbf{y}_c} \;
       (1/2) \| W_{\mathrm{df}} \mathbf{y}_d \|_1 +
             \lambda \left\| W_{\mathrm{tv}} \sqrt{(\mathbf{y}_r)^2 +
             (\mathbf{y}_c)^2} \right\|_1 \;\text{such that}\;
       \left( \begin{array}{c} G_r \\ G_c \\ H \end{array} \right)
       \mathbf{x}  - \left( \begin{array}{c} \mathbf{y}_r \\
       \mathbf{y}_c \\ \mathbf{y}_d \end{array}
       \right) = \left( \begin{array}{c} \mathbf{0} \\ \mathbf{0} \\
       \mathbf{s} \end{array} \right) \;\;.

    While these equations describe the default behaviour of regularisation
    in two dimensions, this class supports an arbitrary number of
    dimensions. For example, for 3D TV regularisation in a 3D array,
    the object should be initialised with parameter `axes` set to
    `(0, 1, 2)`.

    Note that the convolution is implemented in the frequency domain,
    having the same phase offset as :func:`.fftconv`, which differs from
    that of :func:`scipy.ndimage.convolve` with the default ``origin``
    parameter.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`\|
       W_{\mathrm{df}} (H \mathbf{x} - \mathbf{s}) \|_1`

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
        """TVL1Deconv algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMM.Options`, together with
        additional options:

          ``gEvalY`` : Flag indicating whether the :math:`g` component
          of the objective function should be evaluated using variable
          Y (``True``) or X (``False``) as its argument.

          ``LinSolveCheck`` : If ``True``, compute relative residual of
          X step solver.

          ``DFidWeight`` : Data fidelity weight matrix.

          ``TVWeight`` : TV term weight matrix.
        """

        defaults = copy.deepcopy(admm.ADMM.Options.defaults)
        defaults.update(
            {'gEvalY': True, 'RelaxParam': 1.8, 'LinSolveCheck': False,
             'DFidWeight': 1.0, 'TVWeight': 1.0})
        defaults['AutoRho'].update(
            {'Enabled': False, 'Period': 1, 'AutoScaling': True,
             'Scaling': 1000.0, 'RsdlRatio': 1.2})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              TVL1Deconv algorithm options
            """

            if opt is None:
                opt = {}
            admm.ADMM.Options.__init__(self, opt)

            if self['AutoRho', 'RsdlTarget'] is None:
                self['AutoRho', 'RsdlTarget'] = 1.0



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegTV')
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('Fnc', 'DFid', 'RegTV')
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', 'RegTV': 'RegTV'}



    def __init__(self, A, S, lmbda, opt=None, axes=(0, 1), caxis=None):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/tvl1dcn_init.svg
           :width: 20%
           :target: ../_static/jonga/tvl1dcn_init.svg

        |


        Parameters
        ----------
        A : array_like
          Filter kernel corresponding to operator :math:`H` above
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        opt : TVL1Deconv.Options object
          Algorithm options
        axes : tuple, optional (default (0, 1))
          Axes on which TV regularisation is to be applied
        caxis : int or None, optional (default None)
          Axis on which channels of a multi-channel image are stacked.
          If None, TV regularisation is applied indepdendently to each
          channel, otherwise Vector TV :cite:`blomgren-1998-color`
          regularisation is applied jointly to all channels.
        """

        if opt is None:
            opt = TVL1Deconv.Options()

        # Set flag indicating whether problem involves real or complex
        # values, and get appropriate versions of functions from fft
        # module
        self.real_dtype = np.isrealobj(S)
        self.fftn = fftn_func(self.real_dtype)
        self.ifftn = ifftn_func(self.real_dtype)

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        self.axes = axes
        self.axsz = tuple([S.shape[i] for i in axes])
        if caxis is None:
            self.saxes = (-1,)
        else:
            self.saxes = (caxis, -1)
        self.lmbda = real_dtype(self.dtype).type(lmbda)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(2.0*self.lmbda + 0.1),
                      dtype=real_dtype(self.dtype))

        yshape = S.shape + (len(axes)+1,)
        self.S = np.asarray(S, dtype=self.dtype)
        super(TVL1Deconv, self).__init__(S.size, yshape, yshape, S.dtype, opt)

        self.axshp = tuple([S.shape[k] for k in axes])
        self.A = atleast_nd(S.ndim, A.astype(self.dtype))
        self.Af = self.fftn(self.A, self.axshp, axes=axes)
        self.Sf = self.fftn(self.S, axes=axes)
        self.AHAf = np.conj(self.Af)*self.Af
        self.AHSf = np.conj(self.Af)*self.Sf

        self.Wdf = np.asarray(self.opt['DFidWeight'],
                              dtype=real_dtype(self.dtype))
        self.Wtv = np.asarray(self.opt['TVWeight'],
                              dtype=real_dtype(self.dtype))
        if hasattr(self.Wtv, 'ndim') and self.Wtv.ndim == S.ndim:
            self.Wtvna = self.Wtv[..., np.newaxis]
        else:
            self.Wtvna = self.Wtv

        self.Gf, self.GHGf = gradient_filters(S.ndim, axes, self.axshp,
                                              dtype=self.dtype)
        self.GAf = np.concatenate((self.Gf, self.Af[..., np.newaxis]),
                                  axis=self.Gf.ndim-1)



    def uinit(self, ushape):
        """Return initialiser for working variable U."""

        if  self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            Yss = np.sqrt(np.sum(self.Y[..., 0:-1]**2, axis=self.S.ndim,
                                 keepdims=True))
            U0 = (self.lmbda/self.rho)*zdivide(self.Y[..., 0:-1], Yss)
            U1 = (1.0 / self.rho)*np.sign(self.Y[..., -1:])
            return np.concatenate((U0, U1), axis=self.S.ndim)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        b = self.AHSf + np.sum(
            np.conj(self.GAf) * self.fftn(self.Y-self.U, axes=self.axes),
            axis=self.Y.ndim-1)
        self.Xf = b / (self.AHAf + self.GHGf)
        self.X = self.ifftn(self.Xf, self.axsz, axes=self.axes)

        if self.opt['LinSolveCheck']:
            ax = (self.AHAf + self.GHGf)*self.Xf
            self.xrrs = rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y[..., 0:-1] = prox_l2(
            self.AX[..., 0:-1] + self.U[..., 0:-1],
            (self.lmbda/self.rho)*self.Wtvna, axis=self.saxes)
        self.Y[..., -1] = prox_l1(
            self.AX[..., -1] + self.U[..., -1] - self.S,
            (1.0/self.rho)*self.Wdf)



    def obfn_gvar(self):
        """Variable to be evaluated in computing regularisation term,
        depending on 'gEvalY' option value.
        """

        if self.opt['gEvalY']:
            return self.Y
        else:
            return self.cnst_A(None, self.Xf) - self.cnst_c()



    def eval_objfn(self):
        r"""Compute components of objective function as well as total
        contribution to objective function.  Data fidelity term is
        :math:`\| W_{\mathrm{df}} (H \mathbf{x} - \mathbf{s}) \|_1` and
        regularisation term is :math:`\| W_{\mathrm{tv}}
        \sqrt{(G_r \mathbf{x})^2 + (G_c \mathbf{x})^2}\|_1`.
        """

        if self.real_dtype:
            gvr = self.obfn_gvar()
        else:
            gvr = np.abs(self.obfn_gvar())
        dfd = np.sum(self.Wdf * np.abs(gvr[..., -1]))
        reg = np.sum(self.Wtv * np.sqrt(np.sum(gvr[..., 0:-1]**2,
                                               axis=self.saxes)))
        obj = dfd + self.lmbda*reg
        return (obj, dfd, reg)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs,)



    def cnst_A(self, X, Xf=None):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint.  In this case :math:`A \mathbf{x} = (G_r^T \;\;
        G_c^T \;\; H)^T \mathbf{x}`.
        """

        if Xf is None:
            Xf = self.fftn(X, axes=self.axes)
        return self.ifftn(self.GAf*Xf[..., np.newaxis], self.axsz,
                          axes=self.axes)



    def cnst_AT(self, X):
        r"""Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{x} = (G_r^T \;\; G_c^T \;\; H^T) \mathbf{x}`.
        """

        Xf = self.fftn(X, axes=self.axes)
        return np.sum(self.ifftn(np.conj(self.GAf)*Xf, self.axsz,
                                 axes=self.axes), axis=self.Y.ndim-1)



    def cnst_B(self, Y):
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem
        constraint.  In this case :math:`B \mathbf{y} = -\mathbf{y}`.
        """

        return -Y



    def cnst_c(self):
        r"""Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = (\mathbf{0} \;\;
        \mathbf{0} \;\; \mathbf{s})`.
        """

        c = np.zeros(self.S.shape + (len(self.axes)+1,), self.dtype)
        c[..., -1] = self.S
        return c



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        return self.rho*np.linalg.norm(self.cnst_AT(self.U))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho*np.linalg.norm(U)
