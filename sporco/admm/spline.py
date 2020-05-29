# -*- coding: utf-8 -*-
# Copyright (C) 2015-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Classes for ADMM algorithms for :math:`\ell_1` spline optimisation"""

from __future__ import division, absolute_import

import copy
import numpy as np

from sporco.admm import admm
from sporco.fft import dctii, idctii
from sporco.linalg import rrs
from sporco.prox import prox_l1


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class SplineL1(admm.ADMM):
    r"""ADMM algorithm for the :math:`\ell_1`-spline problem
    for equi-spaced samples :cite:`garcia-2010-robust`,
    :cite:`tepper-2013-fast`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
        \| W(\mathbf{x} - \mathbf{s}) \|_1 + \frac{\lambda}{2} \;
        \| D \mathbf{x} \|_2^2 \;\;,

    where :math:`D = \left( \begin{array}{ccc} -1 & 1 & & & \\
    1 & -2 & 1 & & \\ & \ddots & \ddots & \ddots &  \\
    & & 1 & -2 & 1 \\ & & & 1 & -1 \end{array} \right)\;`,
    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
        \| W \mathbf{y} \|_1 + \frac{\lambda}{2} \;
        \| D \mathbf{x} \|_2^2  \;\; \text{such that} \;\;
        \mathbf{x} - \mathbf{y} = \mathbf{s} \;\;.

    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`\| W (\mathbf{x}
       - \mathbf{s}) \|_1`

       ``Reg`` : Value of regularisation term :math:`\frac{1}{2} \| D
       \mathbf{x} \|_2^2`

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
        """SplineL1 algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMM.Options`, together with
        additional options:

          ``gEvalY`` : Flag indicating whether the :math:`g` component
          of the objective function should be evaluated using variable
          Y (``True``) or X (``False``) as its argument.

          ``DFidWeight`` : Data fidelity weight matrix.

          ``LinSolveCheck`` : If ``True``, compute relative residual
          of X step solver.
        """

        defaults = copy.deepcopy(admm.ADMM.Options.defaults)
        defaults.update({'gEvalY': True, 'RelaxParam': 1.8,
                         'DFidWeight': 1.0, 'LinSolveCheck': False
                        })
        defaults['AutoRho'].update({'Enabled': True, 'Period': 1,
                                    'AutoScaling': True, 'Scaling': 1000.0,
                                    'RsdlRatio': 1.2})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              SplineL1 algorithm options
            """

            if opt is None:
                opt = {}
            admm.ADMM.Options.__init__(self, opt)

            if self['AutoRho', 'RsdlTarget'] is None:
                self['AutoRho', 'RsdlTarget'] = 1.0



    itstat_fields_objfn = ('ObjFun', 'DFid', 'Reg')
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('Fnc', 'DFid', 'Reg')
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', 'Reg': 'Reg'}



    def __init__(self, S, lmbda, opt=None, axes=(0, 1)):
        """
        Parameters
        ----------
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        opt : SplineL1.Options object
          Algorithm options
        axes : tuple or list
          Axes on which spline regularisation is to be applied
        """

        if opt is None:
            opt = SplineL1.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        self.axes = axes
        self.lmbda = self.dtype.type(lmbda)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(2.0*self.lmbda + 0.1),
                      dtype=self.dtype)

        Nx = S.size
        super(SplineL1, self).__init__(Nx, S.shape, S.shape, S.dtype, opt)

        self.S = np.asarray(S, dtype=self.dtype)
        self.Wdf = np.asarray(self.opt['DFidWeight'], dtype=self.dtype)

        ashp = [1,] * S.ndim
        for ax in axes:
            ashp[ax] = S.shape[ax]
        self.Alpha = np.zeros(ashp, dtype=self.dtype)
        for ax in axes:
            ashp = [1,] * S.ndim
            ashp[ax] = S.shape[ax]
            axn = np.arange(0, ashp[ax]).reshape(ashp)
            self.Alpha += -2.0 + 2.0*np.cos(axn*np.pi/float(ashp[ax]))
        self.Gamma = 1.0 / (1.0 + (self.lmbda/self.rho)*(self.Alpha**2))



    def uinit(self, ushape):
        """Return initialiser for working variable U."""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            return (self.Wdf/self.rho)*np.sign(self.Y)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`."""

        self.X = idctii(self.Gamma*dctii(self.Y + self.S - self.U,
                                         axes=self.axes), axes=self.axes)
        if self.opt['LinSolveCheck']:
            self.xrrs = rrs(
                self.X + (self.lmbda/self.rho) *
                idctii((self.Alpha**2) *
                          dctii(self.X, axes=self.axes),
                          axes=self.axes), self.Y + self.S - self.U)
        else:
            self.xrrs = None



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y = prox_l1(self.AX - self.S + self.U, self.Wdf / self.rho)



    def rhochange(self):
        """Action to be taken when rho parameter is changed."""

        self.Gamma = 1.0 / (1.0 + (self.lmbda/self.rho)*(self.Alpha**2))



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
        regularisation term is :math:`\| D \mathbf{x} \|_2^2`.
        """

        gvr = self.obfn_gvar()
        dfd = np.sum(np.abs(self.Wdf * gvr))
        reg = 0.5*np.linalg.norm(
            idctii(self.Alpha*dctii(self.X, axes=self.axes),
                   axes=self.axes))**2
        obj = dfd + self.lmbda*reg
        return (obj, dfd, reg)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs,)



    def cnst_A(self, X):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint.  In this case :math:`A \mathbf{x} = \mathbf{x}`.
        """

        return X



    def cnst_AT(self, X):
        r"""Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}`
        is a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{x} = \mathbf{x}`.
        """

        return X



    def cnst_B(self, Y):
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem
        constraint.  In this case :math:`B \mathbf{y} = -\mathbf{y}`.
        """

        return -Y



    def cnst_c(self):
        r"""Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = \mathbf{s}`.
        """

        return self.S
