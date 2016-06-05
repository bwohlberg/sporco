#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithms for :math:`\ell^1` spline optimisation"""

from __future__ import division
from __future__ import absolute_import

import numpy as np
from scipy import linalg
import copy
import collections

from sporco.admm import admm
import sporco.linalg as sl

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class SplineL1(admm.ADMM):
    """ADMM algorithm for the :math:`\ell^1`-spline problem
    for equi-spaced samples :cite:`garcia-2010-robust`,
    :cite:`tepper-2013-fast`.
   
    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
        \| W(\mathbf{x} - \mathbf{s}) \|_1 + \\frac{\lambda}{2} \;
        \| D \mathbf{x} \|_2^2 \;\;,

    where :math:`D = \\left( \\begin{array}{ccc} -1 & 1 & & & \\\\ 
    1 & -2 & 1 & & \\\\ & \\ddots & \\ddots & \ddots &  \\\\ 
    & & 1 & -2 & 1 \\\\ & & & 1 & -1 \\end{array} \\right)`,

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
        \| W \mathbf{y} \|_1 + \\frac{\lambda}{2} \;
        \| D \mathbf{x} \|_2^2  \;\; \\text{such that} \;\; 
        \mathbf{x} - \mathbf{y} = \mathbf{s} \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term \
       :math:`\| W (\mathbf{x} - \mathbf{s}) \|_1`

       ``Reg`` : Value of regularisation term \
       :math:`\\frac{1}{2} \| D \mathbf{x} \|_2^2`

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
        """SplineL1 algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMM.Options`, together with
        additional options:

        ``gEvalY`` : Flag indicating whether the :math:`g` component of the \
        objective function should be evaluated using variable Y \
        (``True``) or X (``False``) as its argument

        ``DFidWeight`` : Data fidelity weight matrix
 
        ``LinSolveCheck`` : If ``True``, compute relative residual of
        X step solver
        """

        defaults = copy.deepcopy(admm.ADMM.Options.defaults)
        defaults.update({'gEvalY' : True, 'RelaxParam' : 1.8,
                         'DFidWeight' : 1.0, 'LinSolveCheck' : False
                        })
        defaults['AutoRho'].update({'Enabled' : True, 'Period' : 1,
                                    'AutoScaling' : True, 'Scaling' : 1000.0,
                                    'RsdlRatio' : 1.2})


        def __init__(self, opt=None):
            """Initialise SplineL1 algorithm options object."""

            if opt is None:
                opt = {}
            admm.ADMM.Options.__init__(self, opt)


        def set_lambda(self, lmbda, override=False):
            """Set parameters depending on lambda value"""

            if override or self['rho'] is None:
                self['rho'] = 2*lmbda + 0.1
            if override or self['AutoRho','RsdlTarget'] is None:
                self['AutoRho','RsdlTarget'] = 1.0




    IterationStats = collections.namedtuple('IterationStats',
                ['Iter', 'ObjFun', 'DFid', 'Reg', 'PrimalRsdl', 'DualRsdl',
                 'EpsPrimal', 'EpsDual', 'Rho', 'XSlvRelRes', 'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    hdrtxt = ['Itn', 'Fnc', 'DFid', 'Reg', 'r', 's', 'rho']
    """Display column header text"""
    hdrval = {'Itn' : 'Iter', 'Fnc' : 'ObjFun', 'DFid' : 'DFid',
              'Reg' : 'Reg', 'r' : 'PrimalRsdl', 's' : 'DualRsdl',
              'rho' : 'Rho'}
    """Dictionary mapping display column headers to IterationStats entries"""


    def __init__(self, S, lmbda, opt=None, axes=(0,1)):
        """
        Initialise a SplineL1 object with problem parameters.

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
        Nx = S.size
        Nc = Nx
        super(SplineL1, self).__init__(Nx, Nc, opt)

        self.axes = axes
        self.lmbda = lmbda

        # Set rho value (computed from lambda if not specified)
        self.opt.set_lambda(self.lmbda)
        self.rho = self.opt['rho']

        # Determine working data type
        self.opt.set_dtype(S.dtype)
        self.dtype = self.opt['DataType']

        self.S = S
        self.Wdf = self.opt['DFidWeight']

        # Initial values for Y
        if  self.opt['Y0'] is None:
            self.Y = np.zeros(S.shape, self.dtype)
        else:
            self.Y = self.opt['Y0']
        self.Yprev = self.Y

        # Initial value for U
        if  self.opt['U0'] is None:
            if  self.opt['Y0'] is None:
                self.U = np.zeros(S.shape, self.dtype)
            else:
                # If Y0 is given, but not U0, then choose the initial
                # U so that the relevant dual optimality criterion
                # (see (3.10) in boyd-2010-distributed) is satisfied.
                self.U = (self.Wdf / self.rho)*np.sign(self.Y)
        else:
            self.U = self.opt['U0']

        ashp = [1,] * S.ndim
        for ax in axes:
            ashp[ax] = S.shape[ax]
        self.Alpha = np.zeros(ashp, dtype=self.dtype)
        for ax in axes:
            ashp = [1,] * S.ndim
            ashp[ax] = S.shape[ax]
            axn = np.arange(0,ashp[ax]).reshape(ashp)
            self.Alpha += -2.0 + 2.0*np.cos(axn*np.pi/float(ashp[ax]))
        self.Gamma = 1.0 / (1.0 + (self.lmbda/self.rho)*(self.Alpha**2))

        self.runtime += self.timer.elapsed()




    def xstep(self):
        """Minimise Augmented Lagrangian with respect to x."""

        self.X = sl.idctii(self.Gamma*sl.dctii(self.Y + self.S - self.U,
                                         axes=self.axes), axes=self.axes)
        if self.opt['LinSolveCheck']:
            self.xrrs = sl.rrs(self.X + (self.lmbda/self.rho)*
                    sl.idctii((self.Alpha**2)*sl.dctii(self.X, axes=self.axes), 
                    axes=self.axes), self.Y + self.S - self.U)
        else:
            self.xrrs = None



    def ystep(self):
        """Minimise Augmented Lagrangian with respect to y."""

        self.Y = sl.shrink1(self.AX - self.S + self.U, self.Wdf / self.rho)


    def rhochange(self):
        """Action to be taken when rho parameter is changed."""

        self.Gamma = 1.0 / (1.0 + (self.lmbda/self.rho)*(self.Alpha**2))
        


    def obfn_gvar(self):
        """Variable to be evaluated in computing regularisation term,
        depending on 'gEvalY' option value."""

        if self.opt['gEvalY']:
            return self.Y
        else:
            return self.cnst_A(self.X) - self.cnst_c()



    def iteration_stats(self, k, r, s, epri, edua, tk):
        """
        Construct iteration stats record tuple. Data fidelity term is
        :math:`(1/2) \| \mathbf{x} - \mathbf{s} \|_2^2` and
        regularisation term is :math:`\| D \mathbf{x} \|_2^2`.
        """

        gvr = self.obfn_gvar()
        dfd = np.sum(np.abs(self.Wdf * gvr))
        reg = 0.5*linalg.norm(sl.idctii(
            self.Alpha*sl.dctii(self.X, axes=self.axes),
            axes=self.axes))**2
        obj = dfd + self.lmbda*reg
        itst = type(self).IterationStats(k, obj, dfd, reg, r, s, epri,
                                edua, self.rho, self.xrrs, tk)
        return itst



    def cnst_A(self, X):
        """Compute :math:`A \mathbf{x}` component of ADMM problem constraint.
        In this case
        :math:`A \mathbf{x} = \mathbf{x}`.
        """

        return X



    def cnst_AT(self, X):
        """Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{x} = \mathbf{x}`."""

        return X



    def cnst_B(self, Y):
        """Compute :math:`B \mathbf{y}` component of ADMM problem constraint.
        In this case :math:`B \mathbf{y} = -\mathbf{y}`."""

        return -Y


    def cnst_c(self):
        """Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case
        :math:`\mathbf{c} = \mathbf{s}`."""

        return self.S
