#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithms for Robust PCA optimisation"""

from __future__ import division
from __future__ import absolute_import

import numpy as np
import copy
import collections

from sporco.admm import admm
import sporco.linalg as sl

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class RobustPCA(admm.ADMM):
    """ADMM algorithm for Robust PCA problem :cite:`candes-2011-robust`
    :cite:`cai-2010-singular`.
   
    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{X, Y} \;
        \| X \|_* + \lambda \| Y \|_1 \quad \\text{such that}
        \quad X + Y = S \;\;.

    This problem is unusual in that it is already in ADMM form without
    the need for any variable splitting.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``NrmNuc`` :  Value of nuclear norm term :math:`\| X \|_*`

       ``NrmL1`` : Value of :math:`\ell^1` norm term :math:`\| Y \|_1`

       ``Cnstr`` : Constraint violation :math:`\| X + Y - S\|_2`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance \
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance \
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time

    """



    class Options(admm.ADMM.Options):
        """RobustPCA algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMM.Options`, together with
        an additional option:

        ``gEvalY`` : Flag indicating whether the :math:`g` component of the \
        objective function should be evaluated using variable Y \
        (``True``) or X (``False``) as its argument
        """

        defaults = copy.deepcopy(admm.ADMM.Options.defaults)
        defaults.update({'gEvalY' : True, 'RelaxParam' : 1.8})
        defaults['AutoRho'].update({'Enabled' : True, 'Period' : 1,
                                    'AutoScaling' : True, 'Scaling' : 1000.0,
                                    'RsdlRatio' : 1.2})


        def __init__(self, opt=None):
            """Initialise RobustPCA algorithm options object."""

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
                ['Iter', 'ObjFun', 'NrmNuc', 'NrmL1', 'Cnstr', 'PrimalRsdl',
                 'DualRsdl', 'EpsPrimal', 'EpsDual', 'Rho', 'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    hdrtxt = ['Itn', 'Fnc', 'Nuc', 'L1', 'Cnstr', 'r', 's', 'rho']
    """Display column header text"""
    hdrval = {'Itn' : 'Iter', 'Fnc' : 'ObjFun', 'Nuc' : 'NrmNuc',
              'L1' : 'NrmL1', 'Cnstr' : 'Cnstr', 'r' :
              'PrimalRsdl', 's' : 'DualRsdl', 'rho' : 'Rho'}
    """Dictionary mapping display column headers to IterationStats entries"""


    def __init__(self, S, lmbda=None, opt=None):
        """
        Initialise a RobustPCA object with problem parameters.

        Parameters
        ----------
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        opt : RobustPCA.Options object
          Algorithm options
        """

        if opt is None:
            opt = RobustPCA.Options()
        Nx = S.size
        Nc = Nx
        super(RobustPCA, self).__init__(Nx, Nc, opt)

        # Set default lambda value if not specified
        if lmbda is None:
            self.lmbda = 1.0 / np.sqrt(S.shape[0])
        else:
            self.lmbda = lmbda

        # Set rho value (computed from lambda if not specified)
        self.opt.set_lambda(self.lmbda)
        self.rho = self.opt['rho']

        # Determine working data type
        self.opt.set_dtype(S.dtype)
        self.dtype = self.opt['DataType']

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
                self.U = (self.lmbda/self.rho)*np.sign(self.Y)
        else:
            self.U = self.opt['U0']

        self.S = S

        self.runtime += self.timer.elapsed()



    def solve(self):
        """Start (or re-start) optimisation."""

        super(RobustPCA, self).solve()
        return self.X, self.Y



    def xstep(self):
        """Minimise Augmented Lagrangian with respect to x."""

        self.X, self.ss = shrinksv(self.S - self.Y - self.U, 1.0 / self.rho)



    def ystep(self):
        """Minimise Augmented Lagrangian with respect to y."""

        self.Y = sl.shrink1(self.S - self.AX - self.U, self.lmbda/self.rho)



    def obfn_gvar(self):
        """Variable to be evaluated in computing regularisation term,
        depending on 'gEvalY' option value."""

        if self.opt['gEvalY']:
            return self.Y
        else:
            return self.cnst_c() - self.cnst_A(self.X) 



    def iteration_stats(self, k, r, s, epri, edua, tk):
        """
        Construct iteration stats record tuple.
        """

        gvr = self.obfn_gvar()
        if 0:
            rnn = nucnorm(self.X)
        else:
            rnn = np.sum(self.ss)
        rl1 = np.sum(np.abs(gvr))
        cns = np.linalg.norm(self.X + self.Y - self.S)
        obj = rnn + self.lmbda*rl1
        itst = type(self).IterationStats(k, obj, rnn, rl1, cns, r, s, epri,
                                edua, self.rho, tk)
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

        return Y


    def cnst_c(self):
        """Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case
        :math:`\mathbf{c} = \mathbf{s}`."""

        return self.S



def shrinksv(x, alpha):
    """Shrinkage of singular values"""

    U, s, V = np.linalg.svd(x, full_matrices=False)
    ss = np.maximum(0.0, s - alpha)
    return np.dot(U, np.dot(np.diag(ss), V)), ss



def nucnorm(x):
    """Nuclear norm"""

    s = np.linalg.svd(x, compute_uv=False)
    return np.sum(s)
