#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithm for the BPDN problem"""

from __future__ import division
from __future__ import absolute_import

import numpy as np
from scipy import linalg
import copy
import collections

from sporco.admm import admm
import sporco.linalg as sl

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class BPDN(admm.ADMMEqual):
    """ADMM algorithm for the Basis Pursuit DeNoising (BPDN)
    :cite:`chen-1998-atomic` problem.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1

    via the ADMM problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{y} \|_1
       \quad \\text{such that} \quad \mathbf{x} = \mathbf{y} \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term \
       :math:`(1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term \
       :math:`\| \mathbf{x} \|_1`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance \
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance \
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """


    class Options(admm.ADMMEqual.Options):
        """BPDN algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMMEqual.Options`, together with
        additional options:

        ``AuxVarObj`` : Flag indicating whether the objective function \
        should be evaluated using variable X  (``False``) or Y (``True``) \
        as its argument

        ``L1Weight`` : An array of weights for the :math:`\ell^1`
        norm. The array shape must be such that the array is
        compatible for multiplication with the X/Y variables. If this
        option is defined, the regularization term is :math:`\lambda \|
        \mathbf{w} \odot \mathbf{x} \|_1` where :math:`\mathbf{w}`
        denotes the weighting array.

        ``NonNegCoef`` : If ``True``, force solution to be non-negative.
        """

        defaults = copy.deepcopy(admm.ADMMEqual.Options.defaults)
        defaults.update({'AuxVarObj' : True, 'ReturnX' : False,
                        'RelaxParam' : 1.8, 'L1Weight' : 1.0,
                        'NonNegCoef' : False})
        defaults['AutoRho'].update({'Enabled' : True, 'Period' : 10,
                                    'AutoScaling' : True, 'Scaling' : 1000.0,
                                    'RsdlRatio' : 1.2})

        def __init__(self, opt=None):
            """Initialise BPDN algorithm options object."""

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
                 'EpsPrimal', 'EpsDual', 'Rho', 'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    hdrtxt = ['Itn', 'Fnc', 'DFid', 'l1', 'r', 's', 'rho']
    """Display column header text"""
    hdrval = {'Itn' : 'Iter', 'Fnc' : 'ObjFun', 'DFid' : 'DFid', 'l1' : 'RegL1',
              'r' : 'PrimalRsdl', 's' : 'DualRsdl', 'rho' : 'Rho'}
    """Dictionary mapping display column headers to IterationStats entries"""



    def __init__(self, D, S, lmbda=None, opt=None):
        """
        Initialise a BPDN object with problem parameters.

        Parameters
        ----------
        D : array_like, shape (N, M)
          Dictionary matrix
        S : array_like, shape (N, K)
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        opt : :class:`BPDN.Options` object
          Algorithm options
        """

        Nr, Nc = D.shape
        Nm = S.shape[1]
        Nx = Nc*Nm
        if opt is None:
            opt = BPDN.Options()
        super(BPDN, self).__init__(Nx, opt)

        # Set default lambda value if not specified
        if lmbda is None:
            DTS = D.T.dot(S)
            self.lmbda = 0.1*abs(DTS).max()
        else:
            self.lmbda = lmbda

        # Set rho value (computed from lambda if not specified)
        self.opt.set_lambda(self.lmbda)
        self.rho = self.opt['rho']

        # Determine working data type
        self.opt.set_dtype(S.dtype)
        dtype = self.opt['DataType']

        # Initial values for Y
        if  self.opt['Y0'] is None:
            self.Y = np.zeros((Nc, Nm), dtype)
        else:
            self.Y = self.opt['Y0']
        self.Yprev = self.Y

        # Initial value for U
        if  self.opt['U0'] is None:
            if  self.opt['Y0'] is None:
                self.U = np.zeros((Nc, Nm), dtype)
            else:
                # If Y0 is given, but not U0, then choose the initial
                # U so that the relevant dual optimality criterion
                # (see (3.10) in boyd-2010-distributed) is satisfied.
                self.U = (self.lmbda/self.rho)*np.sign(self.Y)
        else:
            self.U = self.opt['U0']

        self.runtime += self.timer.elapsed()

        self.S = S
        self.setdict(D)



    def setdict(self, D):
        """Set dictionary array."""

        self.timer.start()
        self.D = D
        self.DTS = D.T.dot(self.S)
        # Factorise dictionary for efficient solves
        self.lu, self.piv = factorise(D, self.rho)
        self.runtime += self.timer.elapsed()



    def xstep(self):
        """Minimise Augmented Lagrangian with respect to x."""

        self.X = linsolve(self.D, self.rho, self.lu, self.piv,
                          self.DTS + self.rho*(self.Y - self.U))



    def ystep(self):
        """Minimise Augmented Lagrangian with respect to y."""

        self.Y = sl.shrink1(self.AX + self.U,
                            (self.lmbda/self.rho)*self.opt['L1Weight'])
        if self.opt['NonNegCoef']:
            self.Y[self.Y < 0.0] = 0.0



    def iteration_stats(self, k, r, s, epri, edua, tk):
        """
        Construct iteration stats record tuple. Data fidelity term is
        :math:`(1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2` and
        regularisation term is :math:`\| \mathbf{y} \|_1`.
        """

        dfd = 0.5*linalg.norm((self.D.dot(self.obfn_fvar()) - self.S))**2
        reg = linalg.norm((self.opt['L1Weight'] * self.obfn_gvar()).ravel(), 1)
        obj = dfd + self.lmbda*reg
        itst = type(self).IterationStats(k, obj, dfd, reg, r, s,
                                         epri, edua, self.rho, tk)
        return itst



    def rhochange(self):
        """Re-factorise matrix when rho changes."""

        self.lu, self.piv = factorise(self.D, self.rho)



def factorise(D, rho):
    """Compute factorisation of either :math:`D^T D + \\rho I`
    or :math:`D D^T + \\rho I`, depending on which matrix is smaller."""

    N, M = D.shape
    # If N < M it is cheaper to factorise D*D^T' + rho*I and then use the
    # matrix inversion lemma to compute the inverse of D^T*D + rho*I
    if N >= M:
        lu, piv = linalg.lu_factor(D.T.dot(D) + rho*np.identity(M))
    else:
        lu, piv = linalg.lu_factor(D.dot(D.T) + rho*np.identity(N))
    return lu, piv



def linsolve(D, rho, lu, piv, b):
    """Solve the linear system :math:`(D^T D + \\rho I)\\mathbf{x} =
    \\mathbf{b}`."""

    N, M = D.shape
    if N >= M:
        x = linalg.lu_solve((lu, piv), b)
    else:
        x = (b - D.T.dot(linalg.lu_solve((lu, piv), D.dot(b), 1))) / rho
    return x




class ElasticNet(BPDN):
    """ADMM algorithm for the elastic net :cite:`zou-2005-regularization`
    problem.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1
       + (\mu/2) \| \mathbf{x} \|_2^2

    via the ADMM problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{y} \|_1
       + (\mu/2) \| \mathbf{x} \|_2^2 \quad \\text{such that} \quad 
       \mathbf{x} = \mathbf{y} \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term \
       :math:`(1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term \
       :math:`\| \mathbf{x} \|_1`

       ``RegL2`` : Value of regularisation term \
       :math:`(1/2) \| \mathbf{x} \|_2^2`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual Residual

       ``EpsPrimal`` : Primal residual stopping tolerance \
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance \
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """


    IterationStats = collections.namedtuple('IterationStats',
                ['Iter', 'ObjFun', 'DFid', 'RegL1', 'RegL2',
                 'PrimalRsdl', 'DualRsdl', 'EpsPrimal', 'EpsDual',
                 'Rho', 'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    hdrtxt = ['Itn', 'Fnc', 'DFid', 'l1', 'l2', 'r', 's', 'rho']
    """Display column header text"""
    hdrval = {'Itn' : 'Iter', 'Fnc' : 'ObjFun', 'DFid' : 'DFid',
              'l1' : 'RegL1', 'l2' : 'RegL2', 'r' : 'PrimalRsdl',
              's' : 'DualRsdl', 'rho' : 'Rho'}
    """Dictionary mapping display column headers to IterationStats entries"""



    def __init__(self, D, S, lmbda=None, mu=0.0, opt=None):
        """
        Initialise an ElasticNet object with problem parameters.

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
        opt : :class:`BPDN.Options` object
          Algorithm options
        """

        if opt is None:
            opt = BPDN.Options()
        self.mu = mu
        super(ElasticNet, self).__init__(D, S, lmbda, opt)




    def setdict(self, D):
        """Set dictionary array."""

        self.timer.start()
        self.D = D
        self.DTS = D.T.dot(self.S)
        # Factorise dictionary for efficient solves
        self.lu, self.piv = factorise(D, self.mu + self.rho)
        self.runtime += self.timer.elapsed()



    def xstep(self):
        """Minimise Augmented Lagrangian with respect to x."""

        self.X = linsolve(self.D, self.mu + self.rho, self.lu, self.piv,
                          self.DTS + self.rho*(self.Y - self.U))




    def iteration_stats(self, k, r, s, epri, edua, tk):
        """
        Construct iteration stats record tuple. Data fidelity term is
        :math:`(1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2` and
        regularisation terms are :math:`\| \mathbf{y} \|_1` and
        :math:`(1/2)\| \mathbf{y} \|_2^2`.
        """

        dfd = 0.5*linalg.norm((self.D.dot(self.obfn_fvar()) - self.S))**2
        rl1 = linalg.norm((self.opt['L1Weight'] * self.obfn_gvar()).ravel(), 1)
        rl2 = 0.5*linalg.norm(self.obfn_gvar())**2
        obj = dfd + self.lmbda*rl1 + self.mu*rl2
        itst = type(self).IterationStats(k, obj, dfd, rl1, rl2, r, s,
                                         epri, edua, self.rho, tk)
        return itst




    def rhochange(self):
        """Re-factorise matrix when rho changes."""

        self.lu, self.piv = factorise(self.D, self.mu + self.rho)
