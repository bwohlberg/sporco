#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""ADMM algorithm for the CMOD problem"""

from __future__ import division
from __future__ import absolute_import

import numpy as np
from scipy import linalg
import copy
import collections

from sporco.admm import admm

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class CnstrMOD(admm.ADMMEqual):
    """ADMM algorithm for a constrained variant of the Method of Optimal
    Directions (MOD) :cite:`engan-1999-method` problem, referred to here
    as Constrained MOD (CMOD).

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_D \| D X - S \|_2^2 \quad \\text{such that}
       \quad \| \mathbf{d}_m \|_2 = 1 \;\;,

    where :math:`\mathbf{d}_m` is column :math:`m` of matrix :math:`D`,
    via the ADMM problem

    .. math::
       \mathrm{argmin}_D \| D X - S \|_2^2 + \iota_C(G) \quad
       \\text{such that} \quad D = G \;\;,

    where :math:`\iota_C(\cdot)` is the indicator function of feasible
    set :math:`C` consisting of matrices with unit-norm columns.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``DFid`` :  Value of data fidelity term \
       :math:`(1/2) \|  D X - S \|_2^2`

       ``Cnstr`` : Constraint violation measure

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
        """CMOD algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMMEqual.Options`, together with
        additional options:

        ``AuxVarObj`` : Flag indicating whether the objective function \
        should be evaluated using variable X  (``False``) or Y (``True``) \
        as its argument

        ``ZeroMean`` : Flag indicating whether the solution dictionary \
        :math:`D` should have zero-mean components

        """

        defaults = copy.deepcopy(admm.ADMMEqual.Options.defaults)
        defaults.update({'AuxVarObj' : True, 'ReturnX' : False,
                        'RelaxParam' : 1.8, 'ZeroMean' : False})
        defaults['AutoRho'].update({'Enabled' : True})


        def __init__(self, opt=None):
            """Initialise CMOD algorithm options object."""

            if opt is None:
                opt = {}
            admm.ADMMEqual.Options.__init__(self, opt)

            if self['AuxVarObj']:
                self['fEvalX'] = False
                self['gEvalY'] = True
            else:
                self['fEvalX'] = True
                self['gEvalY'] = False

            if self['AutoRho','RsdlTarget'] is None:
                self['AutoRho','RsdlTarget'] = 1.0



        def set_K(self, K, override=False):
            """Set parameters depending on K value"""

            if override or self['rho'] is None:
                self['rho'] = K / 500.0



    IterationStats = collections.namedtuple('IterationStats',
                ['Iter', 'DFid', 'Cnstr', 'PrimalRsdl', 'DualRsdl',
                 'EpsPrimal', 'EpsDual', 'Rho', 'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    hdrtxt = ['Itn', 'DFid', 'Cnstr', 'r', 's', 'rho']
    """Display column header text"""
    hdrval = {'Itn' : 'Iter', 'DFid' : 'DFid', 'Cnstr' : 'Cnstr',
              'r' : 'PrimalRsdl', 's' : 'DualRsdl', 'rho' : 'Rho'}
    """Dictionary mapping display column headers to IterationStats entries"""



    def __init__(self, A, S, dsz=None, opt=None):
        """
        Initialise a CnstrMOD object with problem parameters.

        Parameters
        ----------
        A : array_like, shape (M, K)
          Sparse representation coefficient matrix
        S : array_like, shape (N, K)
          Signal vector or matrix
        dsz : tuple
          Dictionary size
        opt : :class:`CnstrMOD.Options` object
          Algorithm options
        """

        if opt is None:
            opt = CnstrMOD.Options()

        Nc = S.shape[0]
        # If A not specified, get dictionary size from dsz
        if A is None:
            Nm = dsz[0]
        else:
            Nm = A.shape[0]
        Nx = Nc*Nm
        super(CnstrMOD, self).__init__(Nx, opt)

        # Create constraint set projection function
        self.Pcn = getPcn(opt)

        # Set rho value (computed from K if not specified)
        self.opt.set_K(S.shape[1])
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
                self.U = self.Y
        else:
            self.U = self.opt['U0']

        self.S = S

        self.runtime += self.timer.elapsed()

        if A is not None:
            self.setcoef(A)



    def setcoef(self, A):
        """Set coefficient array."""

        self.timer.start()
        self.A = A
        self.SAT = self.S.dot(A.T)
        # Factorise dictionary for efficient solves
        self.lu, self.piv = factorise(A, self.rho)
        self.runtime += self.timer.elapsed()



    def xstep(self):
        """Minimise Augmented Lagrangian with respect to x."""

        self.X = linsolve(self.A, self.rho, self.lu, self.piv,
                          self.SAT + self.rho*(self.Y - self.U))


    def ystep(self):
        """Minimise Augmented Lagrangian with respect to y."""

        self.Y = self.Pcn(self.AX + self.U)



    def iteration_stats(self, k, r, s, epri, edua, tk):
        """
        Construct iteration stats record tuple. Data fidelity term is
        :math:`(1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2` and
        measure of constraint violation is
        :math:`\| P(\mathbf{y}) -  \mathbf{y}\|_2`.
        """

        dfd = 0.5*linalg.norm((self.obfn_fvar().dot(self.A) - self.S))**2
        cns = linalg.norm((self.Pcn(self.obfn_gvar()) - self.obfn_gvar()))
        itst = type(self).IterationStats(k, dfd, cns, r, s, epri, edua,
                                         self.rho, tk)
        return itst



    def rhochange(self):
        """Re-factorise matrix when rho changes"""

        self.lu, self.piv = factorise(self.A, self.rho)



def getPcn(opt):
    """Construct constraint set projection function"""

    if opt['ZeroMean']:
        return lambda x: normalise(zeromean(x))
    else:
        return normalise


def zeromean(v):
    """Subtract mean of each column of matrix."""

    return v - np.mean(v, 0)


def normalise(v):
    """Normalise columns of matrix."""

    vn = np.sqrt(np.sum(v**2, 0))
    vn[vn == 0] = 1.0
    return v / vn


def factorise(A, rho):
    """Compute factorisation of either :math:`A^T A + \\rho I`
    or :math:`A A^T + \\rho I`, depending on which matrix is smaller."""

    N, M = A.shape
    # If N < M it is cheaper to factorise A*A^T' + rho*I and then use the
    # matrix inversion lemma to compute the inverse of A^T*A + rho*I
    if N >= M:
        lu, piv = linalg.lu_factor(A.T.dot(A) + rho*np.identity(M))
    else:
        lu, piv = linalg.lu_factor(A.dot(A.T) + rho*np.identity(N))
    return lu, piv


def linsolve(A, rho, lu, piv, b):
    """Solve the linear system :math:`(A A^T + \\rho I)\\mathbf{x} =
    \\mathbf{b}`."""

    N, M = A.shape
    if N >= M:
        x = (b - linalg.lu_solve((lu, piv), b.dot(A).T).T.dot(A.T)) / rho
    else:
        x = linalg.lu_solve((lu, piv), b.T).T
    return x
