#!/usr/bin/env python
#-*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
ADMM Consensus Example
======================

A simple example demonstrating how to construct a solver for an ADMM Consensus problem by specialising :class:`.admm.ADMMConsensus`.
"""

from __future__ import print_function
from builtins import input

import numpy as np

from sporco.admm import admm
import sporco.linalg as sl
import sporco.prox as sp
from sporco import plot


"""
Define class solving a simple synthetic problem demonstrating the construction of an ADMM Consensus solver derived from :class:`.admm.ADMMConsensus`.
"""

class ConsensusTest(admm.ADMMConsensus):
    r"""
    Solve the problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_k \| A_k \mathbf{x} - \mathbf{s}_k \|_2^2 + \lambda
       \| \mathbf{x} \|_1

   via an ADMM consensus problem

   .. math::
       \mathrm{argmin}_{\mathbf{x}_k, \mathbf{y}} \;
       (1/2) \sum_k \| A_k \mathbf{x}_k - \mathbf{s}_k \|_2^2 + \lambda
       \| \mathbf{y} \|_1 \;\; \text{s.t.} \;\;
       \mathbf{x}_k = \mathbf{y} \; \forall k
    """

    def __init__(self, A, s, lmbda, opt=None):
        r"""
        Initialise a ConsensusTest object with problem parameters.

        Parameters
        ----------
        A : list of ndarray
          A list of arrays representing matrices :math:`A_k`
        S : list of ndarray
          A list of arrays representing vectors :math:`\mathbf{s}_k`
        opt : :class:`.ADMMConsensus.Options` object
          Algorithm options
        """

        # Default solver options if none provided
        if opt is None:
            opt = admm.ADMMConsensus.Options()

        # Set object attributes corresponding to initialiser parameters
        self.A = A
        self.s = s
        self.lmbda = lmbda
        self.rho = opt['rho']
        # The number of separate components of the consensus problem
        Nb = len(A)
        # Construct a tuple representing the shape of the auxiliary
        # variable Y in the consensus problem
        shpY = (A[0].shape[1], s[0].shape[1] if s[0].ndim > 1 else 1)
        # Call parent class initialiser
        super(ConsensusTest, self).__init__(Nb, shpY, s[0].dtype, opt)

        # Construct list of products A_k^T s_k
        self.ATS = [A[i].T.dot(s[i]) for i in range(Nb)]
        # Compute an LU factorisation for each A_k
        self.rhochange()

        # Initialise working variables
        self.X = np.zeros(shpY + (Nb,))
        self.Y = np.zeros(shpY)
        self.U = np.zeros(shpY + (Nb,))



    def rhochange(self):
        r"""
        This method is called when the penalty parameter :math:`\rho` is
        updated by the parent class solve method. It computes an LU
        factorisation of :math:`A_k^T A_k + \rho I`.
        """

        self.lu = []
        self.piv = []
        for i in range(self.Nb):
            lu, piv = sl.lu_factor(self.A[i], self.rho)
            self.lu.append(lu)
            self.piv.append(piv)



    def obfn_fi(self, Xi, i):
        r"""
        Compute :math:`(1/2) \sum_k \| A_k \mathbf{x}_k - \mathbf{s}_k
        \|_2^2`.
        """

        return 0.5*np.linalg.norm(self.A[i].dot(Xi) - self.s[i])**2



    def obfn_g(self, Y):
        r"""
        Compute :math:`\lambda \| \mathbf{x} \|_1`.
        """

        return self.lmbda * np.sum(np.abs(Y))



    def xistep(self, i):
        r"""
        Minimise Augmented Lagrangian with respect to :math:`\mathbf{x}`
        component :math:`\mathbf{x}_i`.
        """

        self.X[..., i] = sl.lu_solve_ATAI(self.A[i], self.rho,
                    self.ATS[i] + self.rho*(self.Y - self.U[..., i]),
                    self.lu[i], self.piv[i])



    def prox_g(self, X, rho):
        r"""
        Proximal operator of :math:`(\lambda/\rho) \|\cdot\|_1`.
        """

        return sp.prox_l1(X, (self.lmbda/rho))


r"""
Construct random sparse vector $\mathbf{x}$, random $A_k$` matrices, and vectors $\mathbf{s}_k$ such that $A_k \mathbf{x} = \mathbf{s}_k$.
"""

np.random.seed(12345)
x = np.random.randn(64,1)
x[np.abs(x) < 1.25] = 0
A = [np.random.randn(8, 64) for i in range(8)]
s = [A[i].dot(x) for i in range(8)]


"""
Initialise and run `ConsensusTest` solver.
"""

lmbda = 1e-1
opt = ConsensusTest.Options({'Verbose': True, 'MaxMainIter': 250,
                            'AutoRho': {'Enabled': False},
                            'rho': 2e-1, 'RelaxParam': 1.2,
                            'fEvalX': False})
b = ConsensusTest(A, s, lmbda, opt)
yr = b.solve()
print("ConsensusTest solve time: %.2fs" % b.timer.elapsed('solve'))


"""
Plot reference and reconstructed sparse representations.
"""

plot.plot(np.hstack((x, yr)), title='Sparse representation',
        lgnd=['Reference', 'Reconstructed'])


"""
Plot functional value, residuals, and rho.
"""

its = b.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, ptyp='semilogy', xlbl='Iterations', ylbl='Functional',
          fig=fig)
plot.subplot(1, 3, 2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'], fig=fig);
plot.subplot(1, 3, 3)
plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter', fig=fig)
fig.show()


# Wait for enter on keyboard
input()
