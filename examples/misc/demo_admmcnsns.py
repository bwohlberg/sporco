#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Example ADMM Consensus problem demonstrating admm.ADMMConsensus usage"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco.admm import admm
import sporco.linalg as spl
from sporco import plot



class ConsensusTest(admm.ADMMConsensus):

    def __init__(self, A, s, lmbda, opt=None):

        if opt is None:
            opt = admm.ADMMConsensus.Options()

        self.A = A
        self.s = s
        self.lmbda = lmbda
        self.rho = opt['rho']
        Nb = len(A)
        shpY = (A[0].shape[1], s[0].shape[1] if s[0].ndim > 1 else 1)
        super(ConsensusTest, self).__init__(Nb, shpY, s[0].dtype, opt)

        self.timer.start()
        self.ATS = [A[i].T.dot(s[i]) for i in range(Nb)]
        self.rhochange()

        self.X = np.zeros(shpY + (Nb,))
        self.Y = np.zeros(shpY)
        self.U = np.zeros(shpY + (Nb,))



    def rhochange(self):

        self.lu = []
        self.piv = []
        for i in range(self.Nb):
            lu, piv = spl.lu_factor(self.A[i], self.rho)
            self.lu.append(lu)
            self.piv.append(piv)



    def obfn_fi(self, Xi, i):

        return 0.5*np.linalg.norm(self.A[i].dot(Xi) - self.s[i])**2



    def obfn_g(self, Y):

        return self.lmbda * np.sum(np.abs(Y))



    def xistep(self, i):

        self.X[...,i] = spl.lu_solve_ATAI(self.A[i], self.rho, self.ATS[i] +
                                         self.rho*(self.Y - self.U[...,i]),
                                         self.lu[i], self.piv[i])



    def prox_g(self, X, rho):

        return spl.shrink1(X, (self.lmbda/rho))



if __name__ == "__main__":

    np.random.seed(12345)
    y = np.random.randn(64,1)
    y[np.abs(y) < 1.25] = 0
    A = [np.random.randn(8, 64) for i in range(8)]
    s = [A[i].dot(y) for i in range(8)]


    lmbda = 1e-1
    opt = ConsensusTest.Options({'Verbose': True, 'MaxMainIter': 250,
                                   'AutoRho': {'Enabled': False},
                                   'rho': 2e-1, 'RelaxParam': 1.2,
                                   'fEvalX': False})
    b = ConsensusTest(A, s, lmbda, opt)
    yr = b.solve()
    print("ConsensusTest solve time: %.2fs" % b.timer.elapsed('solve'))


    # Plot reference and reconstructed sparse representations
    plot.plot(np.hstack((y, yr)), fgnm=1, title='Sparse representation',
              lgnd=['Reference', 'Reconstructed'])


    # Plot functional value, residuals, and rho
    its = b.getitstat()
    fig2 = plot.figure(2, figsize=(21,7))
    plot.subplot(1,3,1)
    plot.plot(its.ObjFun, fgrf=fig2, ptyp='semilogy', xlbl='Iterations',
              ylbl='Functional')
    plot.subplot(1,3,2)
    plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T, fgrf=fig2,
              ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
              lgnd=['Primal', 'Dual']);
    plot.subplot(1,3,3)
    plot.plot(its.Rho, fgrf=fig2, xlbl='Iterations', ylbl='Penalty Parameter')
    fig2.show()

    # Wait for enter on keyboard
    input()
