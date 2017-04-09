#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: bpdn.BPDN"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco import util
from sporco import plot
from sporco.admm import bpdn


# Signal and dictionary size
N = 512
M = 4*N
# Number of non-zero coefficients in generator
L = 32
# Noise level
sigma = 0.5


# Construct random dictionary and random sparse coefficients
np.random.seed(12345)
D = np.random.randn(N, M)
x0 = np.zeros((M, 1))
si = np.random.permutation(list(range(0, M-1)))
x0[si[0:L]] = np.random.randn(L, 1)
# Construct reference and noisy signal
s0 = D.dot(x0)
s = s0 + sigma*np.random.randn(N,1)


# Set BPDN options
opt = bpdn.BPDN.Options({'Verbose' : False, 'MaxMainIter' : 500,
                    'RelStopTol' : 1e-3, 'AutoRho' : {'RsdlTarget' : 1.0}})


# Function computing reconstruction error at lmbda
def evalerr(prm):
    lmbda = prm[0]
    b = bpdn.BPDN(D, s, lmbda, opt)
    x = b.solve()
    return np.sum(np.abs(x-x0))


# Parallel evalution of error function on lmbda grid
lrng = np.logspace(1, 2, 10)
sprm, sfvl, fvmx, sidx = util.grid_search(evalerr, (lrng,))
lmbda = sprm[0]
print('Minimum ℓ1 error: %5.2f at 𝜆 = %.2e' % (sfvl, lmbda))


# Initialise and run BPDN object for best lmbda
opt['Verbose'] = True
b = bpdn.BPDN(D, s, lmbda, opt)
b.solve()
print("BPDN solve time: %.2fs" % b.timer.elapsed('solve'))


# Plot results
plot.plot(np.hstack((x0, b.Y)), fgnm=1, title='Sparse representation',
          lgnd=['Reference', 'Reconstructed'])


# Plot lmbda error curve, functional value, residuals, and rho
its = b.getitstat()
fig2 = plot.figure(2, figsize=(14,14))
plot.subplot(2,2,1)
plot.plot(fvmx, x=lrng, ptyp='semilogx', xlbl='$\lambda$',
          ylbl='Error', fgrf=fig2)
plot.subplot(2,2,2)
plot.plot(its.ObjFun, fgrf=fig2, xlbl='Iterations', ylbl='Functional')
plot.subplot(2,2,3)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T, fgrf=fig2,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'])
plot.subplot(2,2,4)
plot.plot(its.Rho, fgrf=fig2, xlbl='Iterations', ylbl='Penalty Parameter')
fig2.show()


# Wait for enter on keyboard
input()
