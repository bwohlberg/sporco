#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: bpdn.BPDNJoint"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco import util
from sporco import plot
from sporco.admm import bpdn


# Signal and dictionary size
N = 32
M = 4*N
# Number of non-zero coefficients in generator
L = 12
# Number of signals
K = 16;
# Noise level
sigma = 0.5


# Construct random dictionary and random sparse coefficients
np.random.seed(12345)
D = np.random.randn(N, M)
x0 = np.zeros((M, K))
si = np.random.permutation(list(range(0, M-1)))
x0[si[0:L],:] = np.random.randn(L, K)
# Construct reference and noisy signal
s0 = D.dot(x0)
s = s0 + sigma*np.random.randn(N,K)


# Set BPDNJoint options
opt = bpdn.BPDNJoint.Options({'Verbose' : False, 'MaxMainIter' : 500,
                              'RelStopTol' : 1e-3, 'rho' : 10.0,
                              'AutoRho' : {'RsdlTarget' : 1.0}})


# Function computing reconstruction error for (lmbda, mu) pair
def evalerr(prm):
    lmbda = prm[0]
    mu = prm[1]
    b = bpdn.BPDNJoint(D, s, lmbda, mu, opt)
    x = b.solve()
    return np.sum(np.abs(x-x0))


# Parallel evalution of error function on lmbda,mu grid
lrng = np.logspace(-4, 0.5, 10)
mrng = np.logspace(0.5, 1.6, 10)
sprm, sfvl, fvmx, sidx = util.grid_search(evalerr, (lrng, mrng))
lmbda = sprm[0]
mu = sprm[1]
print('Minimum ℓ1 error: %5.2f at (𝜆,μ) = (%.2e, %.2e)' % (sfvl, lmbda, mu))


# Initialise and run BPDNJoint object for best lmbda and mu
opt['Verbose'] = True
b = bpdn.BPDNJoint(D, s, lmbda, mu, opt)
b.solve()
print("BPDNJoint solve time: %.2fs" % b.timer.elapsed('solve'))


# Display recovery results
fig1 = plot.figure(1, figsize=(6,8))
plot.subplot(1,2,1)
plot.imview(x0, fgrf=fig1, cmap=plot.cm.Blues, title='Reference')
plot.subplot(1,2,2)
plot.imview(b.Y, fgrf=fig1, cmap=plot.cm.Blues, title='Reconstruction')
fig1.show()


# Plot lmbda,mu error surface, functional value, residuals, and rho
its = b.getitstat()
fig2 = plot.figure(2, figsize=(14,14))
ax = fig2.add_subplot(2, 2, 1, projection='3d')
ax.locator_params(nbins=5, axis='y')
plot.surf(fvmx, x=np.log10(mrng), y=np.log10(lrng), xlbl='log($\mu$)',
          ylbl='log($\lambda$)', zlbl='Error', fgrf=fig2, axrf=ax)
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
