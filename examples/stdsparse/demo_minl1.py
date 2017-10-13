#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: bpdn.MinL1InL2Ball """

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

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
epsilon = 1.1 * np.linalg.norm(s0 - s)
opt = bpdn.MinL1InL2Ball.Options({'Verbose': True, 'MaxMainIter': 500,
                                  'RelStopTol': 1e-6, 'rho': 1e0,
                                  'AutoRho': {'Enabled': False}})


# Initialise and run BPDN object
b = bpdn.MinL1InL2Ball(D, s, epsilon, opt)
b.solve()
print("MinL1InL2Ball solve time: %.2fs" % b.timer.elapsed('solve'))


# Plot results
plot.plot(np.hstack((x0, b.X)), fgnm=1, title='Sparse representation',
          lgnd=['Reference', 'Reconstructed'])


# Plot functional value, residuals, and rho
its = b.getitstat()
fig2 = plot.figure(2, figsize=(21,7))
plot.subplot(1,3,1)
plot.plot(its.ObjFun, fgrf=fig2, xlbl='Iterations', ylbl='Functional')
plot.subplot(1,3,2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T, fgrf=fig2,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'])
plot.subplot(1,3,3)
plot.plot(its.Rho, fgrf=fig2, xlbl='Iterations', ylbl='Penalty Parameter')
fig2.show()


# Wait for enter on keyboard
input()
