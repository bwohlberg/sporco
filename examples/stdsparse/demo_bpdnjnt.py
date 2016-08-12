#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic bpdn.BPDNJoint usage example"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sporco import util
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
lmbda = 0.0
mu = 45.0
opt = bpdn.BPDN.Options({'Verbose' : True, 'MaxMainIter' : 500,
                         'RelStopTol' : 1e-6, 'AutoRho' : {'RsdlTarget' : 1.0}})

# Initialise and run BPDNJoint object
b = bpdn.BPDNJoint(D, s, lmbda, mu, opt)
b.solve()
print("BPDNJoint solve time: %.2fs" % b.runtime)


# Plot results
fig1 = plt.figure(1, figsize=(6,8))
plt.subplot(1,2,1)
util.imview(x0, fgrf=fig1, cmap=cm.Blues, title='Reference')
plt.subplot(1,2,2)
util.imview(b.Y, fgrf=fig1, cmap=cm.Blues, title='Reconstruction')
fig1.show()


# Plot functional value, residuals, and rho
its = b.getitstat()
fig2 = plt.figure(2, figsize=(21,7))
plt.subplot(1,3,1)
util.plot(its.ObjFun, fgrf=fig2, ptyp='semilogy', xlbl='Iterations',
          ylbl='Functional')
plt.subplot(1,3,2)
util.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T, fgrf=fig2,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual']);
plt.subplot(1,3,3)
util.plot(its.Rho, fgrf=fig2, xlbl='Iterations', ylbl='Penalty Parameter')
fig2.show()


# Wait for enter on keyboard
input()
