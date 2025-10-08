#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Minimise ℓ1 Norm with ℓ2 Constraint
====================================

This example demonstrates the use of class :class:`.admm.bpdn.MinL1InL2Ball` to solve the problem

  $$\mathrm{argmin}_\mathbf{x} \| \mathbf{x} \|_1 \; \text{such that} \; \| D \mathbf{x} - \mathbf{s} \|_2 \leq \epsilon$$

where $D$ is the dictionary, $\mathbf{x}$ is the sparse representation, and $\mathbf{s}$ is the signal to be represented. In this example this problem is used to estimate the reference sparse representation that generated a signal from a noisy version of the signal.
"""


from __future__ import print_function
from builtins import input

import numpy as np

from sporco.admm import bpdn
from sporco import plot


"""
Configure problem size, sparsity, and noise level.
"""

N = 512      # Signal size
M = 4*N      # Dictionary size
L = 32       # Number of non-zero coefficients in generator
sigma = 0.5  # Noise level


"""
Construct random dictionary, reference random sparse representation, and test signal consisting of the synthesis of the reference sparse representation with additive Gaussian noise.
"""

# Construct random dictionary and random sparse coefficients
np.random.seed(12345)
D = np.random.randn(N, M)
x0 = np.zeros((M, 1))
si = np.random.permutation(list(range(0, M-1)))
x0[si[0:L]] = np.random.randn(L, 1)

# Construct reference and noisy signal
s0 = D.dot(x0)
s = s0 + sigma*np.random.randn(N,1)


r"""
Set :class:`.admm.bpdn.MinL1InL2Ball` solver class options. The value of $\epsilon$ is estimated from the difference between the noisy and reference signals.
"""

epsilon = 1.1 * np.linalg.norm(s0 - s)
opt = bpdn.MinL1InL2Ball.Options({'Verbose': True, 'MaxMainIter': 500,
                                  'RelStopTol': 1e-3, 'rho': 1e0,
                                  'AutoRho': {'Enabled': False}})


"""
Initialise and run :class:`.admm.bpdn.MinL1InL2Ball` object
"""

b = bpdn.MinL1InL2Ball(D, s, epsilon, opt)
x = b.solve()

print("MinL1InL2Ball solve time: %.2fs" % b.timer.elapsed('solve'))


"""
Plot comparison of reference and recovered representations.
"""

plot.plot(np.hstack((x0, x)), title='Sparse representation',
          lgnd=['Reference', 'Reconstructed'])


"""
Plot functional value, residuals, and rho
"""

its = b.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'], fig=fig)
plot.subplot(1, 3, 3)
plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter', fig=fig)
fig.show()


# Wait for enter on keyboard
input()
