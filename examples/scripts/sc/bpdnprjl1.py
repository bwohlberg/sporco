#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Lasso Optimisation
==================

This example demonstrates the use of class :class:`.bpdn.BPDNProjL1` to solve the least absolute shrinkage and selection operator (lasso) problem :cite:`tibshirani-1996-regression`

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 \; \text{such that} \; \| \mathbf{x} \|_1 \leq \gamma$$

where $D$ is the dictionary, $\mathbf{x}$ is the sparse representation, and $\mathbf{s}$ is the signal to be represented. In this example the lasso problem is used to estimate the reference sparse representation that generated a signal from a noisy version of the signal.
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
Set :class:`.bpdn.BPDNProjL1` solver class options. The value of $\gamma$ has been manually chosen for good performance.
"""

gamma = 2.5e1
opt = bpdn.BPDNProjL1.Options({'Verbose': True, 'MaxMainIter': 500,
                    'RelStopTol': 1e-6, 'AutoRho': {'RsdlTarget': 1.0}})


"""
Initialise and run BPDNProjL1 object
"""

b = bpdn.BPDNProjL1(D, s, gamma, opt)
x = b.solve()

print("BPDNProjL1 solve time: %.2fs" % b.timer.elapsed('solve'))


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
