#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Basis Pursuit DeNoising
=======================

This example demonstrates the use of class :class:`.fista.bpdn.BPDN` to solve the Basis Pursuit DeNoising (BPDN) problem :cite:`chen-1998-atomic`

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1 \;,$$

where $D$ is the dictionary, $\mathbf{x}$ is the sparse representation, and $\mathbf{s}$ is the signal to be represented. In this example the BPDN problem is used to estimate the reference sparse representation that generated a signal from a noisy version of the signal.
"""


from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco.fista import bpdn
from sporco import util
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


"""
Set BPDN solver with standard FISTA backtracking.
"""

L_sc = 9.e2
opt = bpdn.BPDN.Options({'Verbose': True, 'MaxMainIter': 50,
                    'BackTrack': {'Enabled': True }, 'L': L_sc})


lmbda = 2.98e1

print("")
b = bpdn.BPDN(D, s, lmbda, opt)
x = b.solve()

print("BPDN standard FISTA backtracking solve time: %.2fs" %
      b.timer.elapsed('solve'))
print("")


"""
Plot comparison of reference and recovered representations.
"""

plot.plot(np.hstack((x0, x)), title='Sparse representation (standard FISTA)',
          lgnd=['Reference', 'Reconstructed'])


"""
Plot lmbda error curve, functional value, residuals, and rho
"""

its = b.getitstat()
fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(its.Rsdl, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          fig=fig)
plot.subplot(1, 3, 3)
plot.plot(its.L, xlbl='Iterations',
          ylbl='Inverse of Step Size (Standard FISTA)', fig=fig)
fig.show()


"""
Repeat for BPDN solver with robust FISTA backtracking.
"""

opt = bpdn.BPDN.Options({'Verbose': True, 'MaxMainIter': 50,
            'BackTrack': {'Enabled': True, 'Robust' : True }, 'L': L_sc})


b = bpdn.BPDN(D, s, lmbda, opt)
x = b.solve()

print("BPDN robust FISTA backtracking solve time: %.2fs" %
      b.timer.elapsed('solve'))


"""
Plot comparison of reference and recovered representations.
"""

plot.plot(np.hstack((x0, x)), title='Sparse representation (robust FISTA)',
          lgnd=['Reference', 'Reconstructed'])


"""
Plot lmbda error curve, functional value, residuals, and rho
"""

its = b.getitstat()
fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(its.Rsdl, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          fig=fig)
plot.subplot(1, 3, 3)
plot.plot(its.L, xlbl='Iterations',
          ylbl='Inverse of Step Size (robust FISTA)', fig=fig)
fig.show()



# Wait for enter on keyboard
input()
