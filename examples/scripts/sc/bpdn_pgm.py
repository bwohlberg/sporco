#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Basis Pursuit DeNoising
=======================

This example demonstrates the use of class :class:`.pgm.bpdn.BPDN` to solve the Basis Pursuit DeNoising (BPDN) problem :cite:`chen-1998-atomic`

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1 \;,$$

where $D$ is the dictionary, $\mathbf{x}$ is the sparse representation, and $\mathbf{s}$ is the signal to be represented. In this example the BPDN problem is used to estimate the reference sparse representation that generated a signal from a noisy version of the signal.
"""


from __future__ import print_function
from builtins import input

import numpy as np

from sporco.pgm import bpdn
from sporco import plot
from sporco.pgm.backtrack import BacktrackStandard, BacktrackRobust


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
Set regularisation parameter and options for BPDN solver with standard PGM backtracking.
"""

lmbda = 2.98e1
L_sc = 9.e2
opt = bpdn.BPDN.Options({'Verbose': True, 'MaxMainIter': 50,
                         'Backtrack': BacktrackStandard(), 'L': L_sc})


"""
Initialise and run BPDN object
"""

b1 = bpdn.BPDN(D, s, lmbda, opt)
x1 = b1.solve()

print("BPDN standard PGM backtracking solve time: %.2fs" %
      b1.timer.elapsed('solve'))


"""
Set options for BPDN solver with robust PGM backtracking.
"""

opt = bpdn.BPDN.Options({'Verbose': True, 'MaxMainIter': 50, 'L': L_sc,
            'Backtrack': BacktrackRobust()})


"""
Initialise and run BPDN object
"""

b2 = bpdn.BPDN(D, s, lmbda, opt)
x2 = b2.solve()

print("BPDN robust PGM backtracking solve time: %.2fs" %
      b2.timer.elapsed('solve'))



"""
Plot comparison of reference and recovered representations.
"""

plot.plot(np.hstack((x0, x1, x2)), alpha=0.5, title='Sparse representation',
          lgnd=['Reference', 'Reconstructed (Std Backtrack)',
                'Reconstructed (Robust Backtrack)'])


"""
Plot functional value, residual, and L
"""

its1 = b1.getitstat()
its2 = b2.getitstat()
fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.plot(its1.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.plot(its2.ObjFun, xlbl='Iterations', ylbl='Functional',
          lgnd=['Std Backtrack', 'Robust Backtrack'], fig=fig)
plot.subplot(1, 3, 2)
plot.plot(its1.Rsdl, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          fig=fig)
plot.plot(its2.Rsdl, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Std Backtrack', 'Robust Backtrack'], fig=fig)
plot.subplot(1, 3, 3)
plot.plot(its1.L, xlbl='Iterations', ylbl='Inverse of Step Size', fig=fig)
plot.plot(its2.L, xlbl='Iterations', ylbl='Inverse of Step Size',
          lgnd=['Std Backtrack', 'Robust Backtrack'], fig=fig)
fig.show()


# Wait for enter on keyboard
input()
