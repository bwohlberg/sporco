#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Basis Pursuit DeNoising
=======================

This example demonstrates the use of classes :class:`.admm.bpdn.BPDN` and :class:`.pgm.bpdn.BPDN` to solve the Basis Pursuit DeNoising (BPDN) problem :cite:`chen-1998-atomic`

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1 \;,$$

where $D$ is the dictionary, $\mathbf{x}$ is the sparse representation, and $\mathbf{s}$ is the signal to be represented. In this example the BPDN problem is used to estimate the reference sparse representation that generated a signal from a noisy version of the signal.
"""


from __future__ import print_function
from builtins import input

import numpy as np

import sporco.admm.bpdn as abpdn
import sporco.pgm.bpdn as pbpdn
from sporco.pgm.backtrack import BacktrackRobust
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
Set regularisation parameter.
"""

lmbda = 2.98e1


"""
Set options for ADMM solver.
"""

opt_admm = abpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 500,
                        'RelStopTol': 1e-3, 'AutoRho': {'RsdlTarget': 1.0}})


"""
Initialise and run ADMM solver object.
"""

ba = abpdn.BPDN(D, s, lmbda, opt_admm)
xa = ba.solve()

print("ADMM BPDN solve time: %.2fs" % ba.timer.elapsed('solve'))



"""
Set options for PGM solver.
"""

opt_pgm = pbpdn.BPDN.Options({'Verbose': True, 'MaxMainIter': 50, 'L': 9e2,
                              'Backtrack': BacktrackRobust()})


"""
Initialise and run PGM solver.
"""

bp = pbpdn.BPDN(D, s, lmbda, opt_pgm)
xp = bp.solve()

print("PGM BPDN solve time: %.2fs" % bp.timer.elapsed('solve'))


"""
Plot comparison of reference and recovered representations.
"""

plot.plot(np.hstack((x0, xa, xp)), alpha=0.5, title='Sparse representation',
          lgnd=['Reference', 'Reconstructed (ADMM)',
                'Reconstructed (PGM)'])


"""
Plot functional value, residual, and L
"""

itsa = ba.getitstat()
itsp = bp.getitstat()
fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.plot(itsa.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.plot(itsp.ObjFun, xlbl='Iterations', ylbl='Functional',
          lgnd=['ADMM', 'PGM'], fig=fig)
plot.subplot(1, 3, 2)
plot.plot(itsa.PrimalRsdl, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          fig=fig)
plot.plot(itsa.DualRsdl, ptyp='semilogy', fig=fig)
plot.plot(itsp.Rsdl, ptyp='semilogy', lgnd=['Primal Residual (ADMM)',
          'Dual Residual (ADMM)','Residual (PGM)'], fig=fig)
plot.subplot(1, 3, 3)
plot.plot(itsa.Rho, xlbl='Iterations', ylbl='Algorithm Parameter', fig=fig)
plot.plot(itsp.L, lgnd=[r'$\rho$ (ADMM)', '$L$ (PGM)'], fig=fig)
fig.show()


# Wait for enter on keyboard
input()
