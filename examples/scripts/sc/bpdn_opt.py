#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Basis Pursuit DeNoising
=======================

This example demonstrates the use of class :class:`.admm.bpdn.BPDN` to solve the Basis Pursuit DeNoising (BPDN) problem :cite:`chen-1998-atomic`

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1 \;,$$

where $D$ is the dictionary, $\mathbf{x}$ is the sparse representation, and $\mathbf{s}$ is the signal to be represented. In this example the BPDN problem is used to estimate the reference sparse representation that generated a signal from a noisy version of the signal.
"""


from __future__ import print_function
from builtins import input

import numpy as np

from sporco.admm import bpdn
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
Set BPDN solver class options.
"""

opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 500,
                         'RelStopTol': 1e-3, 'AutoRho': {'RsdlTarget': 1.0}})


r"""
Select regularization parameter $\lambda$ by evaluating the error in recovering the sparse representation over a logarithmicaly spaced grid. (The reference representation is assumed to be known, which is not realistic in a real application.) A function is defined that evalues the BPDN recovery error for a specified $\lambda$, and this function is evaluated in parallel by :func:`sporco.util.grid_search`.
"""

# Function computing reconstruction error at lmbda
def evalerr(prm):
    lmbda = prm[0]
    b = bpdn.BPDN(D, s, lmbda, opt)
    x = b.solve()
    return np.sum(np.abs(x-x0))


# Parallel evalution of error function on lmbda grid
lrng = np.logspace(1, 2, 20)
sprm, sfvl, fvmx, sidx = util.grid_search(evalerr, (lrng,))
lmbda = sprm[0]

print('Minimum ℓ1 error: %5.2f at 𝜆 = %.2e' % (sfvl, lmbda))


r"""
Once the best $\lambda$ has been determined, run BPDN with verbose display of ADMM iteration statistics.
"""

# Initialise and run BPDN object for best lmbda
opt['Verbose'] = True
b = bpdn.BPDN(D, s, lmbda, opt)
x = b.solve()

print("BPDN solve time: %.2fs" % b.timer.elapsed('solve'))


"""
Plot comparison of reference and recovered representations.
"""

plot.plot(np.hstack((x0, x)), title='Sparse representation',
          lgnd=['Reference', 'Reconstructed'])


"""
Plot lmbda error curve, functional value, residuals, and rho
"""

its = b.getitstat()
fig = plot.figure(figsize=(15, 10))
plot.subplot(2, 2, 1)
plot.plot(fvmx, x=lrng, ptyp='semilogx', xlbl=r'$\lambda$',
          ylbl='Error', fig=fig)
plot.subplot(2, 2, 2)
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(2, 2, 3)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'], fig=fig)
plot.subplot(2, 2, 4)
plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter', fig=fig)
fig.show()



# Wait for enter on keyboard
input()
