#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Basis Pursuit DeNoising with Joint Sparsity
===========================================

This example demonstrates the use of class :class:`.bpdn.BPDNJoint` to solve the Basis Pursuit DeNoising (BPDN) problem with joint sparsity via an $ℓ_{2,1}$ norm term

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \| D X - S \|_2^2 + \lambda \| X \|_1 + \mu \| X \|_{2,1}$$

where $D$ is the dictionary, $X$ is the sparse representation, and $S$ is the signal to be represented. In this example the BPDN problem is used to estimate the reference sparse representation that generated a signal from a noisy version of the signal.
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

N = 32         # Signal size
M = 4*N        # Dictionary size
L = 12         # Number of non-zero coefficients in generator
K = 16         # Number of signals
sigma = 0.5    # Noise level



"""
Construct random dictionary, reference random sparse representation, and test signal consisting of the synthesis of the reference sparse representation with additive Gaussian noise.
"""

# Construct random dictionary and random sparse coefficients
np.random.seed(12345)
D = np.random.randn(N, M)
x0 = np.zeros((M, K))
si = np.random.permutation(list(range(0, M-1)))
x0[si[0:L],:] = np.random.randn(L, K)
# Construct reference and noisy signal
s0 = D.dot(x0)
s = s0 + sigma*np.random.randn(N,K)



"""
Set BPDNJoint solver class options.
"""

opt = bpdn.BPDNJoint.Options({'Verbose': False, 'MaxMainIter': 500,
                              'RelStopTol': 1e-3, 'rho': 10.0,
                              'AutoRho': {'RsdlTarget': 1.0}})


r"""
Select regularization parameters $\lambda, \mu$ by evaluating the error in recovering the sparse representation over a logarithmicaly spaced grid. (The reference representation is assumed to be known, which is not realistic in a real application.) A function is defined that evalues the BPDN recovery error for a specified $\lambda, \mu$, and this function is evaluated in parallel by :func:`sporco.util.grid_search`.
"""

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



r"""
Once the best $\lambda, \mu$ have been determined, run :meth:`.bpdn.BPDNJoint.solve` with verbose display of ADMM iteration statistics.
"""

# Initialise and run BPDNJoint object for best lmbda and mu
opt['Verbose'] = True
b = bpdn.BPDNJoint(D, s, lmbda, mu, opt)
x = b.solve()

print("BPDNJoint solve time: %.2fs" % b.timer.elapsed('solve'))


"""
Plot comparison of reference and recovered representations.
"""

fig = plot.figure(figsize=(6, 8))
plot.subplot(1, 2, 1)
plot.imview(x0, cmap=plot.cm.Blues, title='Reference', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(x, cmap=plot.cm.Blues, title='Reconstruction', fig=fig)
fig.show()


"""
Plot lmbda, mu error surface, functional value, residuals, and rho
"""

its = b.getitstat()
fig = plot.figure(figsize=(15, 10))
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.locator_params(nbins=5, axis='y')
plot.surf(fvmx, x=np.log10(mrng), y=np.log10(lrng), xlbl=r'log($\mu$)',
          ylbl=r'log($\lambda$)', zlbl='Error', fig=fig, ax=ax)
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
