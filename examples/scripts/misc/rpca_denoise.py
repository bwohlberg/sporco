#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Robust PCA
==========

This example demonstrates the use of class :class:`.rpca.RobustPCA` for solving a synthetic Robust PCA problem. A reference low-rank matrix is corrupted with a sparse set of outliers, and then recovered as the low-rank component of the Robust PCA decomposition into low-rank and sparse components.
"""


from __future__ import print_function
from builtins import input

import numpy as np
from scipy.ndimage import gaussian_filter

from sporco.admm import rpca
from sporco import metric
from sporco import plot


"""
Generate example low-rank matrix. The smoothing using `gaussian_filter` is purely for visualization purposes: the Robust PCA recovery of the low-rank component works just as well without it, but the difference between corrupted and recovered matrices is less obvious without the smoothing.
"""

N = 256
M = 5
np.random.seed(12345)
s0u = gaussian_filter(np.random.randn(N, M), 2.0)
s0v = gaussian_filter(np.random.randn(N, M), 2.0)
S0 = np.dot(s0u, s0v.T)


"""
Corrupt approximately 25% of the samples of the the low-rank matrix.
"""

s1gen = np.random.uniform(low=0.0, high=1.0, size=S0.shape)
S1 = S0.copy()
S1[s1gen > 0.75] = 0.0


"""
Set options for the Robust PCA solver, create the solver object, and solve, returning the estimates of the low rank and sparse components ``X`` and ``Y``. Unlike most other SPORCO classes for optimisation problems, :class:`.rpca.RobustPCA` has a meaningful default regularization parameter, as used here.
"""

opt = rpca.RobustPCA.Options({'Verbose': True, 'gEvalY': False,
                              'MaxMainIter': 200, 'RelStopTol': 5e-4,
                              'AutoRho': {'Enabled': True}})
b = rpca.RobustPCA(S1, None, opt)
X, Y = b.solve()


"""
Display solve time and low-rank component recovery accuracy.
"""

print("RobustPCA solve time:   %5.2f s" % b.timer.elapsed('solve'))
print("Low-rank component SNR: %5.2f dB" % metric.snr(S0, X))


"""
Display reference, corrupted, and recovered matrices.
"""

fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.imview(S0, cmap=plot.cm.Blues, title='Original matrix', fig=fig)
plot.subplot(1, 3, 2)
plot.imview(S1, cmap=plot.cm.Blues, title='Corrupted matrix', fig=fig)
plot.subplot(1, 3, 3)
plot.imview(X, cmap=plot.cm.Blues, title='Recovered matrix', fig=fig)
fig.show()


"""
Get iterations statistics from solver object and plot functional value, ADMM primary and dual residuals, and automatically adjusted ADMM penalty parameter against the iteration number.
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
