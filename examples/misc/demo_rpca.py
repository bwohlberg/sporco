#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: rpca.RobustPCA"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from sporco.admm import rpca
from sporco import plot


N = 256
M = 5
s0u = gaussian_filter(np.random.randn(N, M), 2.0)
s0v = gaussian_filter(np.random.randn(N, M), 2.0)
S0 = np.dot(s0u, s0v.T)

s1 = np.random.uniform(low=0.0, high=1.0, size=S0.shape)
S1 = S0.copy()
S1[s1 > 0.75] = 0.0

opt = rpca.RobustPCA.Options({'Verbose' : True, 'gEvalY' : False,
                              'MaxMainIter' : 200, 'RelStopTol' : 5e-4,
                              'AutoRho' : {'Enabled' : True}})
b = rpca.RobustPCA(S1, None, opt)
X, Y = b.solve()
print("RobustPCA solve time: %.2fs" % b.timer.elapsed('solve'))
print(" low rank error (l2): %.2e" % np.linalg.norm(S0 - X))


# Display S0 and X image
fig1 = plot.figure(1, figsize=(21,7))
plot.subplot(1,3,1)
plot.imview(S0, fgrf=fig1, title='Original matrix')
plot.subplot(1,3,2)
plot.imview(S1, fgrf=fig1, title='Corrupted matrix')
plot.subplot(1,3,3)
plot.imview(X, fgrf=fig1, title='Low rank component')
fig1.show()


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

