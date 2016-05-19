#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic rpca.RobustPCA usage example"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

from sporco.admm import rpca


N = 256
M = 5
s0u = gaussian_filter(np.random.randn(N, M), 2.0)
s0v = gaussian_filter(np.random.randn(N, M), 2.0)
S0 = np.dot(s0u, s0v.T)

s1 = np.random.uniform(low=0.0, high=1.0, size=S0.shape)
S1 = S0.copy()
S1[s1 > 0.75] = 0.0

opt = rpca.RobustPCA.Options({'Verbose' : True, 'gEvalY' : False,
                              'MaxMainIter' : 200,
                              'AutoRho' : {'Enabled' : True}})
b = rpca.RobustPCA(S1, None, opt)
X, Y = b.solve()
print("RobustPCA solve time: %.2fs" % b.runtime, "\n")

print(np.linalg.norm(S0 - X))


# Display S0 and X image
fig1 = plt.figure(1, figsize=(30,10))
plt.subplot(1,3,1)
plt.imshow(S0, interpolation="nearest", cmap=plt.get_cmap('gray'))
plt.title('Original matrix')
plt.subplot(1,3,2)
plt.imshow(S1, interpolation="nearest", cmap=plt.get_cmap('gray'))
plt.title('Corrupted matrix')
plt.subplot(1,3,3)
plt.imshow(X, interpolation="nearest", cmap=plt.get_cmap('gray'))
plt.title('Low rank component')
fig1.show()


# Plot functional value, residuals, and rho
fig2 = plt.figure(2, figsize=(25,10))
plt.subplot(1,3,1)
plt.plot([b.itstat[k].ObjFun for k in range(0, len(b.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Functional')
plt.subplot(1,3,2)
plt.semilogy([b.itstat[k].PrimalRsdl for k in range(0, len(b.itstat))])
plt.semilogy([b.itstat[k].DualRsdl for k in range(0, len(b.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Residual')
plt.legend(['Primal', 'Dual'])
plt.subplot(1,3,3)
plt.plot([b.itstat[k].Rho for k in range(0, len(b.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Penalty Parameter')
fig2.show()

input()

