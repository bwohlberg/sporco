#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic tvl2.TVL2Denoise usage example"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

from sporco import util
from sporco.admm import tvl2


# Load demo image and construct noisy test image
img = util.ExampleImages().image('lena.grey', scaled=True)
np.random.seed(12345)
imgn = img + np.random.normal(0.0, 0.04, img.shape)

# Set up TVL2Denoise options
lmbda = 0.04
opt = tvl2.TVL2Denoise.Options({'Verbose' : True, 'MaxMainIter' : 200,
                                'gEvalY' : False})

# Initialise and run TVL2Denoise object
b = tvl2.TVL2Denoise(imgn, lmbda, opt)
b.solve()
print("TVL2Denoise solve time: %.2fs" % b.runtime, "\n")


# Display input and result image
fig1 = plt.figure(1, figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(imgn, interpolation="nearest", cmap=plt.get_cmap('gray'))
plt.title('Noisy')
plt.subplot(1,2,2)
plt.imshow(b.X, interpolation="nearest", cmap=plt.get_cmap('gray'))
plt.title('TV Result')
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


# Plot xstep solver stats
fig3 = plt.figure(3, figsize=(16,10))
plt.subplot(1,2,1)
plt.plot([b.itstat[k].GSIter for k in range(0, len(b.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Gauss-Seidel Iterations')
plt.subplot(1,2,2)
plt.semilogy([b.itstat[k].GSRelRes for k in range(0, len(b.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Gauss-Seidel Relative Residual')
fig3.show()

input()
