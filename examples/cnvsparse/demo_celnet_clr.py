#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic cbpdn.ConvElasticNet usage example"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

from sporco import util
from sporco.admm import cbpdn
from sporco.admm import ccmod


# Load demo image
img = zoom(util.ExampleImages().image('lena', scaled=True), (0.5, 0.5, 1.0))


# Highpass filter test image
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(img, fltlmbd, npd)


# Create random dictionary
np.random.seed(12345)
D = ccmod.normalise(np.random.randn(8,8,3,32))


# Set up ConvBPDN options
lmbda = 1e-1
mu = 1e-2
opt = cbpdn.ConvElasticNet.Options({'Verbose' : True, 'MaxMainIter' : 100,
                    'LinSolveCheck' : True, 'RelStopTol' : 1e-3,
                    'AuxVarObj' : False})


# Initialise and run ConvBPDN object
b = cbpdn.ConvElasticNet(D, sh, lmbda, mu, opt)
x = b.solve()
print("ConvElasticNet solve time: %.2fs" % b.runtime, "\n")

# Reconstruct representation
Srec = np.squeeze(b.reconstruct())


# Display representation and reconstructed image
fig1 = plt.figure(1, figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(np.squeeze(np.sum(abs(b.Y), axis=b.axisM)),
            interpolation="nearest")
plt.title('Representation')
plt.subplot(1,2,2)
plt.imshow(Srec + sl, interpolation="nearest")
plt.title('Reconstructed image')
fig1.show()


# Plot functional value, residuals, and rho
fig2 = plt.figure(2, figsize=(25,10))
plt.subplot(1,3,1)
plt.plot([b.itstat[k].ObjFun for k in range(0, len(b.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Functional')
ax=plt.subplot(1,3,2)
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
