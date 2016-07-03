#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic ccmod.ConvCnstrMOD usage example"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

from sporco.admm import cbpdn
from sporco.admm import ccmod
from sporco import util


# Training images
exim = util.ExampleImages(scaled=True)
img1 = zoom(exim.image('lena'), (0.5, 0.5, 1.0))
img2 = zoom(exim.image('mandrill'), (0.5, 0.5, 1.0))
S = np.concatenate((img1[:,:,:,np.newaxis], img2[:,:,:,np.newaxis]), axis=3)


# Highpass filter test images
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(S, fltlmbd, npd)


# Construct initial dictionary
D0 = np.ones((1,1,3,1)) * util.convdicts()['G:12x12x36'][...,np.newaxis,:]
D0 = ccmod.normalise(D0)


# Compute sparse representation on current dictionary
lmbda = 0.01
opt = cbpdn.ConvBPDN.Options({'Verbose' : True, 'MaxMainIter' : 200})
b = cbpdn.ConvBPDN(D0, sh, lmbda, opt)
x = b.solve()


# Update dictionary for training set sh
opt = ccmod.ConvCnstrMOD.Options({'Verbose' : True,
                                  'MaxMainIter' : 100, 'rho' : 5.0})
c = ccmod.ConvCnstrMOD(b.Y, sh, D0.shape[0:2]+(D0.shape[3],), opt)
d = c.solve()
print("CCMOD solve time: %.2fs" % c.runtime, "\n")
D1 = ccmod.bcrop(c.Y, D0.shape).squeeze()



# Display dictionaries
fig1 = plt.figure(1, figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(util.tiledict(D0), interpolation="nearest")
plt.title('D0')
plt.subplot(1,2,2)
plt.imshow(util.tiledict(D1), interpolation="nearest")
plt.title('D1')
fig1.show()


# Plot functional value, residuals, and rho
fig2 = plt.figure(2, figsize=(25,10))
plt.subplot(1,3,1)
plt.plot([c.itstat[k].DFid for k in range(0, len(c.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Functional')
ax=plt.subplot(1,3,2)
plt.semilogy([c.itstat[k].PrimalRsdl for k in range(0, len(c.itstat))])
plt.semilogy([c.itstat[k].DualRsdl for k in range(0, len(c.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Residual')
plt.legend(['Primal', 'Dual'])
plt.subplot(1,3,3)
plt.plot([c.itstat[k].Rho for k in range(0, len(c.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Penalty Parameter')
fig2.show()

input()
