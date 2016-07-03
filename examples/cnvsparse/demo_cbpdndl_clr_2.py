#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic cbpdndl.ConvBPDNDictLearn usage example"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

from sporco.admm import cbpdndl
from sporco.admm import ccmod
from sporco import util


# Training images
exim = util.ExampleImages(scaled=True)
img1 = zoom(exim.image('lena'), (0.5, 0.5, 1.0))
img2 = zoom(exim.image('mandrill'), (0.5, 0.5, 1.0))
img3 = zoom(exim.image('barbara')[20:532, 100:612, :], (0.5, 0.5, 1.0))
S = np.concatenate((img1[:,:,:,np.newaxis], img2[:,:,:,np.newaxis],
                    img3[:,:,:,np.newaxis]), axis=3)


# Highpass filter test images
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(S, fltlmbd, npd)


# Initial dictionary
np.random.seed(12345)
D0 = np.random.randn(16, 16, 3, 96)


# Set ConvBPDNDictLearn parameters, including multi-scale dictionary size
lmbda = 0.2
dsz = ((8,8,32), (12,12,32), (16,16,32))
opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose' : True, 'MaxMainIter' : 100,
                                         'DictSize' : dsz,
                    'CBPDN' : {'rho' : 50.0*lmbda + 0.5}})


# Run optimisation
d = cbpdndl.ConvBPDNDictLearn(D0, sh, lmbda, opt)
d.solve()
print("ConvBPDNDictLearn solve time: %.2fs" % d.runtime, "\n")


# Display dictionaries
D1 = ccmod.bcrop(d.ccmod.Y, D0.shape[0:2]+(D0.shape[3],)).squeeze()
fig1 = plt.figure(1, figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(util.tiledict(D0), interpolation="nearest")
plt.title('D0')
plt.subplot(1,2,2)
plt.imshow(util.tiledict(D1, dsz), interpolation="nearest")
plt.title('D1')
fig1.show()


# Plot functional value and residuals
fig2 = plt.figure(2, figsize=(25,10))
plt.subplot(1,3,1)
plt.plot([d.itstat[k].ObjFun for k in range(0, len(d.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Functional')
ax=plt.subplot(1,3,2)
plt.semilogy([d.itstat[k].XPrRsdl for k in range(0, len(d.itstat))])
plt.semilogy([d.itstat[k].XDlRsdl for k in range(0, len(d.itstat))])
plt.semilogy([d.itstat[k].DPrRsdl for k in range(0, len(d.itstat))])
plt.semilogy([d.itstat[k].DDlRsdl for k in range(0, len(d.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Residual')
plt.legend(['X Primal', 'X Dual', 'D Primal', 'D Dual'])
plt.subplot(1,3,3)
plt.semilogy([d.itstat[k].Rho for k in range(0, len(d.itstat))])
plt.semilogy([d.itstat[k].Sigma for k in range(0, len(d.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Penalty Parameter')
plt.legend(['Rho', 'Sigma'])
fig2.show()

input()
