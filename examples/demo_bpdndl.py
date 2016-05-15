#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic bpdndl.BPDNDictLearn usage example"""

from __future__ import division
from __future__ import print_function
from builtins import input
from builtins import range
from past.utils import old_div

import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

from sporco.admm import bpdndl
from sporco import util


# Training images
exim = util.ExampleImages(scaled=True)
img1 = exim.image('lena.grey')
img2 = exim.image('barbara.grey')
img3 = exim.image('kiel.grey')
img4 = util.rgb2gray(exim.image('mandrill'))
img5 = exim.image('man.grey')[100:612, 100:612]


# Reduce images size to speed up demo script
S1 = zoom(img1, 0.5)
S2 = zoom(img2, 0.5)
S3 = zoom(img3, 0.5)
S4 = zoom(img4, 0.5)
S5 = zoom(img5, 0.5)


# Extract all 8x8 image blocks, reshape, and subtract block means
S = util.imageblocks((S1,S2,S3,S4,S5), (8,8))
S = np.reshape(S, (np.prod(S.shape[0:2]), S.shape[2]))
S -= np.mean(S, axis=0)


# Initial dictionary
np.random.seed(12345)
D0 = np.random.randn(S.shape[0], 128)


# Set BPDNDictLearn parameters
lmbda = 0.1
opt = bpdndl.BPDNDictLearn.Options({'Verbose' : True, 'MaxMainIter' : 1000,
                      'BPDN' : {'rho' : 50.0*lmbda + 0.5},
                      'CMOD' : {'rho' : old_div(S.shape[1],200.0)}})

# Run optimisation
d = bpdndl.BPDNDictLearn(D0, S, lmbda, opt)
d.solve()
print("BPDNDictLearn solve time: %.2fs" % d.runtime, "\n")


# Display dictionaries
D1 = d.cmod.Y.reshape((8,8,D0.shape[1]))
D0 = D0.reshape(8,8,D0.shape[-1])
fig1 = plt.figure(1, figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(util.tiledict(D0), interpolation="nearest",
           cmap=plt.get_cmap('gray'))
plt.title('D0')
plt.subplot(1,2,2)
plt.imshow(util.tiledict(D1), interpolation="nearest",
           cmap=plt.get_cmap('gray'))
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
