#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic cmod.CnstrMOD usage example"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

from sporco.admm import bpdn
from sporco.admm import cmod
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


# Load dictionary
D0 = util.convdicts()['G:8x8x64']
D0 = np.reshape(D0, (np.prod(D0.shape[0:2]), D0.shape[2]))


# Compute sparse representation on current dictionary
lmbda = 0.1
opt = bpdn.BPDN.Options({'Verbose' : True,
                         'MaxMainIter' : 200, 'RelStopTol' : 1e-3})
b = bpdn.BPDN(D0, S, lmbda, opt)
b.solve()
print("BPDN solve time: %.2fs" % b.runtime, "\n")


# Update dictionary for training set S
opt = cmod.CnstrMOD.Options({'Verbose' : True,
                             'MaxMainIter' : 500, 'RelStopTol' : 1e-5})
c = cmod.CnstrMOD(b.Y, S, None, opt)
c.solve()
print("CMOD solve time: %.2fs" % c.runtime, "\n")


# Display dictionaries
D0 = D0.reshape(8,8,D0.shape[-1])
D1 = c.Y.reshape((8,8,c.Y.shape[-1]))
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
