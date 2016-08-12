#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic dictlrn.DictLearn usage example"""

from __future__ import division
from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

from sporco.admm import bpdn
from sporco.admm import cmod
from sporco.admm import dictlrn
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

# X and D update options
lmbda = 0.1
optx = bpdn.BPDN.Options({'Verbose' : False, 'MaxMainIter' : 1,
                          'rho' : 50.0*lmbda + 0.5})
optd = cmod.CnstrMOD.Options({'Verbose' : False, 'MaxMainIter' : 1,
                              'rho' : S.shape[1] / 200.0})

# Normalise dictionary according to D update options
D0 = cmod.getPcn(optd['ZeroMean'])(D0)

# Update D update options to include initial values for Y and U
optd.update({'Y0' : D0, 'U0' : np.zeros((S.shape[0], D0.shape[1]))})

# Create X update object
xstep = bpdn.BPDN(D0, S, lmbda, optx)

# Create D update object
dstep = cmod.CnstrMOD(None, S, (D0.shape[1], S.shape[1]), optd)

# Create DictLearn object
opt = dictlrn.DictLearn.Options({'Verbose' : True, 'MaxMainIter' : 100})
d = dictlrn.DictLearn(xstep, dstep, opt)
Dmx = d.solve()
print("DictLearn solve time: %.2fs" % d.runtime)


# Display dictionaries
D1 = Dmx.reshape((8,8,D0.shape[1]))
D0 = D0.reshape(8,8,D0.shape[-1])
fig1 = plt.figure(1, figsize=(14,7))
plt.subplot(1,2,1)
util.imview(util.tiledict(D0), fgrf=fig1, title='D0')
plt.subplot(1,2,2)
util.imview(util.tiledict(D1), fgrf=fig1, title='D1')
fig1.show()


# Plot functional value and residuals
itsx = xstep.getitstat()
itsd = dstep.getitstat()
fig2 = plt.figure(2, figsize=(21,7))
plt.subplot(1,3,1)
util.plot(itsx.ObjFun, fgrf=fig2, xlbl='Iterations', ylbl='Functional')
plt.subplot(1,3,2)
util.plot(np.vstack((itsx.PrimalRsdl, itsx.DualRsdl, itsd.PrimalRsdl,
                     itsd.DualRsdl)).T,
          fgrf=fig2, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['X Primal', 'X Dual', 'D Primal', 'D Dual']);
plt.subplot(1,3,3)
util.plot(np.vstack((itsx.Rho, itsd.Rho)).T, fgrf=fig2, xlbl='Iterations',
          ylbl='Penalty Parameter', ptyp='semilogy', lgnd=['Rho', 'Sigma'])
fig2.show()


# Wait for enter on keyboard
input()
