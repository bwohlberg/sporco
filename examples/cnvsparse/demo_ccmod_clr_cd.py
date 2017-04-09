#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: ccmod.ConvCnstrMOD (colour images, colour dictionary)"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco.admm import cbpdn
from sporco.admm import ccmod
from sporco import util
from sporco import plot


# Training images
exim = util.ExampleImages(scaled=True, zoom=0.5)
img1 = exim.image('standard', 'lena.png')
img2 = exim.image('standard', 'mandrill.png')
S = np.concatenate((img1[...,np.newaxis], img2[...,np.newaxis]), axis=3)


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
b.solve()


# Update dictionary for training set sh
opt = ccmod.ConvCnstrMOD.Options({'Verbose' : True, 'MaxMainIter' : 100,
                                  'rho' : 5.0})
c = ccmod.ConvCnstrMOD(b.Y, sh, D0.shape, opt)
c.solve()
print("ConvCnstrMOD solve time: %.2fs" % c.timer.elapsed('solve'))
D1 = c.getdict().squeeze()


# Display dictionaries
fig1 = plot.figure(1, figsize=(14,7))
plot.subplot(1,2,1)
plot.imview(util.tiledict(D0), fgrf=fig1, title='D0')
plot.subplot(1,2,2)
plot.imview(util.tiledict(D1), fgrf=fig1, title='D1')
fig1.show()


# Plot functional value, residuals, and rho
its = c.getitstat()
fig2 = plot.figure(2, figsize=(21,7))
plot.subplot(1,3,1)
plot.plot(its.DFid, fgrf=fig2, ptyp='semilogy', xlbl='Iterations',
          ylbl='Functional')
plot.subplot(1,3,2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T, fgrf=fig2,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'])
plot.subplot(1,3,3)
plot.plot(its.Rho, fgrf=fig2, xlbl='Iterations', ylbl='Penalty Parameter')
fig2.show()


# Wait for enter on keyboard
input()
