#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: ccmod.ConvCnstrMOD (greyscale images)"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco.admm import cbpdn
from sporco.admm import ccmod
from sporco import util
from sporco import plot


# Training images (size reduced to speed up demo script)
exim = util.ExampleImages(scaled=True, zoom=0.5)
S1 = exim.image('standard', 'lena.grey.png')
S2 = exim.image('standard', 'barbara.grey.png')
S3 = util.rgb2gray(exim.image('standard', 'monarch.png'))[:,80:336]
S4 = util.rgb2gray(exim.image('standard', 'mandrill.png'))
S5 = exim.image('standard', 'man.grey.png')[50:306, 50:306]
S = np.dstack((S1,S2,S3,S4,S5))


# Highpass filter test images
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(S, fltlmbd, npd)


# Load initial dictionary
D0 = util.convdicts()['G:12x12x36']


# Compute sparse representation on current dictionary
lmbda = 0.05
opt = cbpdn.ConvBPDN.Options({'Verbose' : True, 'MaxMainIter' : 100,
                     'HighMemSolve' : True})
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
