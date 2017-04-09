#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: cbpdn.ConvBPDNGradReg (greyscale images)"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco import util
from sporco import plot
from sporco.admm import cbpdn
import sporco.metric as sm


# Load demo image
img = util.ExampleImages().image('standard', 'barbara.grey.png', scaled=True)


# Load dictionary
Db = util.convdicts()['G:12x12x36']
# Append impulse filter for lowpass component representation
di = np.zeros(Db.shape[0:2] + (1,))
di[0,0] = 1
D = np.concatenate((di, Db), axis=2)
# Weights for l1 norm: no regularization on impulse filter
wl1 = np.ones((1,)*4 + (D.shape[2:]))
wl1[...,0] = 0.0
# Weights for l2 norm of gradient: regularization only on impulse filter
wgr = np.zeros((D.shape[2]))
wgr[0] = 1.0


# Set up ConvBPDNGradReg options
lmbda = 1e-2
mu = 1e1
opt = cbpdn.ConvBPDNGradReg.Options({'Verbose' : True, 'MaxMainIter' : 200,
                    'HighMemSolve' : True, 'LinSolveCheck' : True,
                    'RelStopTol' : 1e-2, 'AuxVarObj' : False,
                    'rho' : 1e0, 'AutoRho' : {'Enabled' : True,
                    'Period' : 10, 'RsdlTarget' : 0.04},
                    'L1Weight' : wl1, 'GradWeight' : wgr})

# Initialise and run ConvBPDNGradReg object
b = cbpdn.ConvBPDNGradReg(D, img, lmbda, mu, opt)
X = b.solve()
print("ConvBPDNGradReg solve time: %.2fs" % b.timer.elapsed('solve'))

# Reconstruct representation
imgr = b.reconstruct().squeeze()
print("       reconstruction PSNR: %.2fdB\n" % sm.psnr(img, imgr))


# Display representation and reconstructed image
fig1 = plot.figure(1, figsize=(14,14))
plot.subplot(2,2,1)
plot.imview(b.Y[...,0].squeeze(), fgrf=fig1, cmap=plot.cm.Blues,
            title='Lowpass component')
plot.subplot(2,2,2)
plot.imview(np.sum(abs(b.Y[...,1:]), axis=b.cri.axisM).squeeze(), fgrf=fig1,
            cmap=plot.cm.Blues, title='Main representation')
plot.subplot(2,2,3)
plot.imview(imgr, fgrf=fig1, title='Reconstructed image')
plot.subplot(2,2,4)
plot.imview(imgr - img, fgrf=fig1, title='Reconstruction difference')
fig1.show()


# Plot functional value, residuals, and rho
its = b.getitstat()
fig2 = plot.figure(2, figsize=(21,7))
plot.subplot(1,3,1)
plot.plot(its.ObjFun, fgrf=fig2, ptyp='semilogy', xlbl='Iterations',
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
