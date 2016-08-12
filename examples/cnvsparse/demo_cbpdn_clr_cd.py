#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic cbpdn.ConvBPDN usage example (colour images, colour dictionary)"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sporco import util
from sporco.admm import cbpdn
import sporco.linalg as spl


# Load demo image
img = zoom(util.ExampleImages().image('lena', scaled=True), (0.5, 0.5, 1.0))


# Highpass filter test image
npd = 16
fltlmbd = 10
sl, sh = util.tikhonov_filter(img, fltlmbd, npd)


# Load dictionary
D = util.convdicts()['RGB:8x8x3x64']


# Set up ConvBPDN options
lmbda = 1e-2
opt = cbpdn.ConvBPDN.Options({'Verbose' : True, 'MaxMainIter' : 200,
                    'LinSolveCheck' : True, 'RelStopTol' : 1e-3,
                    'AuxVarObj' : False})


# Initialise and run ConvBPDN object
b = cbpdn.ConvBPDN(D, sh, lmbda, opt)
X = b.solve()
print("ConvBPDN solve time: %.2fs" % b.runtime)


# Reconstruct representation
shr = b.reconstruct().squeeze()
imgr = sl + shr
print("reconstruction PSNR: %.2fdB\n" % spl.psnr(img, imgr))


# Display representation and reconstructed image
fig1 = plt.figure(1, figsize=(21,7))
plt.subplot(1,3,1)
util.imview(np.sum(abs(X), axis=b.axisM).squeeze(), fgrf=fig1, cmap=cm.Blues,
            title='Representation')
plt.subplot(1,3,2)
util.imview(imgr, fgrf=fig1, title='Reconstructed image')
plt.subplot(1,3,3)
util.imview(imgr - img, fgrf=fig1, title='Reconstruction difference')
fig1.show()


# Plot functional value, residuals, and rho
its = b.getitstat()
fig2 = plt.figure(2, figsize=(21,7))
plt.subplot(1,3,1)
util.plot(its.ObjFun, fgrf=fig2, ptyp='semilogy', xlbl='Iterations',
          ylbl='Functional')
plt.subplot(1,3,2)
util.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T, fgrf=fig2,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual']);
plt.subplot(1,3,3)
util.plot(its.Rho, fgrf=fig2, xlbl='Iterations', ylbl='Penalty Parameter')
fig2.show()


# Wait for enter on keyboard
input()
