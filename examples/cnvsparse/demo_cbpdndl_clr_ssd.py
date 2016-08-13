#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic cbpdndl.ConvBPDNDictLearn usage example (colour images,
single-scale colour dictionary).
"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt

from sporco.admm import cbpdndl
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
D0 = np.random.randn(8, 8, 3, 64)


# Set ConvBPDNDictLearn parameters
lmbda = 0.2
opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose' : True, 'MaxMainIter' : 100,
                                         'CBPDN' : {'rho' : 50.0*lmbda + 0.5},
                                         'CCMOD' : {'LinSolve' : 'SM',
                                                    'ZeroMean': True}})


# Run optimisation
d = cbpdndl.ConvBPDNDictLearn(D0, sh, lmbda, opt)
D1 = d.solve()
print("ConvBPDNDictLearn solve time: %.2fs" % d.runtime, "\n")


# Display dictionaries
D1 = D1.squeeze()
fig1 = plt.figure(1, figsize=(14,7))
plt.subplot(1,2,1)
util.imview(util.tiledict(D0), fgrf=fig1, title='D0')
plt.subplot(1,2,2)
util.imview(util.tiledict(D1), fgrf=fig1, title='D1')
fig1.show()


# Plot functional value and residuals
its = d.getitstat()
fig2 = plt.figure(2, figsize=(21,7))
plt.subplot(1,3,1)
util.plot(its.ObjFun, fgrf=fig2, xlbl='Iterations', ylbl='Functional')
plt.subplot(1,3,2)
util.plot(np.vstack((its.XPrRsdl, its.XDlRsdl, its.DPrRsdl, its.DDlRsdl)).T,
          fgrf=fig2, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['X Primal', 'X Dual', 'D Primal', 'D Dual']);
plt.subplot(1,3,3)
util.plot(np.vstack((its.XRho, its.DRho)).T, fgrf=fig2, xlbl='Iterations',
          ylbl='Penalty Parameter', ptyp='semilogy',
          lgnd=['$\\rho_X$', '$\\rho_D$'])
fig2.show()


# Wait for enter on keyboard
input()
