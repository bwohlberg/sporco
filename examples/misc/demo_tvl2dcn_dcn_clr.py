#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: tvl2.TVL2Deconv (deconvolution problem, colour image)"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
from scipy import ndimage

from sporco import util
from sporco import plot
from sporco.admm import tvl2


# Utility functions
n = 13
n2 = n // 2
spad = lambda x: np.pad(x, ((n,n),(n,n),(0,0)), mode='symmetric')
crop = lambda x: x[n+n2:-n+n2,n+n2:-n+n2,:]
conv = lambda h, x: np.fft.ifft2(np.fft.fft2(h, s=x.shape[0:2], axes=(0,1))
                [...,np.newaxis]*np.fft.fft2(x, axes=(0,1)), axes=(0,1)).real


# Load reference image
img = util.ExampleImages().image('standard', 'monarch.png',
                                 scaled=True)[:,160:672]


# Construct smoothing filter
h0 = np.zeros((13,13), dtype=np.float32)
h0[6,6] = 1.0
h = ndimage.filters.gaussian_filter(h0, 2.0)


# Construct test image
imgc = crop(conv(h, spad(img)))
np.random.seed(12345)
imgcn = imgc + np.random.normal(0.0, 0.02, img.shape)


# Set up TVDeconv options
lmbda = 5e-3
opt = tvl2.TVL2Deconv.Options({'Verbose' : True, 'MaxMainIter' : 200,
                               'gEvalY' : False, 'RelStopTol' : 5e-3})


# Initialise and run TVL2Deconv object
b = tvl2.TVL2Deconv(h, spad(imgcn), lmbda, opt, caxis=2)
imgr = b.solve()[n-n2:-n-n2,n-n2:-n-n2,:]
print("TVL2Deconv solve time: %.2fs" % b.timer.elapsed('solve'))


# Display test images
fig1 = plot.figure(1, figsize=(21,7))
plot.subplot(1,3,1)
plot.imview(img, fgrf=fig1, title='Reference')
plot.subplot(1,3,2)
plot.imview(imgcn, fgrf=fig1, title='Blurred/Noisy')
plot.subplot(1,3,3)
plot.imview(imgr, fgrf=fig1, title='l2-TV Result')
fig1.show()


# Plot functional value, residuals, and rho
its = b.getitstat()
fig2 = plot.figure(2, figsize=(21,7))
plot.subplot(1,3,1)
plot.plot(its.ObjFun, fgrf=fig2, xlbl='Iterations', ylbl='Functional')
plot.subplot(1,3,2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T, fgrf=fig2,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'])
plot.subplot(1,3,3)
plot.plot(its.Rho, fgrf=fig2, xlbl='Iterations', ylbl='Penalty Parameter')
fig2.show()


# Wait for enter on keyboard
input()
