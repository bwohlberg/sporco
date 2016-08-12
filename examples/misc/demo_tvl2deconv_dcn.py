#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic tvl2.TVL2Deconv usage example (deconvolution problem)"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

from sporco import util
from sporco.admm import tvl2


# Load demo image and set up blurred and noisy test image
img = util.ExampleImages().image('lena.grey', scaled=True)
h0 = np.zeros((11,11), dtype=np.float32)
h0[5,5] = 1.0
h = ndimage.filters.gaussian_filter(h0, 2.0)
imgc = np.real(np.fft.ifft2(np.fft.fft2(h, img.shape) * np.fft.fft2(img)))
imgcn = imgc + np.random.normal(0.0, 0.01, img.shape)

# Set up TVDeconv options
lmbda = 5e-3
opt = tvl2.TVL2Deconv.Options({'Verbose' : True, 'MaxMainIter' : 200,
                               'gEvalY' : False})

# Initialise and run TVL2Deconv object
b = tvl2.TVL2Deconv(h, imgcn, lmbda, opt)
b.solve()
print("TVL2Deconv solve time: %.2fs" % b.runtime)


# Display input and result image
fig1 = plt.figure(1, figsize=(14,7))
plt.subplot(1,2,1)
util.imview(imgcn, fgrf=fig1, title='Blurred/Noisy')
plt.subplot(1,2,2)
util.imview(b.X, fgrf=fig1, title='l2-TV Result')
fig1.show()


# Plot functional value, residuals, and rho
its = b.getitstat()
fig2 = plt.figure(2, figsize=(21,7))
plt.subplot(1,3,1)
util.plot(its.ObjFun, fgrf=fig2, xlbl='Iterations', ylbl='Functional')
plt.subplot(1,3,2)
util.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T, fgrf=fig2,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual']);
plt.subplot(1,3,3)
util.plot(its.Rho, fgrf=fig2, xlbl='Iterations', ylbl='Penalty Parameter')
fig2.show()


# Wait for enter on keyboard
input()

