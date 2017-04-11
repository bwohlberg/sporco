#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: tvl1.TVL1Deconv (denoising problem)"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco import util
from sporco import plot
from sporco.admm import tvl1


# Load reference image
img = util.rgb2gray(util.ExampleImages().image('standard', 'monarch.png',
                                               scaled=True))[:,160:672]


# Construct test image
np.random.seed(12345)
imgn = util.spnoise(img, 0.2)


# Set up TVL1Deconv options
lmbda = 8e-1
opt = tvl1.TVL1Deconv.Options({'Verbose' : True, 'MaxMainIter' : 200,
                               'RelStopTol' : 5e-3, 'gEvalY' : False,
                               'AutoRho' : {'Enabled' : True}})


# Initialise and run TVDeconv object
b = tvl1.TVL1Deconv(np.ones((1,1)), imgn, lmbda, opt)
imgr = b.solve()
print("TVL1Deconv solve time: %.2fs" % b.timer.elapsed('solve'))


# Display test images
fig1 = plot.figure(1, figsize=(21,7))
plot.subplot(1,3,1)
plot.imview(img, fgrf=fig1, title='Reference')
plot.subplot(1,3,2)
plot.imview(imgn, fgrf=fig1, title='Noisy')
plot.subplot(1,3,3)
plot.imview(imgr, fgrf=fig1, title='l1-TV Result')
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
