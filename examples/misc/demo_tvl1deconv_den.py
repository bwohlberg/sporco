#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Basic tvl1.TVL1Deconv usage example (denoising problem)"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

from sporco import util
from sporco.admm import tvl1


# Load demo image
img = util.ExampleImages().image('lena.grey', scaled=True)
np.random.seed(12345)
imgn = util.spnoise(img, 0.2)

# Set up TVL1Deconv options
lmbda = 8e-1
opt = tvl1.TVL1Deconv.Options({'Verbose' : True, 'MaxMainIter' : 200,
                               'RelStopTol' : 1e-4, 'gEvalY' : False})

# Initialise and run TVDeconv object
b = tvl1.TVL1Deconv(np.ones((1,1)), imgn, lmbda, opt)
b.solve()
print("TVL1Deconv solve time: %.2fs" % b.runtime)


# Display input and result image
fig1 = plt.figure(1, figsize=(14,7))
plt.subplot(1,2,1)
util.imview(imgn, fgrf=fig1, title='Noisy')
plt.subplot(1,2,2)
util.imview(b.X, fgrf=fig1, title='l1-TV Result')
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

