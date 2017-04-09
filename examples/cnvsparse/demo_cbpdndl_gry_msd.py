#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: cbpdndl.ConvBPDNDictLearn (greyscale images,
multi-scale dictionary).
"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco.admm import cbpdndl
from sporco import util
from sporco import plot


# Training images
exim = util.ExampleImages(scaled=True, zoom=0.25)
S1 = exim.image('standard', 'lena.grey.png')
S2 = exim.image('standard', 'barbara.grey.png')
S3 = util.rgb2gray(exim.image('standard', 'monarch.png',
                                idxexp=np.s_[:,160:672]))
S4 = util.rgb2gray(exim.image('standard', 'mandrill.png'))
S5 = exim.image('standard', 'man.grey.png', idxexp=np.s_[100:612, 100:612])
S = np.dstack((S1,S2,S3,S4,S5))


# Highpass filter test images
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(S, fltlmbd, npd)


# Initial dictionary
np.random.seed(12345)
D0 = np.random.randn(10, 10, 48)


# Set ConvBPDNDictLearn parameters, including multi-scale dictionary size
lmbda = 0.2
dsz = ((6,6,16), (8,8,16), (10,10,16))
opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose' : True, 'MaxMainIter' : 200,
                                         'DictSize' : dsz,
                                    'CBPDN' : {'rho' : 50.0*lmbda + 0.5},
                                    'CCMOD' : {'rho' : 10, 'ZeroMean' : True}})


# Run optimisation
d = cbpdndl.ConvBPDNDictLearn(D0, sh, lmbda, opt)
D1 = d.solve()
print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))


# Display dictionaries
D1 = D1.squeeze()
fig1 = plot.figure(1, figsize=(14,7))
plot.subplot(1,2,1)
plot.imview(util.tiledict(D0), fgrf=fig1, title='D0')
plot.subplot(1,2,2)
plot.imview(util.tiledict(D1, dsz), fgrf=fig1, title='D1')
fig1.show()


# Plot functional value and residuals
its = d.getitstat()
fig2 = plot.figure(2, figsize=(21,7))
plot.subplot(1,3,1)
plot.plot(its.ObjFun, fgrf=fig2, xlbl='Iterations', ylbl='Functional')
plot.subplot(1,3,2)
plot.plot(np.vstack((its.XPrRsdl, its.XDlRsdl, its.DPrRsdl, its.DDlRsdl)).T,
          fgrf=fig2, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['X Primal', 'X Dual', 'D Primal', 'D Dual'])
plot.subplot(1,3,3)
plot.plot(np.vstack((its.XRho, its.DRho)).T, fgrf=fig2, xlbl='Iterations',
          ylbl='Penalty Parameter', ptyp='semilogy',
          lgnd=['$\\rho_X$', '$\\rho_D$'])
fig2.show()


# Wait for enter on keyboard
input()
