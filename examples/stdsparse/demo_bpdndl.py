#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: bpdndl.BPDNDictLearn"""

from __future__ import division
from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco.admm import bpdndl
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


# Extract all 8x8 image blocks, reshape, and subtract block means
S = util.imageblocks((S1,S2,S3,S4,S5), (8,8))
S = np.reshape(S, (np.prod(S.shape[0:2]), S.shape[2]))
S -= np.mean(S, axis=0)


# Initial dictionary
np.random.seed(12345)
D0 = np.random.randn(S.shape[0], 128)


# Set BPDNDictLearn parameters
lmbda = 0.1
opt = bpdndl.BPDNDictLearn.Options({'Verbose' : True, 'MaxMainIter' : 100,
                      'BPDN' : {'rho' : 50.0*lmbda + 0.5},
                      'CMOD' : {'rho' : S.shape[1] / 200.0}})

# Run optimisation
d = bpdndl.BPDNDictLearn(D0, S, lmbda, opt)
d.solve()
print("BPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))


# Display dictionaries
D1 = d.getdict().reshape((8,8,D0.shape[1]))
D0 = D0.reshape(8,8,D0.shape[-1])
fig1 = plot.figure(1, figsize=(14,7))
plot.subplot(1,2,1)
plot.imview(util.tiledict(D0), fgrf=fig1, title='D0')
plot.subplot(1,2,2)
plot.imview(util.tiledict(D1), fgrf=fig1, title='D1')
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
