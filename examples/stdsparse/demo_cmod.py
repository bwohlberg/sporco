#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: cmod.CnstrMOD"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco.admm import bpdn
from sporco.admm import cmod
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


# Load dictionary
D0 = util.convdicts()['G:8x8x64']
D0 = np.reshape(D0, (np.prod(D0.shape[0:2]), D0.shape[2]))


# Compute sparse representation on current dictionary
lmbda = 0.1
opt = bpdn.BPDN.Options({'Verbose' : True, 'MaxMainIter' : 200,
                         'RelStopTol' : 1e-3})
b = bpdn.BPDN(D0, S, lmbda, opt)
b.solve()
print("BPDN solve time: %.2fs\n" % b.timer.elapsed('solve'))


# Update dictionary for training set S
opt = cmod.CnstrMOD.Options({'Verbose' : True, 'MaxMainIter' : 100,
                             'RelStopTol' : 1e-3, 'rho' : 4e2})
c = cmod.CnstrMOD(b.Y, S, None, opt)
c.solve()
print("CMOD solve time: %.2fs" % c.timer.elapsed('solve'))


# Display dictionaries
D0 = D0.reshape(8,8,D0.shape[-1])
D1 = c.Y.reshape((8,8,c.Y.shape[-1]))
fig1 = plot.figure(1, figsize=(20,10))
plot.subplot(1,2,1)
plot.imview(util.tiledict(D0), fgrf=fig1, title='D0')
plot.subplot(1,2,2)
plot.imview(util.tiledict(D1), fgrf=fig1, title='D1')
fig1.show()


# Plot functional value, residuals, and rho
its = c.getitstat()
fig2 = plot.figure(2, figsize=(21,7))
plot.subplot(1,3,1)
plot.plot(its.DFid, fgrf=fig2, xlbl='Iterations', ylbl='Functional')
plot.subplot(1,3,2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T, fgrf=fig2,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'])
plot.subplot(1,3,3)
plot.plot(its.Rho, fgrf=fig2, xlbl='Iterations', ylbl='Penalty Parameter')
fig2.show()


# Wait for enter on keyboard
input()
