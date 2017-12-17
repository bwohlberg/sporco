#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: cbpdndl.ConvBPDNDictLearn with 3d data and dictionary"""

from __future__ import print_function
from builtins import input
from builtins import range

import os.path
import tempfile
import sys
import numpy as np
try:
    import skvideo.datasets
    import skvideo.io
except ImportError:
    print('Package sk-video is required by this demo script', file=sys.stderr)
    raise

from sporco.admm import cbpdndl
from sporco import util
from sporco import plot


# Get test video as 3d array
vid = skvideo.io.vread(skvideo.datasets.fullreferencepair()[0],
                outputdict={"-pix_fmt": "gray"})[..., 0]
vid = np.moveaxis(vid, 0, -1)
vid = vid[0:106,40:136, 10:42].astype(np.float32)/255.0


# Highpass filter frames
npd = 16
fltlmbd = 10
vl, vh = util.tikhonov_filter(vid, fltlmbd, npd)


# Initial dictionary
np.random.seed(12345)
D0 = np.random.randn(5, 5, 3, 25)


# Set ConvBPDNDictLearn parameters
lmbda = 0.1
opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 200,
                'CBPDN': {'rho': 5e1*lmbda, 'AutoRho': {'Enabled': True}},
                'CCMOD': {'rho': 1e2, 'AutoRho': {'Enabled': True}}})


# Run optimisation
d = cbpdndl.ConvBPDNDictLearn(D0, vh, lmbda, opt, dimK=0, dimN=3)
D1 = d.solve()
print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))


# Display central temporal slice (index 2) of dictionaries
D1 = D1.squeeze()
fig1 = plot.figure(1, figsize=(14,7))
plot.subplot(1,2,1)
plot.imview(util.tiledict(D0[...,1,:]), fgrf=fig1, title='D0')
plot.subplot(1,2,2)
plot.imview(util.tiledict(D1[...,1,:]), fgrf=fig1, title='D1')
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
