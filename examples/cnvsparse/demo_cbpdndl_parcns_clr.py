#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: parcnsdl.ConvBPDNDictLearn_Consensus (colour images,
colour dictionary).
"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco.admm import parcnsdl
from sporco import util
from sporco import plot


# Training images
exim = util.ExampleImages(scaled=True, zoom=0.25)
S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
S3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
S4 = exim.image('sail.png', idxexp=np.s_[:, 210:722])
S = np.stack((S1,S2,S3,S4), axis=3)


# Highpass filter test images
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(S, fltlmbd, npd)


# Initial dictionary
np.random.seed(12345)
D0 = np.random.randn(8, 8, 3, 64)


# Set ConvBPDNDictLearn parameters
lmbda = 0.2
opt = parcnsdl.ConvBPDNDictLearn_Consensus.Options(
    {'Verbose': True, 'MaxMainIter': 200,
     'CBPDN': {'rho': 50.0*lmbda + 0.5},
     'CCMOD': {'ZeroMean': True}})


# Run optimisation
d = parcnsdl.ConvBPDNDictLearn_Consensus(D0, sh, lmbda, opt)
D1 = d.solve()
print("ConvBPDNDictLearn_Consensus solve time: %.2fs" %
      d.timer.elapsed('solve'))


# Display dictionaries
D1 = D1.squeeze()
fig1 = plot.figure(1, figsize=(14,7))
plot.subplot(1,2,1)
plot.imview(util.tiledict(D0), fgrf=fig1, title='D0')
plot.subplot(1,2,2)
plot.imview(util.tiledict(D1), fgrf=fig1, title='D1')
fig1.show()


# Plot functional value and residuals
its = d.getitstat()
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional')

# Wait for enter on keyboard
input()
