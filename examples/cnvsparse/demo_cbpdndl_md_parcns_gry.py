#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: parcnsdl.ConvBPDNMaskDcplDictLearn_Consensus
(greyscale images).
"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco.admm import parcnsdl
from sporco import util
from sporco import plot


# Training images
exim = util.ExampleImages(scaled=True, zoom=0.25, gray=True)
S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
S = np.dstack((S1, S2))


# Initial dictionary
np.random.seed(12345)
D0 = np.random.randn(8, 8, 32)


# Highpass filter test images
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(S, fltlmbd, npd)


# Zero pad highpass component
Npr, Npc = np.array(D0.shape[0:2]) - 1
shp = np.pad(sh, ((0, Npr), (0, Npc), (0, 0)), mode='constant')


# Create random mask and apply to highpass component
t = 0.5
W = np.random.randn(*sh.shape)
W[np.abs(W) > t] = 1;
W[np.abs(W) < t] = 0;
W = np.pad(W, ((0, Npr), (0, Npc), (0, 0)), mode='constant')
shpw = W * shp


# Solve ConvBPDNDictLearn_Consensus problem
lmbda = 0.05
opt1 = parcnsdl.ConvBPDNDictLearn_Consensus.Options(
                     {'Verbose': True, 'MaxMainIter': 200,
                      'AccurateDFid': True,
                      'CBPDN': {'rho': 50.0*lmbda + 0.5},
                      'CCMOD': {'ZeroMean': True}})
d1 = parcnsdl.ConvBPDNDictLearn_Consensus(D0, shpw, lmbda, opt1)
D1 = d1.solve()


# Reconstruct from ConvBPDNDictLearn_Consensus solution
sr1 = d1.reconstruct()[:-Npr, :-Npc].squeeze() + sl


# Solve ConvBPDNMaskDcplDictLearn_Consensus problem
opt2 = parcnsdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
                     {'Verbose': True, 'MaxMainIter': 200,
                      'AccurateDFid': True,
                      'CBPDN': {'rho': 50.0*lmbda + 0.5},
                      'CCMOD': {'ZeroMean': True}})
Wr = np.reshape(W, W.shape[0:2] + (1, W.shape[2], 1))
d2 = parcnsdl.ConvBPDNMaskDcplDictLearn_Consensus(D0, shpw, lmbda, Wr, opt2)
D2 = d2.solve()


# Reconstruct from ConvBPDNMaskDcplDictLearn_Consensus solution
sr2 = d2.reconstruct()[:-Npr, :-Npc].squeeze() + sl


# Compare dictionaries
fig1 = plot.figure(1, figsize=(14,7))
plot.subplot(1,2,1)
plot.imview(util.tiledict(D1.squeeze()), fgrf=fig1,
            title='Without Mask Decoupling')
plot.subplot(1,2,2)
plot.imview(util.tiledict(D2.squeeze()), fgrf=fig1,
            title='With Mask Decoupling')
fig1.show()


# Display reference and test images (with unmasked lowpass component)
fig2 = plot.figure(2, figsize=(14,14))
plot.subplot(2,2,1)
plot.imview(S[...,0], fgrf=fig2, title='Reference')
plot.subplot(2,2,2)
plot.imview(shpw[:-Npr, :-Npc, 0] + sl[...,0], fgrf=fig2, title='Test')
plot.subplot(2,2,3)
plot.imview(S[...,1], fgrf=fig2, title='Reference')
plot.subplot(2,2,4)
plot.imview(shpw[:-Npr, :-Npc, 1] + sl[...,1], fgrf=fig2, title='Test')
fig2.show()


# Compare reconstructed images
fig3 = plot.figure(3, figsize=(14,14))
plot.subplot(2,2,1)
plot.imview(sr1[...,0], fgrf=fig3, title='Without Mask Decoupling')
plot.subplot(2,2,2)
plot.imview(sr2[...,0], fgrf=fig3, title='With Mask Decoupling')
plot.subplot(2,2,3)
plot.imview(sr1[...,1], fgrf=fig3, title='Without Mask Decoupling')
plot.subplot(2,2,4)
plot.imview(sr2[...,1], fgrf=fig3, title='With Mask Decoupling')
fig3.show()


# Wait for enter on keyboard
input()
