#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Convolutional Dictionary Learning with Spatial Mask
===================================================

This example demonstrates the use of :class:`.cbpdndl.ConvBPDNMaskDcplDictLearn` for convolutional dictionary learning with a spatial mask, from a set of greyscale training images. The dictionary learning algorithm is based on the hybrid mask decoupling / ADMM consensus dictionary update :cite:`garcia-2017-convolutional`.
"""


from __future__ import print_function
from builtins import input
from builtins import range

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.admm import tvl2
from sporco.admm import cbpdndl
from sporco import util
from sporco import plot


"""
Load training images.
"""

exim = util.ExampleImages(scaled=True, zoom=0.25, gray=True)
S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
S = np.dstack((S1, S2))


"""
Construct initial dictionary.
"""

np.random.seed(12345)
D0 = np.random.randn(8, 8, 32)


"""
Create random mask and apply to training images.
"""

t = 0.5
W = np.random.randn(*(S.shape[0:2] + (1,)))
W[np.abs(W) > t] = 1;
W[np.abs(W) < t] = 0;
Sw = W * S


"""
$\ell_2$-TV denoising with a spatial mask as a non-linear lowpass filter.
"""

lmbda = 0.1
opt = tvl2.TVL2Denoise.Options({'Verbose': False, 'MaxMainIter': 200,
            'DFidWeight': W, 'gEvalY': False, 'AutoRho': {'Enabled': True}})
b = tvl2.TVL2Denoise(Sw, lmbda, opt)
sl = b.solve()
sh = Sw - sl


"""
CDL without a spatial mask using :class:`.admm.cbpdndl.ConvBPDNDictLearn`.
"""

lmbda = 0.05
opt1 = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True,
            'MaxMainIter': 200, 'AccurateDFid': True,
            'CBPDN': {'rho': 50.0*lmbda + 0.5},
            'CCMOD': {'rho': 1e2}})
d1 = cbpdndl.ConvBPDNDictLearn(D0, sh, lmbda, opt1)
D1 = d1.solve()


"""
Reconstruct from the CDL solution without a spatial mask.
"""

sr1 = d1.reconstruct().squeeze() + sl


"""
CDL with a spatial mask using :class:`.cbpdndl.ConvBPDNMaskDcplDictLearn`. (Note that :class:`.parcnsdl.ConvBPDNMaskDcplDictLearn_Consensus` solves the same problem, but is substantially faster on a multi-core architecture.)
"""

opt2 = cbpdndl.ConvBPDNMaskDcplDictLearn.Options({'Verbose': True,
            'MaxMainIter': 200, 'AccurateDFid': True,
            'CBPDN': {'rho': 20.0*lmbda + 0.5},
            'CCMOD': {'rho': 2e-1}})
d2 = cbpdndl.ConvBPDNMaskDcplDictLearn(D0, sh, lmbda, W, opt2)
D2 = d2.solve()


"""
Reconstruct from the CDL solution with a spatial mask.
"""

sr2 = d2.reconstruct().squeeze() + sl


"""
Compare dictionaries.
"""

fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D1.squeeze()), fig=fig,
            title='Without Mask Decoupling')
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D2.squeeze()), fig=fig,
            title='With Mask Decoupling')
fig.show()


"""
Display reference and training images.
"""

fig = plot.figure(figsize=(14, 14))
plot.subplot(2, 2, 1)
plot.imview(S[...,0], fig=fig, title='Reference')
plot.subplot(2, 2, 2)
plot.imview(Sw[...,0], fig=fig, title='Test')
plot.subplot(2, 2, 3)
plot.imview(S[...,1], fig=fig, title='Reference')
plot.subplot(2, 2, 4)
plot.imview(Sw[...,1], fig=fig, title='Test')
fig.show()


"""
Compare reconstructed images.
"""

fig = plot.figure(figsize=(14, 14))
plot.subplot(2, 2, 1)
plot.imview(sr1[...,0], fig=fig, title='Without Mask Decoupling')
plot.subplot(2, 2, 2)
plot.imview(sr2[...,0], fig=fig, title='With Mask Decoupling')
plot.subplot(2, 2, 3)
plot.imview(sr1[...,1], fig=fig, title='Without Mask Decoupling')
plot.subplot(2, 2, 4)
plot.imview(sr2[...,1], fig=fig, title='With Mask Decoupling')
fig.show()


# Wait for enter on keyboard
input()
