#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Convolutional Dictionary Learning with Spatial Mask
===================================================

This example demonstrates the use of :class:`.cbpdndlmd.ConvBPDNMaskDictLearn` for convolutional dictionary learning with a spatial mask, from a set of colour training images :cite:`wohlberg-2016-convolutional`. The dictionary learning algorithm is based on the hybrid mask decoupling / ADMM consensus dictionary update :cite:`garcia-2018-convolutional1`.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.admm import tvl2
from sporco.dictlrn import cbpdndl
from sporco.dictlrn import cbpdndlmd
from sporco import util
from sporco import signal
from sporco import plot


"""
Load training images.
"""

exim = util.ExampleImages(scaled=True, zoom=0.25)
S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
S = np.stack((S1, S2), axis=3)


"""
Construct initial dictionary.
"""

np.random.seed(12345)
D0 = np.random.randn(8, 8, 3, 32)


"""
Create random mask and apply to training images.
"""

frc = 0.5
W = signal.rndmask(S.shape[0:3] + (1,), frc, dtype=np.float32)
Sw = W * S


r"""
$\ell_2$-TV denoising with a spatial mask as a non-linear lowpass filter.
"""

lmbda = 0.1
opt = tvl2.TVL2Denoise.Options({'Verbose': False, 'MaxMainIter': 200,
            'DFidWeight': W, 'gEvalY': False, 'AutoRho': {'Enabled': True}})
b = tvl2.TVL2Denoise(Sw, lmbda, opt, caxis=2)
sl = b.solve()
sh = Sw - sl


"""
CDL without a spatial mask using :class:`.dictlrn.cbpdndl.ConvBPDNDictLearn`. (Note that :class:`.prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus` solves the same problem, but is substantially faster on a multi-core architecture.)
"""

lmbda = 0.05
opt1 = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True,
            'MaxMainIter': 200, 'AccurateDFid': True,
            'CBPDN': {'rho': 50.0*lmbda + 0.5},
            'CCMOD': {'rho': 3e2}}, dmethod='cns')
d1 = cbpdndl.ConvBPDNDictLearn(D0, sh, lmbda, opt1, dmethod='cns')
D1 = d1.solve()


"""
Reconstruct from the CDL solution without a spatial mask.
"""

sr1 = d1.reconstruct().squeeze() + sl


"""
CDL with a spatial mask using :class:`.cbpdndlmd.ConvBPDNMaskDictLearn`.
"""

opt2 = cbpdndlmd.ConvBPDNMaskDictLearn.Options({'Verbose': True,
            'MaxMainIter': 200, 'AccurateDFid': True,
            'CBPDN': {'rho': 50.0*lmbda + 0.5},
            'CCMOD': {'rho': 1.0}}, dmethod='cns')
d2 = cbpdndlmd.ConvBPDNMaskDictLearn(D0, sh, lmbda, W, opt2,
                                         dmethod='cns')
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
plot.imview(S[...,0], title='Reference', fig=fig)
plot.subplot(2, 2, 2)
plot.imview(Sw[...,0], title='Test', fig=fig)
plot.subplot(2, 2, 3)
plot.imview(S[...,1], title='Reference', fig=fig)
plot.subplot(2, 2, 4)
plot.imview(Sw[...,1], title='Test', fig=fig)
fig.show()


"""
Compare reconstructed images.
"""

fig = plot.figure(figsize=(14, 14))
plot.subplot(2, 2, 1)
plot.imview(sr1[...,0], title='Without Mask Decoupling', fig=fig)
plot.subplot(2, 2, 2)
plot.imview(sr2[...,0], title='With Mask Decoupling', fig=fig)
plot.subplot(2, 2, 3)
plot.imview(sr1[...,1], title='Without Mask Decoupling', fig=fig)
plot.subplot(2, 2, 4)
plot.imview(sr2[...,1], title='With Mask Decoupling', fig=fig)
fig.show()


# Wait for enter on keyboard
input()
