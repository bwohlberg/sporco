#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Online Convolutional Dictionary Learning with Spatial Mask
==========================================================

This example demonstrates the use of :class:`.dictlrn.onlinecdl.OnlineConvBPDNMaskDictLearn` for learning a convolutional dictionary from a set of training images. The dictionary is learned using the online dictionary learning algorithm proposed in :cite:`liu-2018-first`.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.dictlrn import onlinecdl
from sporco import util
from sporco import signal
from sporco import cuda
from sporco import plot


"""
Load training images.
"""

exim = util.ExampleImages(scaled=True, zoom=0.5, gray=True)
S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
S3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
S4 = exim.image('sail.png', idxexp=np.s_[:, 210:722])
S5 = exim.image('tulips.png', idxexp=np.s_[:, 30:542])
S = np.dstack((S1, S2, S3, S4, S5))


"""
Highpass filter training images.
"""

npd = 16
fltlmbd = 5
sl, sh = signal.tikhonov_filter(S, fltlmbd, npd)


"""
Create random mask and apply to highpass filtered training image set.
"""

np.random.seed(12345)
frc = 0.25
W = signal.rndmask(S.shape, frc, dtype=np.float32)
shw = W * sh


"""
Construct initial dictionary.
"""

D0 = np.random.randn(8, 8, 32)


"""
Set regularization parameter and options for dictionary learning solver.
"""

lmbda = 0.1
opt = onlinecdl.OnlineConvBPDNMaskDictLearn.Options({
                'Verbose': True, 'ZeroMean': False, 'eta_a': 10.0,
                'eta_b': 20.0, 'DataType': np.float32,
                'CBPDN': {'rho': 3.0, 'AutoRho': {'Enabled': False},
                    'RelaxParam': 1.8, 'RelStopTol': 1e-4, 'MaxMainIter': 100,
                    'FastSolve': False, 'DataType': np.float32}})
if cuda.device_count() > 0:
    opt['CUDA_CBPDN'] = True


"""
Create solver object and solve.
"""

d = onlinecdl.OnlineConvBPDNMaskDictLearn(D0, lmbda, opt)

iter = 50
d.display_start()
for it in range(iter):
    img_index = np.random.randint(0, sh.shape[-1])
    d.solve(shw[..., [img_index]], W[..., [img_index]])

d.display_end()
D1 = d.getdict()
print("OnlineConvBPDNMaskDictLearn solve time: %.2fs" %
      d.timer.elapsed('solve'))


"""
Display initial and final dictionaries.
"""

D1 = D1.squeeze()
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D1), title='D1', fig=fig)
fig.show()


"""
Get iterations statistics from solver object and plot functional value.
"""

its = d.getitstat()
fig = plot.figure(figsize=(7, 7))
plot.plot(np.vstack((its.DeltaD, its.Eta)).T, xlbl='Iterations',
          lgnd=('Delta D', 'Eta'), fig=fig)
fig.show()


# Wait for enter on keyboard
input()
