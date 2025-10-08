#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
PGM CBPDN Solver
==================

This example demonstrates use of a PGM solver for a convolutional sparse coding problem with a colour dictionary and a colour signal :cite:`wohlberg-2016-convolutional` :cite:`garcia-2018-convolutional1`

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \sum_c \left\| \sum_m \mathbf{d}_{c,m} * \mathbf{x}_m -\mathbf{s}_c \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 \;,$$

where $\mathbf{d}_{c,m}$ is channel $c$ of the $m^{\text{th}}$ dictionary filter, $\mathbf{x}_m$ is the coefficient map corresponding to the $m^{\text{th}}$ dictionary filter, and $\mathbf{s}_c$ is channel $c$ of the input image.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco import util
from sporco import signal
from sporco import plot
import sporco.metric as sm
from sporco.pgm import cbpdn
from sporco.pgm.backtrack import BacktrackStandard

"""
Load example image.
"""

img = util.ExampleImages().image('kodim23.png', scaled=True,
                                idxexp=np.s_[160:416,60:316])


"""
Highpass filter example image.
"""

npd = 16
fltlmbd = 10
sl, sh = signal.tikhonov_filter(img, fltlmbd, npd)


"""
Load colour dictionary and display it.
"""

D = util.convdicts()['RGB:8x8x3x64']
plot.imview(util.tiledict(D), fgsz=(7, 7))


"""
Set :class:`.pgm.cbpdn.ConvBPDN` solver options. Note the possibility of changing parameters in the backtracking algorithm.
"""

lmbda = 1e-1
L = 1e1
opt = cbpdn.ConvBPDN.Options({
    'Verbose': True, 'MaxMainIter': 250, 'RelStopTol': 8e-5, 'L': L,
    'Backtrack': BacktrackStandard(maxiter=15)})


"""
Initialise and run CSC solver.
"""

b = cbpdn.ConvBPDN(D, sh, lmbda, opt)
X = b.solve()
print("ConvBPDN solve time: %.2fs" % b.timer.elapsed('solve'))


"""
Reconstruct image from sparse representation.
"""

shr = b.reconstruct().squeeze()
imgr = sl + shr
print("Reconstruction PSNR: %.2fdB\n" % sm.psnr(img, imgr))


"""
Display low pass component and sum of absolute values of coefficient maps of highpass component.
"""

fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(sl, title='Lowpass component', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(np.sum(abs(X), axis=b.cri.axisM).squeeze(), cmap=plot.cm.Blues,
            title='Sparse representation', fig=fig)
fig.show()


"""
Display original and reconstructed images.
"""

fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(img, title='Original', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(imgr, title='Reconstructed', fig=fig)
fig.show()


"""
Get iterations statistics from solver object and plot functional value, residual, and inverse step size parameter against the iteration number.
"""

its = b.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(its.Rsdl, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          fig=fig)
plot.subplot(1, 3, 3)
plot.plot(its.L, xlbl='Iterations',
          ylbl='Inverse of Gradient Step Parameter', fig=fig)
fig.show()


# Wait for enter on keyboard
input()
