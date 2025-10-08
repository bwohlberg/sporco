#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Single-channel CSC (Constrained Penalty Term)
=============================================

This example demonstrates solving a constrained convolutional sparse coding problem with a greyscale signal

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \right\|_2^2 \; \text{such that} \; \sum_m \| \mathbf{x}_m \|_1 \leq \gamma \;,$$

where $\mathbf{d}_{m}$ is the $m^{\text{th}}$ dictionary filter, $\mathbf{x}_{m}$ is the coefficient map corresponding to the $m^{\text{th}}$ dictionary filter, and $\mathbf{s}$ is the input image.
"""



from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco import util
from sporco import signal
from sporco import plot
import sporco.metric as sm
from sporco.admm import cbpdn


"""
Load example image.
"""

img = util.ExampleImages().image('kodim23.png', scaled=True, gray=True,
                                 idxexp=np.s_[160:416,60:316])


"""
Highpass filter example image.
"""

npd = 16
fltlmbd = 10
sl, sh = signal.tikhonov_filter(img, fltlmbd, npd)


"""
Load dictionary and display it.
"""

D = util.convdicts()['G:12x12x36']
plot.imview(util.tiledict(D), fgsz=(7, 7))


"""
Set :class:`.admm.cbpdn.ConvBPDNProjL1` solver options.
"""

gamma = 4.05e2
opt = cbpdn.ConvBPDNProjL1.Options({'Verbose': True, 'MaxMainIter': 250,
                    'HighMemSolve': True, 'LinSolveCheck': False,
                    'RelStopTol': 5e-3, 'AuxVarObj': True, 'rho': 3e0,
                    'AutoRho': {'Enabled': True}})


"""
Initialise and run CSC solver.
"""

b = cbpdn.ConvBPDNProjL1(D, sh, gamma, opt)
X = b.solve()
print("ConvBPDNProjL1 solve time: %.2fs" % b.timer.elapsed('solve'))


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
Get iterations statistics from solver object and plot functional value, ADMM primary and dual residuals, and automatically adjusted ADMM penalty parameter against the iteration number.
"""

its = b.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'], fig=fig)
plot.subplot(1, 3, 3)
plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter', fig=fig)
fig.show()


# Wait for enter on keyboard
input()
