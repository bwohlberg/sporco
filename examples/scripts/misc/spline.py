#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
ℓ1-Spline
=========

This example demonstrates the use of class :class:`.spline.SplineL1` for removing salt & pepper noise from a greyscale image using ℓ1-spline fitting.
"""


from __future__ import print_function
from builtins import input

import numpy as np

from sporco.admm import spline
from sporco import util
from sporco import signal
from sporco import metric
from sporco import plot

"""
Load reference image.
"""

img = util.ExampleImages().image('monarch.png', scaled=True,
                                 idxexp=np.s_[:,160:672], gray=True)


"""
Construct test image corrupted by 20% salt & pepper noise.
"""

np.random.seed(12345)
imgn = signal.spnoise(img, 0.2)


"""
Set regularization parameter and options for ℓ1-spline solver. The regularization parameter used here has been manually selected for good performance.
"""

lmbda = 5.0
opt = spline.SplineL1.Options({'Verbose': True, 'gEvalY': False})


"""
Create solver object and solve, returning the the denoised image ``imgr``.
"""

b = spline.SplineL1(imgn, lmbda, opt)
imgr = b.solve()


"""
Display solve time and denoising performance.
"""

print("SplineL1 solve time: %5.2f s" % b.timer.elapsed('solve'))
print("Noisy image PSNR:    %5.2f dB" % metric.psnr(img, imgn))
print("Denoised image PSNR: %5.2f dB" % metric.psnr(img, imgr))


"""
Display reference, corrupted, and denoised images.
"""

fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.imview(img, title='Reference', fig=fig)
plot.subplot(1, 3, 2)
plot.imview(imgn, title='Corrupted', fig=fig)
plot.subplot(1, 3, 3)
plot.imview(imgr, title=r'Restored ($\ell_1$-spline)', fig=fig)
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
