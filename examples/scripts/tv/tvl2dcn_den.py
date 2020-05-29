#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Greyscale ℓ2-TV Denoising
=========================

This example demonstrates the use of class :class:`.tvl2.TVL2Deconv` for removing Gaussian white noise from a greyscale image using Total Variation regularization with an ℓ2 data fidelity term (ℓ2-TV denoising). (This class is primarily intended for deconvolution problems, but can be applied to denoising problems by choosing an impulse filter as the blurring kernel.)
"""


from __future__ import print_function
from builtins import input

import numpy as np

from sporco.admm import tvl2
from sporco import util
from sporco import metric
from sporco import plot


"""
Load reference image.
"""

img = util.ExampleImages().image('monarch.png', scaled=True,
                                 idxexp=np.s_[:,160:672], gray=True)


"""
Construct test image corrupted Gaussian white noise with a 0.05 standard deviation.
"""

np.random.seed(12345)
imgn = img + np.random.normal(0.0, 0.05, img.shape)


"""
Set regularization parameter and options for ℓ2-TV deconvolution solver. The regularization parameter used here has been manually selected for good performance.
"""

lmbda = 0.04
opt = tvl2.TVL2Deconv.Options({'Verbose': True, 'MaxMainIter': 200,
                               'gEvalY': False})



"""
Create solver object and solve, returning the the denoised image ``imgr``.
"""

b = tvl2.TVL2Deconv(np.ones((1,1)), imgn, lmbda, opt)
imgr = b.solve()


"""
Display solve time and denoising performance.
"""

print("TVL2Deconv solve time: %5.2f s" % b.timer.elapsed('solve'))
print("Noisy image PSNR:    %5.2f dB" % metric.psnr(img, imgn))
print("Denoised image PSNR: %5.2f dB" % metric.psnr(img, imgr))


"""
Display reference, corrupted, and denoised images.
"""

fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.imview(img, title='Reference', fig=fig)
plot.subplot(1, 3, 2)
plot.imview(imgn, title='Corrupted', fig=fig)
plot.subplot(1, 3, 3)
plot.imview(imgr, title=r'Restored ($\ell_2$-TV)', fig=fig)
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
