#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Gaussian White Noise Restoration via SC
=======================================

This example demonstrates the removal of Gaussian white noise from a colour image using sparse coding.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco import util
from sporco import array
from sporco import plot
import sporco.metric as sm
from sporco.admm import bpdn


r"""
Load a reference image and corrupt it with Gaussian white noise with $\sigma = 0.1$. (The call to ``numpy.random.seed`` ensures that the pseudo-random noise is reproducible.)
"""

img = util.ExampleImages().image('monarch.png', zoom=0.5, scaled=True,
                                 idxexp=np.s_[:, 160:672])
np.random.seed(12345)
imgn = img + np.random.normal(0.0, 0.1, img.shape)


"""
Extract blocks and center each channel of image patches, taking steps of size 2.
"""

blksz = (8, 8, 3)
stpsz = (2, 2, 1)

blocks = array.extract_blocks(imgn, blksz, stpsz)
blockmeans = np.mean(blocks, axis=(0, 1))
blocks -= blockmeans
blocks = blocks.reshape(np.prod(blksz), -1)


"""
Load dictionary.
"""

D = util.convdicts()['RGB:8x8x3x64'].reshape(np.prod(blksz), -1)


"""
Set solver options.
"""

lmbda = 1e-1
opt = bpdn.BPDN.Options({'Verbose': True, 'MaxMainIter': 250,
                         'RelStopTol': 3e-3, 'AuxVarObj': False,
                         'AutoRho': {'Enabled': False}, 'rho':
                         1e1*lmbda})


"""
Initialise the :class:`.admm.bpdn.BPDN` object and call the ``solve`` method.
"""

b = bpdn.BPDN(D, blocks, lmbda, opt)
X = b.solve()


"""
The denoised estimate of the image is by aggregating the block reconstructions from the coefficient maps.
"""

imgd_mean = array.average_blocks(np.dot(D, X).reshape(blksz + (-1,))
                                 + blockmeans, img.shape, stpsz)
imgd_median = array.combine_blocks(np.dot(D, X).reshape(blksz + (-1,))
                                   + blockmeans, img.shape, stpsz, np.median)


"""
Display solve time and denoising performance.
"""

print("BPDN solve time: %5.2f s" % b.timer.elapsed('solve'))
print("Noisy image PSNR:    %5.2f dB" % sm.psnr(img, imgn))
print("Denoised mean image PSNR: %5.2f dB" % sm.psnr(img, imgd_mean))
print("Denoised median image PSNR: %5.2f dB" % sm.psnr(img, imgd_median))


"""
Display the reference, noisy, and denoised images.
"""

fig = plot.figure(figsize=(14, 14))
plot.subplot(2, 2, 1)
plot.imview(img, title='Reference', fig=fig)
plot.subplot(2, 2, 2)
plot.imview(imgn, title='Noisy', fig=fig)
plot.subplot(2, 2, 3)
plot.imview(imgd_mean, title='SC mean Result', fig=fig)
plot.subplot(2, 2, 4)
plot.imview(imgd_median, title='SC median Result', fig=fig)
fig.show()


"""
Plot functional evolution during ADMM iterations.
"""

its = b.getitstat()
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional')



# Wait for enter on keyboard
input()
