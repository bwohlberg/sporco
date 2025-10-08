#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Gaussian White Noise Restoration via CSC
========================================

This example demonstrates the removal of Gaussian white noise from a greyscale image using the CSC problem

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \sum_c \left\| \sum_m \mathbf{d}_{c,m} * \mathbf{x}_m -\mathbf{s}_c \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 \;,$$

where $\mathbf{d}_m$ is the $m^{\text{th}}$ dictionary filter, $\mathbf{x}_m$ is the coefficient map corresponding to the $m^{\text{th}}$ dictionary filter, $\mathbf{s}$ is the input image.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco import util
from sporco import signal
from sporco import fft
from sporco import metric
from sporco import plot
from sporco.admm import cbpdn


"""
Boundary artifacts are handled by performing a symmetric extension on the image to be denoised and then cropping the result to the original image support. This approach is simpler than the boundary handling strategies that involve the insertion of a spatial mask into the data fidelity term, and for many problems gives results of comparable quality. The functions defined here implement symmetric extension and cropping of images.
"""

def pad(x, n=8):

    if x.ndim == 2:
        return np.pad(x, n, mode='symmetric')
    else:
        return np.pad(x, ((n, n), (n, n), (0, 0)), mode='symmetric')


def crop(x, n=8):

    return x[n:-n, n:-n]


r"""
Load a reference image and corrupt it with Gaussian white noise with $\sigma = 0.1$. (The call to ``numpy.random.seed`` ensures that the pseudo-random noise is reproducible.)
"""

img = util.ExampleImages().image('monarch.png', zoom=0.5, scaled=True,
                                 gray=True, idxexp=np.s_[:, 160:672])
np.random.seed(12345)
imgn = img + np.random.normal(0.0, 0.1, img.shape).astype(np.float32)


"""
Highpass filter test image.
"""

npd = 16
fltlmbd = 5.0
imgnl, imgnh = signal.tikhonov_filter(imgn, fltlmbd, npd)


"""
Load dictionary.
"""

D = util.convdicts()['G:8x8x128']


r"""
Set solver options. See Section 8 of :cite:`wohlberg-2017-convolutional2` for details of construction of $\ell_1$ weighting matrix $W$.
"""

imgnpl, imgnph = signal.tikhonov_filter(pad(imgn), fltlmbd, npd)
W = fft.irfftn(np.conj(fft.rfftn(D, imgnph.shape, (0, 1))) *
               fft.rfftn(imgnph[..., np.newaxis], None, (0, 1)),
               imgnph.shape, (0,1))
W = W**2
W = 1.0/(np.maximum(np.abs(W), 1e-8))

lmbda = 4.8e-2

opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 250,
            'HighMemSolve': True, 'RelStopTol': 3e-3, 'AuxVarObj': False,
            'L1Weight': W, 'AutoRho': {'Enabled': False}, 'rho': 4e2*lmbda})


"""
Initialise the :class:`.admm.cbpdn.ConvBPDN` object and call the ``solve`` method.
"""

b = cbpdn.ConvBPDN(D, pad(imgnh), lmbda, opt, dimK=0)
X = b.solve()


"""
The denoised estimate of the image is just the reconstruction from the coefficient maps.
"""

imgdp = b.reconstruct().squeeze()
imgd = np.clip(crop(imgdp) + imgnl, 0, 1)


"""
Display solve time and denoising performance.
"""

print("ConvBPDN solve time: %5.2f s" % b.timer.elapsed('solve'))
print("Noisy image PSNR:    %5.2f dB" % metric.psnr(img, imgn))
print("Denoised image PSNR: %5.2f dB" % metric.psnr(img, imgd))


"""
Display the reference, noisy, and denoised images.
"""

fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.imview(img, title='Reference', fig=fig)
plot.subplot(1, 3, 2)
plot.imview(imgn, title='Noisy', fig=fig)
plot.subplot(1, 3, 3)
plot.imview(imgd, title='CSC Result', fig=fig)
fig.show()


"""
Plot functional evolution during ADMM iterations.
"""

its = b.getitstat()
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional')


"""
Plot evolution of ADMM residuals and ADMM penalty parameter.
"""

plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'])
plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter')



# Wait for enter on keyboard
input()
