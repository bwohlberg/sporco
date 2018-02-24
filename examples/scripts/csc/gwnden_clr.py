#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""

Gaussian White Noise Restoration via CSC
========================================

This example demonstrates the removal of Gaussian white noise from a colour image using convolutional sparse coding :cite:`wohlberg-2016-convolutional`,

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \sum_c \left\| \sum_m \mathbf{d}_{m} * \mathbf{x}_{c,m} -\mathbf{s}_c \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 + \mu \| \{ \mathbf{x}_{c,m} \} \|_{2,1}$$

where $\mathbf{d}_m$ is the $m^{\text{th}}$ dictionary filter, $\mathbf{x}_{c,m}$ is the coefficient map corresponding to the $c^{\text{th}}$ colour band and $m^{\text{th}}$ dictionary filter, and $\mathbf{s}_c$ is colour band $c$ of the input image.
"""


from __future__ import print_function
from builtins import input
from builtins import range

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco import util
from sporco import plot
import sporco.linalg as spl
import sporco.metric as sm
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


"""
Load a reference image and corrupt it with Gaussian white noise with $\sigma = 0.1$. (The call to :func:`numpy.random.seed` ensures that the pseudo-random noise is reproducible.)
"""

img = util.ExampleImages().image('monarch.png', zoom=0.5, scaled=True,
                                 idxexp=np.s_[:, 160:672])
np.random.seed(12345)
imgn = img + np.random.normal(0.0, 0.1, img.shape)


"""
Highpass filter test image.
"""

npd = 16
fltlmbd = 5.0
imgnl, imgnh = util.tikhonov_filter(imgn, fltlmbd, npd)


"""
Load dictionary.
"""

D = util.convdicts()['G:8x8x128']


"""
Set solver options. See Section 8 of :cite:`wohlberg-2017-convolutional2` for details of construction of $\ell_1$ weighting matrix $W$.
"""

imgnpl, imgnph = util.tikhonov_filter(pad(imgn), fltlmbd, npd)
W = spl.irfftn(np.conj(spl.rfftn(D[..., np.newaxis, :], imgnph.shape[0:2],
               (0, 1))) * spl.rfftn(imgnph[..., np.newaxis], None, (0, 1)),
               imgnph.shape[0:2], (0, 1))
W = 1.0/(np.maximum(np.abs(W), 1e-8))

lmbda = 1.5e-2
mu = 2.7e-1

opt = cbpdn.ConvBPDNJoint.Options({'Verbose': True, 'MaxMainIter': 250,
            'HighMemSolve': True, 'RelStopTol': 3e-3, 'AuxVarObj': False,
            'L1Weight': W, 'AutoRho': {'Enabled': False}, 'rho': 1e3*lmbda})


"""
Initialise the :class:`.admm.cbpdn.ConvBPDNJoint` object and call the ``solve`` method.
"""

b = cbpdn.ConvBPDNJoint(D, pad(imgnh), lmbda, mu, opt, dimK=0)
X = b.solve()


"""
The denoised estimate of the image is just the reconstruction from the coefficient maps.
"""

imgdp = b.reconstruct().squeeze()
imgd = np.clip(crop(imgdp) + imgnl, 0, 1)


"""
Display solve time and denoising performance.
"""

print("ConvBPDNJoint solve time: %5.2f s" % b.timer.elapsed('solve'))
print("Noisy image PSNR:    %5.2f dB" % sm.psnr(img, imgn))
print("Denoised image PSNR: %5.2f dB" % sm.psnr(img, imgd))


"""
Display the reference, noisy, and denoised images.
"""

fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.imview(img, fig=fig, title='Reference')
plot.subplot(1, 3, 2)
plot.imview(imgn, fig=fig, title='Noisy')
plot.subplot(1, 3, 3)
plot.imview(imgd, fig=fig, title='CSC Result')
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
