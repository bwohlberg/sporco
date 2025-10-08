#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Gaussian White Noise Restoration via CSC
========================================

This example demonstrates the removal of Gaussian white noise from a colour image using convolutional sparse coding :cite:`wohlberg-2016-convolutional`,

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \sum_c \left\| \sum_m \mathbf{d}_{m} * \mathbf{x}_{c,m} -\mathbf{s}_c \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 + \mu \| \{ \mathbf{x}_{c,m} \} \|_{2,1}$$

where $\mathbf{d}_m$ is the $m^{\text{th}}$ dictionary filter, $\mathbf{x}_{c,m}$ is the coefficient map corresponding to the $c^{\text{th}}$ colour band and $m^{\text{th}}$ dictionary filter, and $\mathbf{s}_c$ is colour band $c$ of the input image.
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
from sporco.cupy import (cupy_enabled, np2cp, cp2np, select_device_by_load,
                         gpu_info)
from sporco.cupy.admm import cbpdn


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
                                 idxexp=np.s_[:, 160:672])
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
W = fft.irfftn(np.conj(fft.rfftn(D[..., np.newaxis, :], imgnph.shape[0:2],
               (0, 1))) * fft.rfftn(imgnph[..., np.newaxis], None, (0, 1)),
               imgnph.shape[0:2], (0, 1))
W = 1.0/(np.maximum(np.abs(W), 1e-8))

lmbda = 1.5e-2
mu = 2.7e-1

opt = cbpdn.ConvBPDNJoint.Options({'Verbose': True, 'MaxMainIter': 250,
            'HighMemSolve': True, 'RelStopTol': 3e-3, 'AuxVarObj': False,
            'L1Weight': np2cp(W), 'AutoRho': {'Enabled': False},
            'rho': 1e3*lmbda})


"""
Initialise a ``sporco.cupy`` version of a :class:`.admm.cbpdn.ConvBPDNJoint` object and call the ``solve`` method.
"""

if not cupy_enabled():
    print('CuPy/GPU device not available: running without GPU acceleration\n')
else:
    id = select_device_by_load()
    info = gpu_info()
    if info:
        print('Running on GPU %d (%s)\n' % (id, info[id].name))

b = cbpdn.ConvBPDNJoint(np2cp(D), np2cp(pad(imgnh)), lmbda, mu, opt, dimK=0)
X = cp2np(b.solve())


"""
The denoised estimate of the image is just the reconstruction from the coefficient maps.
"""

imgdp = cp2np(b.reconstruct().squeeze())
imgd = np.clip(crop(imgdp) + imgnl, 0, 1)


"""
Display solve time and denoising performance.
"""

print("ConvBPDNJoint solve time: %5.2f s" % b.timer.elapsed('solve'))
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
ObjFun = [float(x) for x in its.ObjFun]
plot.plot(ObjFun, xlbl='Iterations', ylbl='Functional')


"""
Plot evolution of ADMM residuals and ADMM penalty parameter.
"""

PrimalRsdl = [float(x) for x in its.PrimalRsdl]
DualRsdl = [float(x) for x in its.DualRsdl]
plot.plot(np.vstack((PrimalRsdl, DualRsdl)).T,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'])
plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter')



# Wait for enter on keyboard
input()
