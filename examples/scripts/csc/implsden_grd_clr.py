#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Impulse Noise Restoration via CSC
=================================

This example demonstrates the removal of salt & pepper noise from a colour image using convolutional sparse coding, with a colour dictionary :cite:`wohlberg-2016-convolutional` and with an $\ell_1$ data fidelity term, an $\ell_1$ regularisation term, and an additional gradient regularization term :cite:`wohlberg-2016-convolutional2`

  $$\mathrm{argmin}_\mathbf{x} \; \sum_c \left\| \sum_m \mathbf{d}_{c,m} * \mathbf{x}_m -\mathbf{s}_c \right\|_1 + \lambda \sum_m \| \mathbf{x}_m \|_1 + (\mu/2) \sum_i \sum_m \| G_i \mathbf{x}_m \|_2^2$$

where $\mathbf{d}_{c,m}$ is channel $c$ of the $m^{\text{th}}$ dictionary filter, $\mathbf{x}_m$ is the coefficient map corresponding to the $m^{\text{th}}$ dictionary filter, $\mathbf{s}_c$ is channel $c$ of the input image, and $G_i$ is an operator computing the derivative along spatial index $i$.

This example uses the GPU accelerated version of :mod:`.admm.cbpdn` within the :mod:`sporco.cupy` subpackage.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco import util
from sporco import signal
from sporco import plot
from sporco.metric import psnr
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


"""
Load a reference image and corrupt it with 33% salt and pepper noise. (The call to ``np.random.seed`` ensures that the pseudo-random noise is reproducible.)
"""

img = util.ExampleImages().image('monarch.png', zoom=0.5, scaled=True,
                                 idxexp=np.s_[:, 160:672])
np.random.seed(12345)
imgn = signal.spnoise(img, 0.33)


"""
We use a colour dictionary. The impulse denoising problem is solved by appending some additional filters to the learned dictionary ``D0``, which is one of those distributed with SPORCO. These additional components consist of a set of three impulse filters, one per colour channel, that will represent the low frequency image components when used together with a gradient penalty on the coefficient maps, as discussed below.
"""

D0 = util.convdicts()['RGB:8x8x3x64']
Di = np.zeros(D0.shape[0:2] + (3, 3), dtype=np.float32)
np.fill_diagonal(Di[0, 0], 1.0)
D = np.concatenate((Di, D0), axis=3)


r"""
The problem is solved using class :class:`.admm.cbpdn.ConvL1L1Grd`, which implements a convolutional sparse coding problem with an $\ell_1$ data fidelity term, an $\ell_1$ regularisation term, and an additional gradient regularization term :cite:`wohlberg-2016-convolutional2`, as defined above. The regularization parameters for the $\ell_1$ and gradient terms are ``lmbda`` and ``mu`` respectively. Setting correct weighting arrays for these regularization terms is critical to obtaining good performance. For the $\ell_1$ norm, the weights on the filters that are intended to represent low frequency components are set to zero (we only want them penalised by the gradient term), and the weights of the remaining filters are set to zero. For the gradient penalty, all weights are set to zero except for those corresponding to the filters intended to represent low frequency components, which are set to unity.
"""

lmbda = 3e0
mu = 2e1
w1 = np.ones((1, 1, 1, 1, D.shape[-1]), dtype=np.float32)
w1[..., 0:3] = 0.0
wg = np.zeros((D.shape[-1]), dtype=np.float32)
wg[..., 0:3] = 1.0

opt = cbpdn.ConvL1L1Grd.Options({'Verbose': True, 'MaxMainIter': 200,
                                'RelStopTol': 5e-3, 'AuxVarObj': False,
                                'rho': 4e1, 'RelaxParam': 1.8,
                                'L1Weight': np2cp(w1),
                                'GradWeight': np2cp(wg)})


"""
Initialise the :class:`.admm.cbpdn.ConvL1L1Grd` object and call the ``solve`` method.
"""

if not cupy_enabled():
    print('CuPy/GPU device not available: running without GPU acceleration\n')
else:
    id = select_device_by_load()
    info = gpu_info()
    if info:
        print('Running on GPU %d (%s)\n' % (id, info[id].name))

b = cbpdn.ConvL1L1Grd(np2cp(D), np2cp(pad(imgn)), lmbda, mu, opt=opt, dimK=0)
X = cp2np(b.solve())


"""
The denoised estimate of the image is just the reconstruction from all coefficient maps.
"""

imgdp = cp2np(b.reconstruct().squeeze())
imgd = crop(imgdp)


"""
Display solve time and denoising performance.
"""

print("ConvL1L1Grd solve time: %5.2f s" % b.timer.elapsed('solve'))
print("Noisy image PSNR:    %5.2f dB" % psnr(img, imgn))
print("Denoised image PSNR: %5.2f dB" % psnr(img, imgd))


"""
Display the reference, noisy, and denoised images.
"""

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(21, 7))
fig.suptitle('ConvL1L1Grd Results')
plot.imview(img, ax=ax[0], title='Reference', fig=fig)
plot.imview(imgn, ax=ax[1], title='Noisy', fig=fig)
plot.imview(imgd, ax=ax[2], title='CSC Result', fig=fig)
fig.show()


"""
Display the low frequency image component.
"""

plot.imview(X[..., 0, 0:3].squeeze(), title='Low frequency component')



# Wait for enter on keyboard
input()
