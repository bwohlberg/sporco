#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Impulse Noise Restoration via CSC
=================================

This example demonstrates the removal of salt & pepper noise from a hyperspectral image using convolutional sparse coding, with a product dictionary :cite:`garcia-2018-convolutional2` and with an :math:`\ell_1` data fidelity term, an :math:`\ell_1` regularisation term, and an additional gradient regularization term :cite:`wohlberg-2016-convolutional2`

  $$\mathrm{argmin}_X \; \left\| D X B^T - S \right\|_1 + \lambda \| X \|_1 + (\mu / 2) \sum_i \| G_i X \|_2^2$$

where $D$ is a convolutional dictionary, $B$ is a standard dictionary, $G_i$ is an operator that computes the gradient along array axis $i$, and $S$ is a multi-channel input image.

This example uses the GPU accelerated version of :mod:`.admm.pdcsc` within the :mod:`sporco.cupy` subpackage.
"""


from __future__ import print_function
from builtins import input

import os.path
import tempfile
import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np
import scipy.io as sio

from sporco import util
from sporco import signal
from sporco import plot
from sporco.metric import psnr
from sporco.cupy import (cupy_enabled, np2cp, cp2np, select_device_by_load,
                         gpu_info)
from sporco.cupy.admm import pdcsc
from sporco.dictlrn import bpdndl


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
Load a reference hyperspectral image and corrupt it with 33% salt and pepper noise. (The call to ``np.random.seed`` ensures that the pseudo-random noise is reproducible.)
"""

pth = os.path.join(tempfile.gettempdir(), 'Indian_pines.mat')
if not os.path.isfile(pth):
    url = 'http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat'
    vid = util.netgetdata(url)
    f = open(pth, 'wb')
    f.write(vid.read())
    f.close()

img = sio.loadmat(pth)['indian_pines'].astype(np.float32)
img = img[16:-17, 16:-17, 0:200:2]
img /= img.max()

np.random.seed(12345)
imgn = signal.spnoise(img, 0.33)


"""
We use a product dictionary :cite:`garcia-2018-convolutional2` constructed from a single-channel convolutional dictionary for the spatial axes of the image, and a standard (non-convolutional) dictionary for the spectral axis of the image. The impulse denoising problem is solved by appending an additional filter to the learned dictionary ``D0``, which is one of those distributed with SPORCO. This additional component consist of an impulse filters that will represent the low frequency image components when used together with a gradient penalty on the coefficient maps, as discussed below. The spectral axis dictionary is learned from the noise-free ground-truth image since the primary purpose of this script is as a code usage example: in a real application, this dictionary would be estimated from a relevant noise-free image.
"""

D0 = util.convdicts()['G:8x8x32']
Di = np.zeros(D0.shape[0:2] + (1,), dtype=np.float32)
Di[0, 0] = 1.0
D = np.concatenate((Di, D0), axis=2)

S = img.reshape((-1, img.shape[-1])).T
np.random.seed(12345)
B0 = np.random.randn(S.shape[0], 20)
lmbda = 0.02
opt = bpdndl.BPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 100,
                                    'BPDN': {'rho': 10.0*lmbda + 0.1},
                                    'CMOD': {'rho': S.shape[1] / 2e2}})

d = bpdndl.BPDNDictLearn(B0, S, lmbda, opt)
B = d.solve()


r"""
The problem is solved using class :class:`.admm.pdcsc.ConvProdDictL1L1Grd`, which implements a convolutional sparse coding problem with a product dictionary :cite:`garcia-2018-convolutional2`, an :math:`\ell_1` data fidelity term, an :math:`\ell_1` regularisation term, and an additional gradient regularization term :cite:`wohlberg-2016-convolutional2`, as defined above. The regularization parameters for the $\ell_1$ and gradient terms are ``lmbda`` and ``mu`` respectively. Setting correct weighting arrays for these regularization terms is critical to obtaining good performance. For the $\ell_1$ norm, the weights on the filters that are intended to represent low frequency components are set to zero (we only want them penalised by the gradient term), and the weights of the remaining filters are set to zero. For the gradient penalty, all weights are set to zero except for those corresponding to the filters intended to represent low frequency components, which are set to unity.
"""

lmbda = 1.4e0
mu = 9e0


r"""
Set up weights for the $\ell_1$ norm to disable regularization of the coefficient map corresponding to the impulse filter.
"""

wl1 = np.ones((1,)*4 + (D.shape[2],), dtype=np.float32)
wl1[..., 0] = 0.0


r"""
Set of weights for the $\ell_2$ norm of the gradient to disable regularization of all coefficient maps except for the one corresponding to the impulse filter.
"""

wgr = np.zeros((D.shape[2]), dtype=np.float32)
wgr[0] = 1.0


"""
Set :class:`.admm.pdcsc.ConvProdDictL1L1Grd` solver options.
"""

opt = pdcsc.ConvProdDictL1L1Grd.Options(
    {'Verbose': True, 'MaxMainIter': 100, 'RelStopTol': 5e-3,
     'AuxVarObj': False, 'rho': 1e1, 'RelaxParam': 1.8,
     'L1Weight': np2cp(wl1), 'GradWeight': np2cp(wgr)})


"""
Initialise the :class:`.admm.pdcsc.ConvProdDictL1L1Grd` object and call the ``solve`` method.
"""

if not cupy_enabled():
    print('CuPy/GPU device not available: running without GPU acceleration\n')
else:
    id = select_device_by_load()
    info = gpu_info()
    if info:
        print('Running on GPU %d (%s)\n' % (id, info[id].name))

b = pdcsc.ConvProdDictL1L1Grd(np2cp(D), np2cp(B), np2cp(pad(imgn)),
                              lmbda, mu, opt=opt, dimK=0)
X = cp2np(b.solve())


"""
The denoised estimate of the image is just the reconstruction from all coefficient maps.
"""

imgdp = cp2np(b.reconstruct().squeeze())
imgd = crop(imgdp)


"""
Display solve time and denoising performance.
"""

print("ConvProdDictL1L1Grd solve time: %5.2f s" % b.timer.elapsed('solve'))
print("Noisy image PSNR:    %5.2f dB" % psnr(img, imgn))
print("Denoised image PSNR: %5.2f dB" % psnr(img, imgd))


"""
Display the reference, noisy, and denoised images.
"""

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(21, 7))
fig.suptitle('ConvProdDictL1L1GrdJoint Results (false colour, '
             'bands 10, 20, 30)')
plot.imview(img[..., 10:40:10], title='Reference', ax=ax[0], fig=fig)
plot.imview(imgn[..., 10:40:10], title='Noisy', ax=ax[1], fig=fig)
plot.imview(imgd[..., 10:40:10], title='Denoised', ax=ax[2], fig=fig)
fig.show()


"""
Get iterations statistics from solver object and plot functional value, ADMM primary and dual residuals, and automatically adjusted ADMM penalty parameter against the iteration number.
"""

its = b.getitstat()
ObjFun = [float(x) for x in its.ObjFun]
PrimalRsdl = [float(x) for x in its.PrimalRsdl]
DualRsdl = [float(x) for x in its.DualRsdl]
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(np.vstack((PrimalRsdl, DualRsdl)).T,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'], fig=fig)
plot.subplot(1, 3, 3)
plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter', fig=fig)
fig.show()



# Wait for enter on keyboard
input()
