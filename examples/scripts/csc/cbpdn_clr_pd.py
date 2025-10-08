#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Multi-channel CSC
=================

This example demonstrates convolutional sparse coding of a colour signal with a product dictionary :cite:`garcia-2018-convolutional2`

  $$\mathrm{argmin}_X \; \left\| D X B^T - S \right\|_1 + \lambda \| X \|_1$$

where $D$ is a convolutional dictionary, $B$ is a standard dictionary, and $S$ is a multi-channel input image.

This example uses the GPU accelerated version of :mod:`.admm.pdcsc` within the :mod:`sporco.cupy` subpackage.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco import util
from sporco import signal
from sporco import fft
from sporco import linalg
from sporco import plot
from sporco import metric
from sporco.cupy import (cupy_enabled, np2cp, cp2np, select_device_by_load,
                         gpu_info)
from sporco.cupy.admm import pdcsc
from sporco.dictlrn import bpdndl


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
slc, shc = signal.tikhonov_filter(img, fltlmbd, npd)


"""
Load greyscale convolutional dictionary.
"""

D = util.convdicts()['G:8x8x64']


"""
Learn a standard dictionary $B$ to represent all pixel colours in the example image. Since the standard dictionary is a $3 \times 6$ matrix, the sparse representation $X$ has 6 pseudo-channels, which are converted to the 3 channels of the example image by right-multiplication by the dictionary $B$, giving $XB$.
"""

S = shc.reshape((-1, shc.shape[-1])).T
np.random.seed(12345)
B0 = np.random.randn(S.shape[0], 6)
lmbda = 1e-2
opt = bpdndl.BPDNDictLearn.Options(
    {'Verbose': True, 'MaxMainIter': 100,
     'BPDN': {'rho': 10.0*lmbda, 'AutoRho': {'Enabled': False}},
     'CMOD': {'rho': S.shape[1] / 3e2, 'AutoRho': {'Enabled': False}}})

d = bpdndl.BPDNDictLearn(B0, S, lmbda, opt)
B = d.solve()


"""
Set :class:`.pdcsc.ConvProdDictBPDN` solver options.
"""

lmbda = 1e-1
opt = pdcsc.ConvProdDictBPDN.Options({'Verbose': True, 'MaxMainIter': 100,
                                      'RelStopTol': 5e-3, 'AuxVarObj': False})


"""
Initialise and run CSC solver.
"""

if not cupy_enabled():
    print('CuPy/GPU device not available: running without GPU acceleration\n')
else:
    id = select_device_by_load()
    info = gpu_info()
    if info:
        print('Running on GPU %d (%s)\n' % (id, info[id].name))

b = pdcsc.ConvProdDictBPDN(np2cp(D), np2cp(B), np2cp(shc), lmbda, opt, dimK=0)
X = cp2np(b.solve())
print("ConvProdDictBPDN solve time: %.2fs" % b.timer.elapsed('solve'))


"""
Compute partial and full reconstructions from sparse representation $X$ with respect to convolutional dictionary $D$ and standard dictionary $B$. The partial reconstructions are $DX$ and $XB$, and the full reconstruction is $DXB$.
"""

DX = fft.fftconv(D[..., np.newaxis, np.newaxis, :], X, axes=(0, 1))
XB = linalg.dot(B, X, axis=2)
shr = cp2np(b.reconstruct().squeeze())
imgr = slc + shr
print("Reconstruction PSNR: %.2fdB\n" % metric.psnr(img, imgr))


"""
Display original and reconstructed images.
"""

gamma = lambda x, g: np.sign(x) * (np.abs(x)**g)

fig, ax = plot.subplots(nrows=2, ncols=2, figsize=(14, 14))
plot.imview(img, title='Original image', ax=ax[0, 0], fig=fig)
plot.imview(slc, title='Lowpass component', ax=ax[0, 1], fig=fig)
plot.imview(imgr, title='Reconstructed image', ax=ax[1, 0], fig=fig)
plot.imview(gamma(shr, 0.6), title='Reconstructed highpass component',
            ax=ax[1, 1], fig=fig)
fig.show()


"""
Display sparse representation components as sums of absolute values of coefficient maps for $X$, $DX$, and $XB$.
"""

fig, ax = plot.subplots(nrows=2, ncols=2, figsize=(14, 14))
plot.imview(gamma(np.sum(abs(X[..., 0:3, :, :]), axis=b.cri.axisM).squeeze(),
                  0.5), title='X (false colour, bands 0, 1, 2)', ax=ax[0, 0],
            fig=fig)
plot.imview(gamma(np.sum(abs(X[..., 3:6, :, :]), axis=b.cri.axisM).squeeze(),
                  0.5), title='X (false colour, bands 3, 4, 5)', ax=ax[0, 1],
            fig=fig)
plot.imview(gamma(np.sum(abs(DX[..., 0:3, :, :]), axis=b.cri.axisM).squeeze(),
                  0.5), title='DX (false colour, bands 0, 1, 2)', ax=ax[1, 0],
            fig=fig)
plot.imview(gamma(np.sum(abs(DX[..., 3:6, :, :]), axis=b.cri.axisM).squeeze(),
                  0.5), title='DX (false colour, bands 3, 4, 5)', ax=ax[1, 1],
            fig=fig)
fig.show()

plot.imview(gamma(np.sum(abs(XB), axis=b.cri.axisM).squeeze(), 0.5),
            title='XB', fgsz=(6.4, 6.4))


"""
Get iterations statistics from solver object and plot functional value, ADMM primary and dual residuals, and automatically adjusted ADMM penalty parameter against the iteration number.
"""

its = b.getitstat()
ObjFun = [float(x) for x in its.ObjFun]
PrimalRsdl = [float(x) for x in its.PrimalRsdl]
DualRsdl = [float(x) for x in its.DualRsdl]
fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(20, 5))
plot.plot(ObjFun, xlbl='Iterations', ylbl='Functional', ax=ax[0], fig=fig)
plot.plot(np.vstack((PrimalRsdl, DualRsdl)).T,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'], ax=ax[1], fig=fig)
plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter', ax=ax[2],
          fig=fig)
fig.show()


# Wait for enter on keyboard
input()
