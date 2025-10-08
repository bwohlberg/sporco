#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Single-channel CSC With Lateral Inhibition / No Self Inhibition
===============================================================

This example demonstrates solving a convolutional sparse coding problem with a greyscale signal

  $$\mathrm{argmin}_\mathbf{x} \; \frac{1}{2} \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{m} - \mathbf{s} \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_{m} \|_1 + \sum_m \boldsymbol{\omega}^T_m | \mathbf{x}_m | + \sum_m \mathbf{z}^T_m | \mathbf{x}_m | \;,$$

where $\mathbf{d}_{m}$ is the $m^{\text{th}}$ dictionary filter, $\mathbf{x}_{m}$ is the coefficient map corresponding to the $m^{\text{th}}$ dictionary filter, $\mathbf{s}$ is the input image, and $\boldsymbold{\omega}^T_m$ and $\mathbf{z}^T_m$ are inhibition weights corresponding to lateral and self inhibition, respectively (see :class:`.admm.cbpdnin.ConvBPDNInhib`).
"""


from __future__ import print_function
from builtins import input

import numpy as np

from sporco import util
from sporco import signal
from sporco import plot
import sporco.metric as sm
from sporco.admm import cbpdnin


"""
Load example image.
"""

img = util.ExampleImages().image('kodim23.png', scaled=True, gray=True,
                                 idxexp=np.s_[160:416, 60:316])


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
# Repeat the dictionary twice, adding noise to each respective pair
D = np.append(D + 0.01 * np.random.randn(*D.shape),
              D + 0.01 * np.random.randn(*D.shape), axis=-1)
plot.imview(util.tiledict(D), fgsz=(10, 10))


"""
Set :class:`.admm.cbpdnin.ConvBPDNInhib` solver options.
"""

lmbda = 5e-2
mu = 5e-2
opt = cbpdnin.ConvBPDNInhib.Options({'Verbose': True, 'MaxMainIter': 200,
                                     'RelStopTol': 5e-3, 'AuxVarObj': False})


"""
Initialise and run CSC solver.
"""

# Create the Ng x M grouping matrix, where Ng is the number of groups,
# and M is the number of dictionary elements. A non-zero entry at
# Wg(n, m), means that element m belongs to group n. Our dictionary
# was repeated contiguously, so elements i and i + 36 are paired for
# i = 0, ..., 35.
Wg = np.append(np.eye(36), np.eye(36), axis=-1)
# We additionally choose a rectangular inhibition window of sample
# diameter 12.
b = cbpdnin.ConvBPDNInhib(D, sh, Wg, 12, ('boxcar'),
                          lmbda, mu, None, opt, dimK=0)
X = b.solve()
print("ConvBPDN solve time: %.2fs" % b.timer.elapsed('solve'))


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
Show activation of grouped elements column-wise for first four groups.  As mu is lowered, the vertical pairs should look more and more similar.  You will likely need to zoom in to see the activations clearly.
"""

fig = plot.figure(figsize=(14, 7))
for i in range(4):
    plot.subplot(2, 4, i + 1)
    plot.imview(abs(X[:, :, :, :, i]).squeeze(), cmap=plot.cm.Blues,
                title=f'X[{i}]', fig=fig)
    plot.subplot(2, 4, i + 5)
    plot.imview(abs(X[:, :, :, :, i + 36]).squeeze(), cmap=plot.cm.Blues,
                title=f'X[{i+36}]', fig=fig)
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
