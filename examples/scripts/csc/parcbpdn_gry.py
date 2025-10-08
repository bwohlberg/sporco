#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Parallel Single-channel CSC
===========================

This example compares the use of :class:`.parcbpdn.ParConvBPDN` with :class:`.admm.cbpdn.ConvBPDN` solving a convolutional sparse coding problem with a greyscale signal

  $$\mathrm{argmin}_\mathbf{x} \; \frac{1}{2} \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{m} - \mathbf{s} \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_{m} \|_1 \;,$$

where $\mathbf{d}_{m}$ is the $m^{\text{th}}$ dictionary filter, $\mathbf{x}_{m}$ is the coefficient map corresponding to the $m^{\text{th}}$ dictionary filter, and $\mathbf{s}$ is the input image.
"""

from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco import util
from sporco import signal
from sporco import plot
import sporco.metric as sm
from sporco.admm import cbpdn
from sporco.admm import parcbpdn


"""
Load example image.
"""

img = util.ExampleImages().image('kodim23.png', zoom=1.0, scaled=True,
                                 gray=True, idxexp=np.s_[160:416, 60:316])

"""
Highpass filter example image.
"""

npd = 16
fltlmbd = 10
sl, sh = signal.tikhonov_filter(img, fltlmbd, npd)


"""
Load dictionary and display it.
"""

D = util.convdicts()['G:12x12x216']
plot.imview(util.tiledict(D), fgsz=(7, 7))

lmbda = 5e-2

"""
The RelStopTol option was chosen for the two different methods to stop with similar functional values
"""

"""
Initialise and run standard serial CSC solver using ADMM with an equality constraint :cite:`wohlberg-2014-efficient`.
"""

opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                              'RelStopTol': 5e-3, 'AuxVarObj': False,
                              'AutoRho': {'Enabled': False}})
b = cbpdn.ConvBPDN(D, sh, lmbda, opt=opt, dimK=0)
X = b.solve()


"""
Initialise and run parallel CSC solver using ADMM dictionary partition method :cite:`skau-2018-fast`.
"""

opt_par = parcbpdn.ParConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                            'RelStopTol': 1e-2, 'AuxVarObj': False, 'AutoRho':
                                        {'Enabled': False}, 'alpha': 2.5})
b_par = parcbpdn.ParConvBPDN(D, sh, lmbda, opt=opt_par, dimK=0)
X_par = b_par.solve()


"""
Report runtimes of different methods of solving the same problem.
"""

print("ConvBPDN solve time: %.2fs" % b.timer.elapsed('solve_wo_rsdl'))
print("ParConvBPDN solve time: %.2fs" % b_par.timer.elapsed('solve_wo_rsdl'))
print("ParConvBPDN was %.2f times faster than ConvBPDN\n" %
      (b.timer.elapsed('solve_wo_rsdl')/b_par.timer.elapsed('solve_wo_rsdl')))

"""
Reconstruct images from sparse representations.
"""

shr = b.reconstruct().squeeze()
imgr = sl + shr

shr_par = b_par.reconstruct().squeeze()
imgr_par = sl + shr_par


"""
Report performances of different methods of solving the same problem.
"""

print("Serial reconstruction PSNR: %.2fdB" % sm.psnr(img, imgr))
print("Parallel reconstruction PSNR: %.2fdB\n" % sm.psnr(img, imgr_par))


"""
Display original and reconstructed images.
"""

fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.imview(img, title='Original', fig=fig)
plot.subplot(1, 3, 2)
plot.imview(imgr, title=('Serial Reconstruction PSNR:  %5.2f dB' %
            sm.psnr(img, imgr)), fig=fig)
plot.subplot(1, 3, 3)
plot.imview(imgr_par, title=('Parallel Reconstruction PSNR:  %5.2f dB' %
            sm.psnr(img, imgr_par)), fig=fig)
fig.show()


"""
Display low pass component and sum of absolute values of coefficient maps of highpass component.
"""

fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.imview(sl, title='Lowpass component', fig=fig)
plot.subplot(1, 3, 2)
plot.imview(np.sum(abs(X), axis=b.cri.axisM).squeeze(),
            cmap=plot.cm.Blues, title='Serial Sparse Representation',
            fig=fig)
plot.subplot(1, 3, 3)
plot.imview(np.sum(abs(X_par), axis=b.cri.axisM).squeeze(),
            cmap=plot.cm.Blues, title='Parallel Sparse Representation',
            fig=fig)
fig.show()

# Wait for enter on keyboard
input()
