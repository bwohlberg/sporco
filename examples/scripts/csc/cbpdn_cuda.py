#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
CUDA Convolutional Sparse Coding
================================

This example demonstrates the use of the interface to the CUDA CSC solver extension package, with a test for the availablity of a GPU that runs the Python version of the ConvBPDN solver if one is not available, or if the extension package is not installed.
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
from sporco import cuda
from sporco.admm import cbpdn


# If running in a notebook, try to use wurlitzer so that output from the CUDA
# code will be properly captured in the notebook.
sys_pipes = util.notebook_system_output()


"""
Load example image.
"""

img = util.ExampleImages().image('barbara.png', scaled=True, gray=True,
                                 idxexp=np.s_[10:522, 100:612])


"""
Highpass filter example image.
"""

npd = 16
fltlmbd = 20
sl, sh = signal.tikhonov_filter(img, fltlmbd, npd)


"""
Load dictionary.
"""

D = util.convdicts()['G:12x12x36']


"""
Set up :class:`.admm.cbpdn.ConvBPDN` options.
"""

lmbda = 1e-2
opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 250,
                    'HighMemSolve': True, 'RelStopTol': 5e-3,
                    'AuxVarObj': False})


"""
If GPU available, run CUDA ConvBPDN solver, otherwise run standard Python version.
"""

if cuda.device_count() > 0:
    print('%s GPU found: running CUDA solver' % cuda.device_name())
    tm = util.Timer()
    with sys_pipes(), util.ContextTimer(tm):
        X = cuda.cbpdn(D, sh, lmbda, opt)
    t = tm.elapsed()
else:
    print('GPU not found: running Python solver')
    c = cbpdn.ConvBPDN(D, sh, lmbda, opt)
    X = c.solve().squeeze()
    t = c.timer.elapsed('solve')
print('Solve time: %.2f s' % t)


"""
Reconstruct the image from the sparse representation.
"""

shr = np.sum(fft.fftconv(D, X, axes=(0, 1)), axis=2)
imgr = sl + shr
print("Reconstruction PSNR: %.2fdB\n" % metric.psnr(img, imgr))


"""
Display representation and reconstructed image.
"""

fig = plot.figure(figsize=(14, 14))
plot.subplot(2, 2, 1)
plot.imview(sl, title='Lowpass component', fig=fig)
plot.subplot(2, 2, 2)
plot.imview(np.sum(abs(X), axis=2).squeeze(),
            cmap=plot.cm.Blues, title='Main representation', fig=fig)
plot.subplot(2, 2, 3)
plot.imview(imgr, title='Reconstructed image', fig=fig)
plot.subplot(2, 2, 4)
plot.imview(imgr - img, fltscl=True, title='Reconstruction difference',
            fig=fig)
fig.show()


# Wait for enter on keyboard
input()
