#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Parallel CSC with a Spatial Mask
================================

This example compares the use of :class:`.parcbpdn.ParConvBPDN` with :class:`.cbpdn.ConvBPDNMaskDcpl` for convolutional sparse coding with a spatial mask :cite:`wohlberg-2016-boundary`. The example problem is inpainting of randomly distributed corruption of a greyscale image.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.admm import tvl2
from sporco.admm import cbpdn
from sporco.admm import parcbpdn
from sporco import util
from sporco import metric
from sporco import plot


"""
Load a reference image.
"""

img = util.ExampleImages().image('monarch.png', zoom=0.25, scaled=True,
                                 gray=True, idxexp=np.s_[:, 160:672])


"""
Create random mask and apply to reference image to obtain test image. (The call to ``numpy.random.seed`` ensures that the pseudo-random noise is reproducible.)
"""

t = 0.5
np.random.seed(12345)
msk = np.random.randn(*(img.shape))
msk[np.abs(msk) > t] = 1
msk[np.abs(msk) < t] = 0
imgw = msk * img


"""
Define pad and crop functions.
"""

pn = 8
spad = lambda x:  np.pad(x, pn, mode='symmetric')
zpad = lambda x:  np.pad(x, pn, mode='constant')
crop = lambda x: x[pn:-pn, pn:-pn]


"""
Construct padded mask and test image.
"""

mskp = zpad(msk)
imgwp = spad(imgw)


r"""
$\ell_2$-TV denoising with a spatial mask as a non-linear lowpass filter. The highpass component is the difference between the test image and the lowpass component, multiplied by the mask for faster convergence of the convolutional sparse coding (see :cite:`wohlberg-2017-convolutional3`).
"""

lmbda = 0.05
opt = tvl2.TVL2Denoise.Options({'Verbose': False, 'MaxMainIter': 200,
                                'DFidWeight': mskp, 'gEvalY': False,
                                'AutoRho': {'Enabled': True}})
b = tvl2.TVL2Denoise(imgwp, lmbda, opt)
sl = b.solve()
sh = mskp * (imgwp - sl)


"""
Load dictionary.
"""

D = util.convdicts()['G:12x12x216']
plot.imview(util.tiledict(D), fgsz=(7, 7))

lmbda = 2e-2

"""
The RelStopTol was chosen for the two different methods to stop with similar functional values
"""

"""
Initialise and run serial CSC solver using masked decoupling :cite:`heide-2015-fast`.
"""

opt = cbpdn.ConvBPDNMaskDcpl.Options({'Verbose': True, 'MaxMainIter': 200,
                            'HighMemSolve': True, 'RelStopTol': 5e-2,
                            'AuxVarObj': False, 'RelaxParam': 1.8,
                            'rho': 5e1*lmbda + 1e-1, 'AutoRho':
                            {'Enabled': False, 'StdResiduals': False}})
b = cbpdn.ConvBPDNMaskDcpl(D, sh, lmbda, mskp, opt=opt)
X = b.solve()


"""
Initialise and run parallel CSC solver using an ADMM dictionary partition :cite:`skau-2018-fast`.
"""

opt_par = parcbpdn.ParConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                            'HighMemSolve': True, 'RelStopTol': 1e-2,
                            'AuxVarObj': False, 'RelaxParam': 1.8,
                            'rho': 5e1*lmbda + 1e-1, 'alpha': 1.5,
                            'AutoRho': {'Enabled': False,
                            'StdResiduals': False}})
b_par = parcbpdn.ParConvBPDN(D, sh, lmbda, mskp, opt=opt_par)
X_par = b_par.solve()


"""
Report runtimes of different methods of solving the same problem.
"""

print("ConvBPDNMaskDcpl solve time: %.2fs" % b.timer.elapsed('solve_wo_rsdl'))
print("ParConvBPDN solve time: %.2fs" % b_par.timer.elapsed('solve_wo_rsdl'))
print("ParConvBPDN was %.2f times faster than ConvBPDNMaskDcpl\n" %
      (b.timer.elapsed('solve_wo_rsdl')/b_par.timer.elapsed('solve_wo_rsdl')))


"""
Reconstruct images from sparse representations.
"""

imgr = crop(sl + b.reconstruct().squeeze())
imgr_par = crop(sl + b_par.reconstruct().squeeze())


"""
Report performances of different methods of solving the same problem.
"""

print("Corrupted image PSNR: %5.2f dB" % metric.psnr(img, imgw))
print("Serial Reconstruction PSNR: %5.2f dB" % metric.psnr(img, imgr))
print("Parallel Reconstruction PSNR: %5.2f dB\n" % metric.psnr(img, imgr_par))


"""
Display reference, test, and reconstructed images
"""

fig = plot.figure(figsize=(14, 14))
plot.subplot(2, 2, 1)
plot.imview(img, fig=fig, title='Reference Image')
plot.subplot(2, 2, 2)
plot.imview(imgw, fig=fig, title=('Corrupted Image PSNR: %5.2f dB' %
            metric.psnr(img, imgw)))
plot.subplot(2, 2, 3)
plot.imview(imgr, fig=fig, title=('Serial reconstruction PSNR: %5.2f dB' %
            metric.psnr(img, imgr)))
plot.subplot(2, 2, 4)
plot.imview(imgr_par, fig=fig, title=('Parallel reconstruction PSNR: %5.2f dB' %
            metric.psnr(img, imgr_par)))
fig.show()


"""
Display lowpass component and sparse representation
"""

fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.imview(sl, fig=fig, cmap=plot.cm.Blues, title='Lowpass component')
plot.subplot(1, 3, 2)
plot.imview(np.squeeze(np.sum(abs(X), axis=b.cri.axisM)), fig=fig,
            cmap=plot.cm.Blues, title='Serial sparse representation')
plot.subplot(1, 3, 3)
plot.imview(np.squeeze(np.sum(abs(X_par), axis=b.cri.axisM)), fig=fig,
            cmap=plot.cm.Blues, title='Parallel sparse representation')
fig.show()


# Wait for enter on keyboard
input()
