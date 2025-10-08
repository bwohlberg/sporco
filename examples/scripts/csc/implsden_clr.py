#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
Impulse Noise Restoration via CSC
=================================

This example demonstrates the removal of salt & pepper noise from a colour image using convolutional sparse coding with a colour dictionary :cite:`wohlberg-2017-sporco`,

  $$\mathrm{argmin}_\mathbf{x} \; (1/2) \sum_c \left\| \sum_m \mathbf{d}_{c,m} * \mathbf{x}_m -\mathbf{s}_c \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 + (\mu/2) \sum_i \sum_m \| G_i \mathbf{x}_m \|_2^2$$

where $\mathbf{d}_{c,m}$ is channel $c$ of the $m^{\text{th}}$ dictionary filter, $\mathbf{x}_m$ is the coefficient map corresponding to the $m^{\text{th}}$ dictionary filter, $\mathbf{s}_c$ is channel $c$ of the input image, and $G_i$ is an operator computing the derivative along spatial index $i$.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco import util
from sporco import signal
from sporco import plot
import sporco.metric as sm
import sporco.prox as sp
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
Load a reference image and corrupt it with 33% salt and pepper noise. (The call to ``np.random.seed`` ensures that the pseudo-random noise is reproducible.)
"""

img = util.ExampleImages().image('monarch.png', zoom=0.5, scaled=True,
                                 idxexp=np.s_[:, 160:672])
np.random.seed(12345)
imgn = signal.spnoise(img, 0.33)


"""
We use a colour dictionary. The impulse denoising problem is solved by appending some additional filters to the learned dictionary ``D0``, which is one of those distributed with SPORCO. The first of these additional components is a set of three impulse filters, one per colour channel, that will represent the impulse noise, and the second is an identical set of impulse filters that will represent the low frequency image components when used together with a gradient penalty on the coefficient maps, as discussed below.
"""

D0 = util.convdicts()['RGB:8x8x3x64']
Di = np.zeros(D0.shape[0:2] + (3, 3))
np.fill_diagonal(Di[0, 0], 1.0)
D = np.concatenate((Di, Di, D0), axis=3)


r"""
The problem is solved using class :class:`.admm.cbpdn.ConvBPDNGradReg`, which implements the form of CBPDN with an additional gradient regularization term, as defined above. The regularization parameters for the $\ell_1$ and gradient terms are ``lmbda`` and ``mu`` respectively. Setting correct weighting arrays for these regularization terms is critical to obtaining good performance. For the $\ell_1$ norm, the weights on the filters that are intended to represent the impulse noise are tuned to an appropriate value for the impulse noise density (this value sets the relative cost of representing an image feature by one of the impulses or by one of the filters in the learned dictionary), the weights on the filters that are intended to represent low frequency components are set to zero (we only want them penalised by the gradient term), and the weights of the remaining filters are set to zero. For the gradient penalty, all weights are set to zero except for those corresponding to the filters intended to represent low frequency components, which are set to unity.
"""

lmbda = 2.8e-2
mu = 3e-1
w1 = np.ones((1, 1, 1, 1, D.shape[-1]))
w1[..., 0:3] = 0.33
w1[..., 3:6] = 0.0
wg = np.zeros((D.shape[-1]))
wg[..., 3:6] = 1.0
opt = cbpdn.ConvBPDNGradReg.Options({'Verbose': True, 'MaxMainIter': 100,
                    'RelStopTol': 5e-3, 'AuxVarObj': False,
                    'L1Weight': w1, 'GradWeight': wg})


"""
Initialise the :class:`.admm.cbpdn.ConvBPDNGradReg` object and call the ``solve`` method.
"""

b = cbpdn.ConvBPDNGradReg(D, pad(imgn), lmbda, mu, opt, dimK=0)
X = b.solve()


"""
The denoised estimate of the image is just the reconstruction from all coefficient maps except those that represent the impulse noise, which is why we subtract the slice of ``X`` corresponding the impulse noise representing filters from the result of ``reconstruct``.
"""

imgdp = b.reconstruct().squeeze() - X[..., 0, 0:3].squeeze()
imgd = crop(imgdp)


"""
Keep a copy of the low-frequency component estimate from this solution for use in the next approach.
"""

imglp = X[..., 0, 3:6].squeeze()


"""
Display solve time and denoising performance.
"""

print("ConvBPDNGradReg solve time: %5.2f s" % b.timer.elapsed('solve'))
print("Noisy image PSNR:    %5.2f dB" % sm.psnr(img, imgn))
print("Denoised image PSNR: %5.2f dB" % sm.psnr(img, imgd))


"""
Display the reference, noisy, and denoised images.
"""

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(21, 7))
fig.suptitle('Method 1 Results')
plot.imview(img, ax=ax[0], title='Reference', fig=fig)
plot.imview(imgn, ax=ax[1], title='Noisy', fig=fig)
plot.imview(imgd, ax=ax[2], title='CSC Result', fig=fig)
fig.show()



r"""
The previous method gave good results, but the weight on the filter representing the impulse noise is an additional parameter that has to be tuned. This parameter can be avoided by switching to an $\ell_1$ data fidelity term instead of including dictionary filters to represent the impulse noise, as in the problem :cite:`wohlberg-2016-convolutional2`

  $$\mathrm{argmin}_{\{\mathbf{x}_m\}} \;
  \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
  \right \|_1 + \lambda \sum_m \| \mathbf{x}_m \|_1 \;.$$

Ideally we would also include a gradient penalty term to assist in the representation of the low frequency image component. While this relatively straightforward, it is a bit more complex to implement, and is omitted from this example. Instead of including a representation of the low frequency image component within the optimization, we use the low frequency component estimated by the previous example, subtracting it from the signal passed to the CSC algorithm, and adding it back to the solution of this algorithm.

An algorithm for the problem above is not included in SPORCO, but :class:`.cbpdn.ConvBPDNMaskDcpl` is easily adapted by deriving a new class that overrides two of its methods :cite:`wohlberg-2017-sporco`.
"""

class ConvRepL1L1(cbpdn.ConvBPDNMaskDcpl):

    def ystep(self):

        AXU = self.AX + self.U
        Y0 = sp.prox_l1(self.block_sep0(AXU) - self.S, (1.0/self.rho)*self.W)
        Y1 = sp.prox_l1(self.block_sep1(AXU), (self.lmbda/self.rho)*self.wl1)
        self.Y = self.block_cat(Y0, Y1)

        super(cbpdn.ConvBPDNMaskDcpl, self).ystep()


    def obfn_g0(self, Y0):

        return np.sum(np.abs(self.W * self.obfn_g0var()))



"""
Set the options for our new class.
"""

opt = ConvRepL1L1.Options({'Verbose': True, 'MaxMainIter': 200,
                    'RelStopTol': 5e-3, 'AuxVarObj': False,
                    'rho': 1e1, 'RelaxParam': 1.8})


"""
Initialise the ``ConvRepL1L1`` object and call the ``solve`` method.
"""

lmbda = 3.0e0
b = ConvRepL1L1(D0, pad(imgn) - imglp, lmbda, opt=opt, dimK=0)
X = b.solve()


"""
Reconstruct the denoised estimate.
"""

imgdp = b.reconstruct().squeeze() + imglp
imgd = crop(imgdp)


"""
Display solve time and denoising performance.
"""

print("ConvRepL1L1 solve time: %5.2f s" % b.timer.elapsed('solve'))
print("Noisy image PSNR:    %5.2f dB" % sm.psnr(img, imgn))
print("Denoised image PSNR: %5.2f dB" % sm.psnr(img, imgd))


"""
Display the reference, noisy, and denoised images.
"""

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(21, 7))
fig.suptitle('Method 2 Results')
plot.imview(img, ax=ax[0], title='Reference', fig=fig)
plot.imview(imgn, ax=ax[1], title='Noisy', fig=fig)
plot.imview(imgd, ax=ax[2], title='CSC Result', fig=fig)
fig.show()



# Wait for enter on keyboard
input()
