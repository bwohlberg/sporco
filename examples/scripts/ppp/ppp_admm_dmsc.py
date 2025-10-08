#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Plug-and-Play ADMM Demosaicing
==============================

This example demonstrates the use of class :class:`.admm.ppp.PPP` for solving a raw image demosaicing problem via the ADMM Plug and Play Priors (PPP) algorithm :cite:`venkatakrishnan-2013-plugandplay2` :cite:`sreehari-2016-plug`.
"""


from __future__ import print_function
from builtins import input, range

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from bm3d import bm3d_rgb
try:
    # workaround for colour_demosaicing NumPy 2.0 incompatibility
    np.float_ = np.float32
    from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
except ImportError:
    have_demosaic = False
else:
    have_demosaic = True

from sporco.admm.ppp import PPP
from sporco.interp import bilinear_demosaic
from sporco import metric
from sporco import util
from sporco import plot



"""
Define demosaicing forward operator and its transpose.
"""

def A(x):
    """Map an RGB image to a single channel image with each pixel
    representing a single colour according to the colour filter array.
    """

    y = np.zeros(x.shape[0:2])
    y[1::2, 1::2] = x[1::2, 1::2, 0]
    y[0::2, 1::2] = x[0::2, 1::2, 1]
    y[1::2, 0::2] = x[1::2, 0::2, 1]
    y[0::2, 0::2] = x[0::2, 0::2, 2]
    return y


def AT(x):
    """Back project a single channel raw image to an RGB image with zeros
    at the locations of undefined samples.
    """

    y = np.zeros(x.shape + (3,))
    y[1::2, 1::2, 0] = x[1::2, 1::2]
    y[0::2, 1::2, 1] = x[0::2, 1::2]
    y[1::2, 0::2, 1] = x[1::2, 0::2]
    y[0::2, 0::2, 2] = x[0::2, 0::2]
    return y


"""
Define baseline demosaicing function. If package [colour_demosaicing](https://github.com/colour-science/colour-demosaicing) is installed, use the demosaicing algorithm of :cite:`menon-2007-demosaicing`, othewise use simple bilinear demosaicing.
"""

if have_demosaic:
    def demosaic(cfaimg):
        return demosaicing_CFA_Bayer_Menon2007(cfaimg, pattern='BGGR')
else:
    def demosaic(cfaimg):
        return bilinear_demosaic(cfaimg)


"""
Load reference image.
"""

img = util.ExampleImages().image('kodim23.png', scaled=True,
                                 idxexp=np.s_[160:416,60:316])


"""
Construct test image constructed by colour filter array sampling and adding Gaussian white noise.
"""

np.random.seed(12345)
s = A(img)
rgbshp = s.shape + (3,)  # Shape of reconstructed RGB image
rgbsz = s.size * 3       # Size of reconstructed RGB image
nsigma = 2e-2            # Noise standard deviation
sn = s + nsigma * np.random.randn(*s.shape)



"""
Define data fidelity term for PPP problem.
"""

def f(x):
    return 0.5 * np.linalg.norm((A(x) - sn).ravel())**2


"""
Define proximal operator of data fidelity term for PPP problem.
"""

def proxf(x, rho, tol=1e-3, maxit=100):
    ATA = lambda z: AT(A(z))
    ATAI = lambda z: ATA(z.reshape(rgbshp)).ravel() + rho * z.ravel()
    lop = LinearOperator((rgbsz, rgbsz), matvec=ATAI, dtype=s.dtype)
    b = AT(sn) + rho * x
    vx, cgit = cg(lop, b.ravel(), None, maxiter=maxit, rtol=tol)
    return vx.reshape(rgbshp)


"""
Define proximal operator of (implicit, unknown) regularisation term for PPP problem. In this case we use BM3D :cite:`dabov-2008-image` as the denoiser, using the [code](https://pypi.org/project/bm3d) released with :cite:`makinen-2019-exact`.
"""

bsigma = 6.1e-2  # Denoiser parameter

def proxg(x, rho):
    return bm3d_rgb(x, bsigma)


"""
Construct a baseline solution and initaliser for the PPP solution by BM3D denoising of a simple bilinear demosaicing solution. The `3 * nsigma` denoising parameter for BM3D is chosen empirically for best performance.
"""

imgb = bm3d_rgb(demosaic(sn), 3 * nsigma)


"""
Set algorithm options for PPP solver, including use of bilinear demosaiced solution as an initial solution.
"""

opt = PPP.Options({'Verbose': True, 'RelStopTol': 1e-3,
                   'MaxMainIter': 12, 'rho': 1.8e-1, 'Y0': imgb})


"""
Create solver object and solve, returning the the demosaiced image ``imgp``.
"""

b = PPP(img.shape, f, proxf, proxg, opt=opt)
imgp = b.solve()


"""
Display solve time and demosaicing performance.
"""

print("PPP ADMM solve time:        %5.2f s" % b.timer.elapsed('solve'))
print("Baseline demosaicing PSNR:  %5.2f dB" % metric.psnr(img, imgb))
print("PPP demosaicing PSNR:       %5.2f dB" % metric.psnr(img, imgp))


"""
Display reference and demosaiced images.
"""

fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                        figsize=(21, 7))
plot.imview(img, title='Reference', fig=fig, ax=ax[0])
plot.imview(imgb, title='Baseline demoisac: %.2f (dB)' %
            metric.psnr(img, imgb), fig=fig, ax=ax[1])
plot.imview(imgp, title='PPP demoisac: %.2f (dB)' %
            metric.psnr(img, imgp), fig=fig, ax=ax[2])
fig.show()


# Wait for enter on keyboard
input()
