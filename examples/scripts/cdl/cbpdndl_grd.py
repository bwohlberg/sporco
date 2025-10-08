#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Convolutional Dictionary Learning
=================================

This example demonstrates the use of :class:`.dictlrn.DictLearn` to construct a dictionary learning algorithm with the flexibility of choosing the sparse coding and dictionary update classes. In this case they are :class:`.cbpdn.ConvBPDNGradReg` and :func:`.admm.ccmod.ConvCnstrMOD` respectively, so the resulting dictionary learning algorithm is not equivalent to :class:`.dictlrn.cbpdndl.ConvBPDNDictLearn`. Sparse coding with a CBPDN variant that includes a gradient regularization term on one of the coefficient maps :cite:`wohlberg-2016-convolutional2` enables CDL without the need for the usual lowpass/highpass filtering as a pre-processing of the training images.
"""


from __future__ import division
from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.admm import cbpdn
from sporco.admm import ccmod
from sporco.dictlrn import dictlrn
from sporco import cnvrep
from sporco import util
from sporco import plot


"""
Load training images.
"""

exim = util.ExampleImages(scaled=True, zoom=0.5, gray=True)
img1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
img2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
img3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
S = np.stack((img1, img2, img3), axis=2)


"""
Construct initial dictionary.
"""

np.random.seed(12345)
D0 = np.random.randn(8, 8, 64)


"""
Construct object representing problem dimensions.
"""

cri = cnvrep.CDU_ConvRepIndexing(D0.shape, S)


r"""
Set up weights for the $\ell_1$ norm to disable regularization of the coefficient map corresponding to the impulse filter.
"""

wl1 = np.ones((1,)*4 + (D0.shape[2:]), dtype=np.float32)
wl1[..., 0] = 0.0


r"""
Set of weights for the $\ell_2$ norm of the gradient to disable regularization of all coefficient maps except for the one corresponding to the impulse filter.
"""

wgr = np.zeros((D0.shape[2]), dtype=np.float32)
wgr[0] = 1.0


"""
Define X and D update options.
"""

lmbda = 0.1
mu = 0.5
optx = cbpdn.ConvBPDNGradReg.Options({'Verbose': False, 'MaxMainIter': 1,
            'rho': 20.0*lmbda + 0.5, 'AutoRho': {'Period': 10,
            'AutoScaling': False, 'RsdlRatio': 10.0, 'Scaling': 2.0,
            'RsdlTarget': 1.0}, 'HighMemSolve': True, 'AuxVarObj': False,
            'L1Weight': wl1, 'GradWeight': wgr})
optd = ccmod.ConvCnstrMODOptions({'Verbose': False, 'MaxMainIter': 1,
            'rho': 5.0*cri.K, 'AutoRho': {'Period': 10, 'AutoScaling': False,
            'RsdlRatio': 10.0, 'Scaling': 2.0, 'RsdlTarget': 1.0}},
            method='cns')


"""
Normalise dictionary according to dictionary Y update options.
"""

D0n = cnvrep.Pcn(D0, D0.shape, cri.Nv, dimN=2, dimC=0, crp=True,
                 zm=optd['ZeroMean'])


"""
Update D update options to include initial values for Y and U.
"""

optd.update({'Y0': cnvrep.zpad(cnvrep.stdformD(D0n, cri.Cd, cri.M), cri.Nv),
             'U0': np.zeros(cri.shpD + (cri.K,))})


"""
Create X update object.
"""

xstep = cbpdn.ConvBPDNGradReg(D0n, S, lmbda, mu, optx)


"""
Create D update object.
"""

dstep = ccmod.ConvCnstrMOD(None, S, D0.shape, optd, method='cns')


"""
Create DictLearn object and solve.
"""

opt = dictlrn.DictLearn.Options({'Verbose': True, 'MaxMainIter': 200})
d = dictlrn.DictLearn(xstep, dstep, opt)
D1 = d.solve()
print("DictLearn solve time: %.2fs" % d.timer.elapsed('solve'), "\n")


"""
Display dictionaries.
"""

D1 = D1.squeeze()
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D1), title='D1', fig=fig)
fig.show()


"""
Plot functional value and residuals.
"""

itsx = xstep.getitstat()
itsd = dstep.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(itsx.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(np.vstack((itsx.PrimalRsdl, itsx.DualRsdl, itsd.PrimalRsdl,
          itsd.DualRsdl)).T, ptyp='semilogy', xlbl='Iterations',
          ylbl='Residual', lgnd=['X Primal', 'X Dual', 'D Primal', 'D Dual'],
          fig=fig)
plot.subplot(1, 3, 3)
plot.plot(np.vstack((itsx.Rho, itsd.Rho)).T,  xlbl='Iterations',
          ylbl='Penalty Parameter', ptyp='semilogy', lgnd=['Rho', 'Sigma'],
          fig=fig)
fig.show()


# Wait for enter on keyboard
input()
