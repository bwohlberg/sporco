#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Convolutional Constained MOD
============================

This example demonstrates the use of :class:`.pgm.ccmod.ConvCnstrMOD` for computing a convolutional dictionary update via the convolutional constrained method of optimal directions problem :cite:`garcia-2018-convolutional1`. It also illustrates the use of :class:`.pgm.momentum.MomentumNesterov`, :class:`.pgm.momentum.MomentumLinear` and :class:`.pgm.momentum.MomentumGenLinear` to adapt the momentum coefficients of PGM. This problem is mainly useful as a component within convolutional dictionary learning, but its use is demonstrated here since a user may wish to construct such objects as part of a custom convolutional dictionary learning algorithm, using :class:`.dictlrn.DictLearn`.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.admm import cbpdn
from sporco.pgm import ccmod
from sporco.pgm.momentum import MomentumLinear, MomentumGenLinear
from sporco import util
from sporco import signal
from sporco import plot


"""
Load training images.
"""

exim = util.ExampleImages(scaled=True, zoom=0.25, gray=True)
S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
S3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
S4 = exim.image('sail.png', idxexp=np.s_[:, 210:722])
S5 = exim.image('tulips.png', idxexp=np.s_[:, 30:542])
S = np.dstack((S1, S2, S3, S4, S5))


"""
Highpass filter training images.
"""

npd = 16
fltlmbd = 5
sl, sh = signal.tikhonov_filter(S, fltlmbd, npd)


"""
Load initial dictionary.
"""

D0 = util.convdicts()['G:12x12x36']


"""
Compute sparse representation on current dictionary.
"""

lmbda = 0.1
opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 100,
                              'HighMemSolve': True})
c = cbpdn.ConvBPDN(D0, sh, lmbda, opt)
X = c.solve()


"""
Update dictionary for training image set. Nesterov momentum coefficients :cite:`beck-2009-fast`.
"""

opt = ccmod.ConvCnstrMOD.Options({'Verbose': True,
            'MaxMainIter': 100, 'L': 50})
c1 = ccmod.ConvCnstrMOD(X, sh, D0.shape, opt)
c1.solve()
D11 = c1.getdict().squeeze()
print("ConvCnstrMOD solve time: %.2fs" % c1.timer.elapsed('solve'))


"""
Update dictionary for training image set. Linear momentum coefficients :cite:`chambolle-2015-convergence`.
"""

opt = ccmod.ConvCnstrMOD.Options({'Verbose': True, 'MaxMainIter': 100,
             'Momentum': MomentumLinear(), 'L': 50})
c2 = ccmod.ConvCnstrMOD(X, sh, D0.shape, opt)
c2.solve()
D12 = c2.getdict().squeeze()
print("ConvCnstrMOD solve time: %.2fs" % c2.timer.elapsed('solve'))


"""
Update dictionary for training image set. Generalized linear momentum coefficients :cite:`rodriguez-2019-convergence`.
"""

opt = ccmod.ConvCnstrMOD.Options({'Verbose': True, 'MaxMainIter': 100,
             'Momentum': MomentumGenLinear(), 'L': 50})
c3 = ccmod.ConvCnstrMOD(X, sh, D0.shape, opt)
c3.solve()
D13 = c3.getdict().squeeze()
print("ConvCnstrMOD solve time: %.2fs" % c3.timer.elapsed('solve'))



"""
Display initial and final dictionaries.
"""

fig = plot.figure(figsize=(7, 7))
plot.subplot(2, 2, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(2, 2, 2)
plot.imview(util.tiledict(D11), title='D1 Nesterov', fig=fig)
plot.subplot(2, 2, 3)
plot.imview(util.tiledict(D12), title='D1 Linear', fig=fig)
plot.subplot(2, 2, 4)
plot.imview(util.tiledict(D13), title='D1 GenLinear', fig=fig)
fig.show()


"""
Get iterations statistics from CCMOD solver object and plot functional value, and residuals.
"""

its1 = c1.getitstat()
its2 = c2.getitstat()
its3 = c3.getitstat()
fig = plot.figure(figsize=(15, 5))
plot.subplot(1, 2, 1)
plot.plot(its1.DFid, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.plot(its2.DFid, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.plot(its3.DFid, xlbl='Iterations', ylbl='Functional',
          lgnd=['Nesterov', 'Linear', 'GenLinear'], fig=fig)
plot.subplot(1, 2, 2)
plot.plot(its1.Rsdl, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          fig=fig)
plot.plot(its2.Rsdl, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          fig=fig)
plot.plot(its3.Rsdl, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Nesterov', 'Linear', 'GenLinear'], fig=fig)
fig.show()


# Wait for enter on keyboard
input()
