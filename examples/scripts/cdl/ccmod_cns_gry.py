#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Convolutional Constained MOD
============================

This example demonstrates the use of :class:`.ccmod.ConvCnstrMOD_Consensus` for computing a convolutional dictionary update via the convolutional constrained method of optimal directions problem :cite:`sorel-2016-fast` :cite:`garcia-2018-convolutional1`. This problem is mainly useful as a component within convolutional dictionary learning, but its use is demonstrated here since a user may wish to construct such objects as part of a custom convolutional dictionary learning algorithm, using :class:`.dictlrn.DictLearn`.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.admm import cbpdn
from sporco.admm import ccmod
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
Update dictionary for training image set.
"""

opt = ccmod.ConvCnstrMOD_Consensus.Options({'Verbose': True,
            'MaxMainIter': 100, 'rho': 1e1})
c = ccmod.ConvCnstrMOD_Consensus(X, sh, D0.shape, opt)
c.solve()
D1 = c.getdict().squeeze()
print("ConvCnstrMOD_Consensus solve time: %.2fs" % c.timer.elapsed('solve'))


"""
Display initial and final dictionaries.
"""

fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D1), title='D1', fig=fig)
fig.show()


"""
Get iterations statistics from CCMOD solver object and plot functional value, ADMM primary and dual residuals, and automatically adjusted ADMM penalty parameter against the iteration number.
"""

its = c.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(its.DFid, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'], fig=fig)
plot.subplot(1, 3, 3)
plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter', fig=fig)
fig.show()


# Wait for enter on keyboard
input()
