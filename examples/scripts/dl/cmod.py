#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Constained MOD
==============

This example demonstrates the use of :class:`.admm.cmod.CnstrMOD` for computing a dictionary update via a constrained variant of the method of optimal directions :cite:`engan-1999-method`. This problem is mainly useful as a component within dictionary learning, but its use is demonstrated here since a user may wish to construct such objects as part of a custom dictionary learning algorithm, using :class:`.dictlrn.DictLearn`.
"""


from __future__ import print_function
from builtins import input

import numpy as np

from sporco.admm import bpdn
from sporco.admm import cmod
from sporco import util
from sporco import array
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


"""
Extract all 8x8 image blocks, reshape, and subtract block means.
"""

S = array.extract_blocks((S1, S2, S3, S4, S5), (8, 8))
S = np.reshape(S, (np.prod(S.shape[0:2]), S.shape[2]))
S -= np.mean(S, axis=0)


"""
Load initial dictionary.
"""

D0 = util.convdicts()['G:8x8x64']
D0 = np.reshape(D0, (np.prod(D0.shape[0:2]), D0.shape[2]))


"""
Compute sparse representation on current dictionary.
"""

lmbda = 0.1
opt = bpdn.BPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                         'RelStopTol': 1e-3})
b = bpdn.BPDN(D0, S, lmbda, opt)
X = b.solve()


"""
Update dictionary for training image set.
"""

opt = cmod.CnstrMOD.Options({'Verbose': True, 'MaxMainIter': 100,
                             'RelStopTol': 1e-3, 'rho': 4e2})
c = cmod.CnstrMOD(X, S, None, opt)
D1 = c.solve()
print("CMOD solve time: %.2fs" % c.timer.elapsed('solve'))


"""
Display initial and final dictionaries.
"""

D0 = D0.reshape((8, 8, D0.shape[-1]))
D1 = D1.reshape((8, 8, D1.shape[-1]))
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D1), title='D1', fig=fig)
fig.show()


"""
Get iterations statistics from CMOD solver object and plot functional value, ADMM primary and dual residuals, and automatically adjusted ADMM penalty parameter against the iteration number.
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
