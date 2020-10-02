#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Constained MOD
==============

This example demonstrates the use of :class:`.pgm.cmod.CnstrMOD` for computing a dictionary update via a constrained variant of the method of optimal directions :cite:`engan-1999-method`. It also illustrates the use of :class:`.pgm.backtrack.BacktrackRobust`, :class:`.pgm.stepsize.StepSizePolicyCauchy` and :class:`.pgm.stepsize.StepSizePolicyBB` to adapt the step size parameter of PGM. This problem is mainly useful as a component within dictionary learning, but its use is demonstrated here since a user may wish to construct such objects as part of a custom dictionary learning algorithm, using :class:`.dictlrn.DictLearn`.
"""


from __future__ import print_function
from builtins import input

import numpy as np

from sporco.pgm import bpdn
from sporco.pgm import cmod
from sporco.pgm.backtrack import BacktrackRobust
from sporco.pgm.stepsize import StepSizePolicyBB, StepSizePolicyCauchy
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
opt = bpdn.BPDN.Options({'Verbose': True, 'MaxMainIter': 50, 'L': 100,
            'Backtrack': BacktrackRobust()})
b = bpdn.BPDN(D0, S, lmbda, opt)
X = b.solve()


"""
Update dictionary for training image set using PGM with Cauchy step size policy :cite:`yuan-2008-stepsize`.
"""

opt = cmod.CnstrMOD.Options({'Verbose': True, 'MaxMainIter': 100, 'L': 50,
        'StepSizePolicy': StepSizePolicyCauchy()})
c1 = cmod.CnstrMOD(X, S, None, opt)
D11 = c1.solve()
print("CMOD solve time: %.2fs" % c1.timer.elapsed('solve'))


"""
Update dictionary for training image set using PGM with Barzilai-Borwein step size policy :cite:`barzilai-1988-stepsize`.
"""

opt = cmod.CnstrMOD.Options({'Verbose': True, 'MaxMainIter': 100, 'L': 50,
        'StepSizePolicy': StepSizePolicyBB()})
c2 = cmod.CnstrMOD(X, S, None, opt)
D12 = c2.solve()
print("CMOD solve time: %.2fs" % c2.timer.elapsed('solve'))


"""
Display initial and final dictionaries.
"""

D0 = D0.reshape((8, 8, D0.shape[-1]))
D11 = D11.reshape((8, 8, D11.shape[-1]))
D12 = D12.reshape((8, 8, D12.shape[-1]))
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 3, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(1, 3, 2)
plot.imview(util.tiledict(D11), title='D1 Cauchy', fig=fig)
plot.subplot(1, 3, 3)
plot.imview(util.tiledict(D12), title='D1 BB', fig=fig)
fig.show()


"""
Get iterations statistics from CMOD solver object and plot functional value, residuals, and automatically adjusted L against the iteration number.
"""

its1 = c1.getitstat()
its2 = c2.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(its1.DFid, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.plot(its2.DFid, xlbl='Iterations', ylbl='Functional',
          lgnd=['Cauchy', 'BB'], fig=fig)
plot.subplot(1, 3, 2)
plot.plot(its1.Rsdl, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          fig=fig)
plot.plot(its2.Rsdl, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Cauchy', 'BB'], fig=fig)
plot.subplot(1, 3, 3)
plot.plot(its1.L, xlbl='Iterations', ylbl='Inverse of Step Size', fig=fig)
plot.plot(its2.L, xlbl='Iterations', ylbl='Inverse of Step Size',
          lgnd=[r'$L_{Cauchy}$', '$L_{BB}$'], fig=fig)
fig.show()


# Wait for enter on keyboard
input()
