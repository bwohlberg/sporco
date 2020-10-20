#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Dictionary Learning
===================

This example demonstrates the use of class :class:`.wbpdndl.WeightedBPDNDictLearn` for learning a dictionary (standard, not convolutional) from a set of training images. The primary purpose of this example is to demonstrate the use of a dictionary learning class based on PGM solvers for the sparse coding and dictionary update stages; the support for a weighted data fidelity term that is included in :class:`.wbpdndl.WeightedBPDNDictLearn` is not used.
"""


from __future__ import division, print_function
from builtins import input

import numpy as np

from sporco.dictlrn import wbpdndl
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
Construct initial dictionary.
"""

np.random.seed(12345)
D0 = np.random.randn(S.shape[0], 128)


"""
Set regularization parameter and options for dictionary learning solver.
"""

lmbda = 0.1
opt = wbpdndl.WeightedBPDNDictLearn.Options(
    {'Verbose': True, 'MaxMainIter': 150,
     'BPDN': {'L': 1e1}, 'CMOD': {'L': 1e3}})


"""
Create solver object and solve.
"""

d = wbpdndl.WeightedBPDNDictLearn(D0, S, lmbda, opt=opt)
d.solve()
print("WeightedBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))


"""
Display initial and final dictionaries.
"""

D1 = d.getdict().reshape((8, 8, D0.shape[1]))
D0 = D0.reshape(8, 8, D0.shape[-1])
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D1), title='D1', fig=fig)
fig.show()


"""
Get iterations statistics from solver object and plot functional value and PGM residuals.
"""

its = d.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 2, 1)
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 2, 2)
plot.plot(np.vstack((its.XRsdl, its.DRsdl)).T, ptyp='semilogy',
          xlbl='Iterations', ylbl='Residual', lgnd=['X', 'D'],
          fig=fig)
fig.show()


# Wait for enter on keyboard
input()
