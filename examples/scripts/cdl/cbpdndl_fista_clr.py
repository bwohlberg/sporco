#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Convolutional Dictionary Learning
=================================

This example demonstrates the use of :class:`.fista.cbpdndl.ConvBPDNDictLearn` for learning a convolutional dictionary from a set of colour training images :cite:`wohlberg-2016-convolutional`, using FISTA solvers for both sparse coding :cite:`chalasani-2013-fast` :cite:`wohlberg-2016-efficient` and dictionary update steps :cite:`garcia-2017-convolutional`.
"""


from __future__ import print_function
from builtins import input
from builtins import range

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.fista import cbpdndl
from sporco import util
from sporco import plot


"""
Load training images.
"""

exim = util.ExampleImages(scaled=True, zoom=0.5)
img1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
img2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
img3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
S = np.stack((img1, img2, img3), axis=3)


"""
Highpass filter training images.
"""

npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(S, fltlmbd, npd)


"""
Construct initial dictionary.
"""

np.random.seed(12345)
D0 = np.random.randn(16, 16, 3, 96)


"""
Set regularization parameter and options for dictionary learning solver. Note the multi-scale dictionary filter sizes.
"""

lmbda = 0.2
L_sc = 360.0
L_du = 50.0
dsz = ((8, 8, 3, 32), (12, 12, 3, 32), (16, 16, 3, 32))
opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True,
                'MaxMainIter': 200, 'DictSize': dsz,
                'CBPDN': {'BackTrack': {'Enabled': True }, 'L': L_sc},
                'CCMOD': {'BackTrack': {'Enabled': True }, 'L': L_du } })


"""
Create solver object and solve.
"""

d = cbpdndl.ConvBPDNDictLearn(D0, sh, lmbda, opt)
D1 = d.solve()
print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))


"""
Display initial and final dictionaries.
"""

D1 = D1.squeeze()
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D0), fig=fig, title='D0')
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D1, dsz), fig=fig, title='D1')
fig.show()


"""
Get iterations statistics from solver object and plot functional value, residuals, and automatically adjusted gradient step parameters against the iteration number.
"""

its = d.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, fig=fig, xlbl='Iterations', ylbl='Functional')
plot.subplot(1, 3, 2)
plot.plot(np.vstack((its.X_Rsdl, its.D_Rsdl)).T,
          fig=fig, ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['X', 'D'])
plot.subplot(1, 3, 3)
plot.plot(np.vstack((its.X_L, its.D_L)).T, fig=fig, xlbl='Iterations',
          ylbl='Inverse of Gradient Step Parameter', ptyp='semilogy',
          lgnd=['$\L_X$', '$\L_D$'])
fig.show()


# Wait for enter on keyboard
input()
