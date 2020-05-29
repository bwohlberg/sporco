#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Convolutional Dictionary Learning
=================================

This example demonstrates the use of :class:`.prlcnscdl.ConvBPDNDictLearn_Consensus` for learning a convolutional dictionary from a set of training images. The dictionary learning algorithm is based on the ADMM consensus dictionary update :cite:`sorel-2016-fast` :cite:`garcia-2018-convolutional1`.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.dictlrn import prlcnscdl
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
Construct initial dictionary.
"""

np.random.seed(12345)
D0 = np.random.randn(8, 8, 64)


"""
Set regularization parameter and options for dictionary learning solver.
"""

lmbda = 0.2
opt = prlcnscdl.ConvBPDNDictLearn_Consensus.Options({'Verbose': True,
                        'MaxMainIter': 200,
                        'CBPDN': {'rho': 50.0*lmbda + 0.5},
                        'CCMOD': {'rho': 1.0, 'ZeroMean': True}})


"""
Create solver object and solve.
"""

d = prlcnscdl.ConvBPDNDictLearn_Consensus(D0, sh, lmbda, opt)
D1 = d.solve()
print("ConvBPDNDictLearn_Consensus solve time: %.2fs" %
      d.timer.elapsed('solve'))


"""
Display initial and final dictionaries.
"""

D1 = D1.squeeze()
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D1), title='D1', fig=fig)
fig.show()


"""
Get iterations statistics from solver object and plot functional value
"""

its = d.getitstat()
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional')


# Wait for enter on keyboard
input()
