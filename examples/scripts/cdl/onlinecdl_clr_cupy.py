#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Online Convolutional Dictionary Learning (CuPy Version)
=======================================================

This example demonstrates the use of :class:`.onlinecdl.OnlineConvBPDNDictLearn` for learning a convolutional dictionary from a set of training images. The dictionary is learned using the online dictionary learning algorithm proposed in :cite:`liu-2018-first`. This variant of the example uses the GPU accelerated version of :mod:`.onlinecdl` within the :mod:`sporco.cupy` subpackage.
"""


from __future__ import print_function
from builtins import input

import numpy as np

from sporco import util
from sporco import signal
from sporco import plot
from sporco.cupy import (cupy_enabled, np2cp, cp2np, select_device_by_load,
                         gpu_info)
from sporco.cupy.dictlrn import onlinecdl


"""
Load training images.
"""

exim = util.ExampleImages(scaled=True, zoom=0.25)
S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
S3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
S4 = exim.image('sail.png', idxexp=np.s_[:, 210:722])
S5 = exim.image('tulips.png', idxexp=np.s_[:, 30:542])
S = np.stack((S1, S2, S3, S4, S5), axis=3)


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
D0 = np.random.randn(8, 8, 3, 64)


"""
Set regularization parameter and options for dictionary learning solver.
"""

lmbda = 0.2
opt = onlinecdl.OnlineConvBPDNDictLearn.Options({
                'Verbose': True, 'ZeroMean': False, 'eta_a': 10.0,
                'eta_b': 20.0, 'DataType': np.float32,
                'CBPDN': {'rho': 5.0, 'AutoRho': {'Enabled': True},
                    'RelaxParam': 1.8, 'RelStopTol': 1e-7, 'MaxMainIter': 50,
                    'FastSolve': False, 'DataType': np.float32}})


"""
Create solver object and solve.
"""

if not cupy_enabled():
    print('CuPy/GPU device not available: running without GPU acceleration\n')
else:
    id = select_device_by_load()
    info = gpu_info()
    if info:
        print('Running on GPU %d (%s)\n' % (id, info[id].name))

d = onlinecdl.OnlineConvBPDNDictLearn(np2cp(D0), lmbda, opt)

iter = 50
d.display_start()
for it in range(iter):
    img_index = np.random.randint(0, sh.shape[-1])
    d.solve(np2cp(sh[..., [img_index]]))

d.display_end()
D1 = cp2np(d.getdict())
print("OnlineConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))


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
Get iterations statistics from solver object and plot functional value.
"""

its = d.getitstat()
DeltaD = [float(x) for x in its.DeltaD]
fig = plot.figure(figsize=(7, 7))
plot.plot(np.vstack((DeltaD, its.Eta)).T, xlbl='Iterations',
          lgnd=('Delta D', 'Eta'), fig=fig)
fig.show()


# Wait for enter on keyboard
input()
