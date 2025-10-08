#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Robust PCA
==========

This example demonstrates the use of class :class:`.rpca.RobustPCA` for video foreground/background separation via Robust PCA, the low-rank component representing the static background and the sparse component representing the moving foreground.
"""


from __future__ import print_function
from builtins import input

import numpy as np
import imageio

from sporco.admm import rpca
from sporco import signal
from sporco import plot


"""
Load example video.
"""

reader = imageio.get_reader('imageio:newtonscradle.gif')
nfrm = reader.get_length()
frmlst = []
for i, frm in enumerate(reader):
    frmlst.append(signal.rgb2gray(np.array(frm, dtype=np.float32)[..., 0:3]/255.0))
v = np.stack(frmlst, axis=2)


"""
Construct matrix with each column consisting of a vectorised video frame.
"""

S = v.reshape((-1, v.shape[-1]))


"""
Set options for the Robust PCA solver, create the solver object, and solve, returning the estimates of the low rank and sparse components ``X`` and ``Y``. Unlike most other SPORCO classes for optimisation problems, :class:`.rpca.RobustPCA` has a meaningful default regularization parameter, as used here.
"""

opt = rpca.RobustPCA.Options({'Verbose': True, 'gEvalY': False,
                              'MaxMainIter': 200, 'RelStopTol': 1e-3,
                              'AutoRho': {'Enabled': True}})
b = rpca.RobustPCA(S, None, opt)
X, Y = b.solve()


"""
Display solve time.
"""

print("RobustPCA solve time: %5.2f s" % b.timer.elapsed('solve'))


"""
Get iterations statistics from solver object and plot functional value, ADMM primary and dual residuals, and automatically adjusted ADMM penalty parameter against the iteration number.
"""

its = b.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'], fig=fig)
plot.subplot(1, 3, 3)
plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter', fig=fig)
fig.show()


"""
Reshape low-rank component ``X`` as background video sequence and sparse component ``Y`` as foreground video sequence.
"""

vbg = X.reshape(v.shape)
vfg = Y.reshape(v.shape)


"""
Display original video frames and corresponding background and foreground frames.
"""

fig, ax = plot.subplots(nrows=6, ncols=3, figsize=(12, 22))
ax[0][0].set_title("Original")
ax[0][1].set_title("Background")
ax[0][2].set_title("Foreground")
for n, fn in enumerate(range(1, 13, 2)):
    plot.imview(v[..., fn], fig=fig, ax=ax[n][0])
    plot.imview(vbg[..., fn], fig=fig, ax=ax[n][1])
    plot.imview(vfg[..., fn], fig=fig, ax=ax[n][2])
    ax[n][0].set_ylabel("Frame %d" % fn, labelpad=35, rotation=0, size='large')
fig.tight_layout()
fig.show()


# Wait for enter on keyboard
input()
