#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example: cbpdn.AddMaskSim (greyscale images)"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np

from sporco import plot
from sporco.admm import cbpdn
import sporco.metric as sm


# Construct image, mask, and dictionary. The example is such that the
# most effective representation, consisting of only 4 non-zero
# coefficients, is only possible if the mask on the reconstructed
# image is applied to the data fidelity term.
N = 16
L = 6
S = np.zeros((2*(N+L)+1,)*2)
S[L+N,L:2*L+1] = 1.0
S[L+N,-2*L-1:-L] = 1.0
S[L:2*L+1,L+N] = 1.0
S[-2*L-1:-L,L+N] = 1.0
W = np.zeros(S.shape)
W[L:-L,L:-L] = 1.0
D = np.zeros((2*L+1,)*2+(2,))
D[L,:,0] = 1.0
D[:,L,1] = 1.0


# Set up ConvBPDNMaskDcpl options
lmbda = 1e-3
opt = cbpdn.ConvBPDNMaskDcpl.Options({'Verbose' : True, 'MaxMainIter' : 500,
                              'HighMemSolve' : False, 'RelStopTol' : 1e-3,
                              'AuxVarObj' : False, 'RelaxParam' : 1.0,
                              'rho' : 2e-1, 'LinSolveCheck' : True,
                    'AutoRho' : {'Enabled' : False, 'StdResiduals' : True}})

# Initialise and run ConvBPDNMaskDcpl object
b = cbpdn.ConvBPDNMaskDcpl(D, S, lmbda, W, opt)
X = b.solve()
print("ConvBPDNMaskDcpl solve time: %.2fs" % b.timer.elapsed('solve'))

# Reconstruct representation
Sr = b.reconstruct().squeeze()
print("        reconstruction PSNR: %.2fdB\n" % sm.psnr(S, Sr))


# Display representation and reconstructed image
fig1 = plot.figure(1, figsize=(14,14))
plot.subplot(2,2,1)
plot.imview(np.squeeze(np.sum(abs(X), axis=b.cri.axisM)), fgrf=fig1,
            cmap=plot.cm.Blues, title='Representation')
plot.subplot(2,2,2)
plot.imview(S, fgrf=fig1, cmap=plot.cm.Blues, title='Reference image')
plot.subplot(2,2,3)
plot.imview(Sr, fgrf=fig1, cmap=plot.cm.Blues, title='Reconstructed image')
plot.subplot(2,2,4)
plot.imview(W * Sr, fgrf=fig1, cmap=plot.cm.Blues,
            title='Masked reconstructed image')
fig1.show()


# Display mask and Y1 component
fig2 = plot.figure(2, figsize=(14,7))
plot.subplot(1,2,1)
plot.imview(W, fgrf=fig2, cmap=plot.cm.Blues, title='Mask')
plot.subplot(1,2,2)
plot.imview(b.block_sep0(b.Y).squeeze(), fgrf=fig2, cmap=plot.cm.Blues,
            title='Y0 component')
fig2.show()


# Plot functional value, residuals, and rho
its = b.getitstat()
fig3 = plot.figure(3, figsize=(21,7))
plot.subplot(1,3,1)
plot.plot(its.ObjFun, fgrf=fig3, xlbl='Iterations', ylbl='Functional')
plot.subplot(1,3,2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T, fgrf=fig3,
          ptyp='semilogy', xlbl='Iterations', ylbl='Residual',
          lgnd=['Primal', 'Dual'])
plot.subplot(1,3,3)
plot.plot(its.Rho, fgrf=fig3, xlbl='Iterations', ylbl='Penalty Parameter')
fig3.show()


# Wait for enter on keyboard
input()
