#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Usage example for cbpdndl.ConvBPDNDictLearn with 3d data and dictionary"""

from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
import urllib2
import os.path
import tempfile
import sys
import matplotlib.pyplot as plt
try:
    import cv2
except ImportError:
    print('Module cv2 is required by this demo script', file=sys.stderr)
    raise

from sporco.admm import cbpdndl
from sporco.admm import ccmod
from sporco import util


# Get test video
pth = os.path.join(tempfile.gettempdir(), 'foreman_qcif_mono.y4m')
if not os.path.isfile(pth):
    url = 'https://media.xiph.org/video/derf/y4m/foreman_qcif_mono.y4m'
    response = urllib2.urlopen(url, timeout=5)
    content = response.read()
    f = open(pth, 'w')
    f.write(content)
    f.close()

# Extract video as 3d array
vid = np.zeros((144,176,300))
cap = cv2.VideoCapture(pth)
k = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vid[...,k] = gray
    k += 1
cap.release()
vid = vid[50:114,50:114,0:32] / 255.0


# Highpass filter frames
npd = 16
fltlmbd = 5
vl, vh = util.tikhonov_filter(vid, fltlmbd, npd)


# Initial dictionary
np.random.seed(12345)
D0 = np.random.randn(5, 5, 3, 75)


# Set ConvBPDNDictLearn parameters
lmbda = 0.1
opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose' : True, 'MaxMainIter' : 200,
                                         'CBPDN' : {'rho' : 5e4*lmbda,
                                            'AutoRho' : {'Enabled' : True}},
                                         'CCMOD' : {'rho' : 5, 
                                           'AutoRho' : {'Enabled' : True},
                                                    'ZeroMean' : True}})


# Run optimisation
d = cbpdndl.ConvBPDNDictLearn(D0, vh, lmbda, opt, dimN=3, dimK=0)
d.solve()
print("ConvBPDNDictLearn solve time: %.2fs" % d.runtime, "\n")


# Display central temporal slice (index 2) of dictionaries
D1 = ccmod.bcrop(d.ccmod.Y, D0.shape).squeeze()
fig1 = plt.figure(1, figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(util.tiledict(D0[...,1,:]), interpolation="nearest",
           cmap=plt.get_cmap('gray'))
plt.title('D0 slice')
plt.subplot(1,2,2)
plt.imshow(util.tiledict(D1[...,1,:]), interpolation="nearest",
           cmap=plt.get_cmap('gray'))
plt.title('D1 slice')
fig1.show()


# Plot functional value and residuals
fig2 = plt.figure(2, figsize=(25,10))
plt.subplot(1,3,1)
plt.plot([d.itstat[k].ObjFun for k in range(0, len(d.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Functional')
ax=plt.subplot(1,3,2)
plt.semilogy([d.itstat[k].XPrRsdl for k in range(0, len(d.itstat))])
plt.semilogy([d.itstat[k].XDlRsdl for k in range(0, len(d.itstat))])
plt.semilogy([d.itstat[k].DPrRsdl for k in range(0, len(d.itstat))])
plt.semilogy([d.itstat[k].DDlRsdl for k in range(0, len(d.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Residual')
plt.legend(['X Primal', 'X Dual', 'D Primal', 'D Dual'])
plt.subplot(1,3,3)
plt.semilogy([d.itstat[k].Rho for k in range(0, len(d.itstat))])
plt.semilogy([d.itstat[k].Sigma for k in range(0, len(d.itstat))])
plt.xlabel('Iterations')
plt.ylabel('Penalty Parameter')
plt.legend(['Rho', 'Sigma'])
fig2.show()

input()
