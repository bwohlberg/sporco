#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Utility functions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from future.utils import PY2
from builtins import range
from builtins import object

import numpy as np
from scipy import misc
from timeit import default_timer as timer
import os
import glob
import multiprocessing as mp
import itertools

import sporco.linalg as sla
import sporco.plot as spl

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


import warnings

def plot(*args, **kwargs):
    warnings.warn("sporco.util.plot is deprecated: please use sporco.plot.plot")
    return spl.plot(*args, **kwargs)

def surf(*args, **kwargs):
    warnings.warn("sporco.util.surf is deprecated: please use sporco.plot.surf")
    return spl.surf(*args, **kwargs)

def imview(*args, **kwargs):
    warnings.warn("sporco.util.imview is deprecated: please use "
                  "sporco.plot.imview")
    return spl.imview(*args, **kwargs)



# Python 2/3 unicode literal compatibility
if PY2:
    import codecs
    def u(x):
        return x.decode('utf8')
else:
    def u(x):
        return x



def tiledict(D, sz=None):
    """Construct an image allowing visualization of dictionary content.

    Parameters
    ----------
    D : array_like
      Dictionary matrix/array.
    sz : tuple
      Size of each block in dictionary.

    Returns
    -------
    im : array_like
      Image tiled with dictionary entries.
    """

    # Handle standard 2D (non-convolutional) dictionary
    if D.ndim == 2:
        D = D.reshape((sz + (D.shape[1],)))
        sz = None
    dsz = D.shape

    if D.ndim == 4:
        axisM = 3
        szni = 3
    else:
        axisM = 2
        szni = 2

    # Construct dictionary atom size vector if not provided
    if sz is None:
        sz = np.tile(np.array(dsz[0:2]).reshape([2, 1]), (1, D.shape[axisM]))
    else:
        sz = np.array(sum(tuple((x[0:2],) * x[szni] for x in sz), ())).T

    # Compute the maximum atom dimensions
    mxsz = np.amax(sz, 1)

    # Shift and scale values to [0, 1]
    D = D - D.min()
    D = D / D.max()

    # Construct tiled image
    N = dsz[axisM]
    Vr = int(np.floor(np.sqrt(N)))
    Vc = int(np.ceil(N/float(Vr)))
    if D.ndim == 4:
        im = np.ones((Vr*mxsz[0] + Vr-1, Vc*mxsz[1] + Vc-1, dsz[2]))
    else:
        im = np.ones((Vr*mxsz[0] + Vr-1, Vc*mxsz[1] + Vc-1))
    k = 0
    for l in range(0, Vr):
        for m in range(0, Vc):
            r = mxsz[0]*l + l
            c = mxsz[1]*m + m
            if D.ndim == 4:
                im[r:(r+sz[0, k]), c:(c+sz[1, k]), :] = D[0:sz[0, k],
                                                          0:sz[1, k], :, k]
            else:
                im[r:(r+sz[0, k]), c:(c+sz[1, k])] = D[0:sz[0, k],
                                                       0:sz[1, k], k]
            k = k + 1
            if k >= N:
                break
        if k >= N:
            break

    return im



def imageblocks(imgs, blksz):
    """Extract all blocks of specified size from an image or list of images."""

    # See http://stackoverflow.com/questions/16774148 and
    # sklearn.feature_extraction.image.extract_patches_2d

    if not isinstance(imgs, tuple):
        imgs = (imgs,)

    blks = np.array([]).reshape(blksz + (0,))
    for im in imgs:
        Nr, Nc = im.shape
        nr, nc = blksz
        shape = (Nr-nr+1, Nc-nc+1, nr, nc)
        strides = im.itemsize*np.array([Nc, 1, Nc, 1])
        sb = np.lib.stride_tricks.as_strided(np.ascontiguousarray(im),
                                             shape=shape, strides=strides)
        sb = np.ascontiguousarray(sb)
        sb.shape = (-1, nr, nc)
        sb = np.rollaxis(sb, 0, 3)
        blks = np.dstack((blks, sb))

    return blks



def rgb2gray(rgb):
    """RGB to gray conversion function."""

    return np.dot(rgb[..., :3], np.array([0.299, 0.587, 0.144],
                                         dtype=np.float32))



def spnoise(s, frc, smn=0.0, smx=1.0):
    """Return image with salt & pepper noise imposed on it"""

    sn = s.copy()
    spm = np.random.uniform(-1.0, 1.0, s.shape)
    sn[spm < frc - 1.0] = smn
    sn[spm > 1.0 - frc] = smx
    return sn



def tikhonov_filter(s, lmbda, npd=16):
    """Lowpass filter based on Tikhonov regularization.

    Lowpass filter image(s) and return low and high frequency components,
    consisting of the lowpass filtered image and its difference with
    the input image. The lowpass filter is equivalent to Tikhonov
    regularization with lmbda as the regularization parameter and a
    discrete gradient as the operator in the regularization term.

    Parameters
    ----------
    s : array_like
      Input image or array of images.
    lmbda : float
      Regularization parameter controlling lowpass filtering.
    npd : int, optional (default=16)
      Number of samples to pad at image boundaries.

    Returns
    -------
    sl : array_like
      Lowpass image or array of images.
    sh : array_like
      Highpass image or array of images.
    """

    grv = np.array([-1.0, 1.0]).reshape([2, 1])
    gcv = np.array([-1.0, 1.0]).reshape([1, 2])
    Gr = sla.fftn(grv, (s.shape[0]+2*npd, s.shape[1]+2*npd), (0, 1))
    Gc = sla.fftn(gcv, (s.shape[0]+2*npd, s.shape[1]+2*npd), (0, 1))
    A = 1.0 + lmbda*np.conj(Gr)*Gr + lmbda*np.conj(Gc)*Gc
    if s.ndim > 2:
        A = A[(slice(None),)*2 + (np.newaxis,)*(s.ndim-2)]
    sp = np.pad(s, ((npd, npd),)*2 + ((0,0),)*(s.ndim-2), 'symmetric')
    slp = np.real(sla.ifftn(sla.fftn(sp, axes=(0,1)) / A, axes=(0,1)))
    sl = slp[npd:(slp.shape[0]-npd), npd:(slp.shape[1]-npd)]
    sh = s - sl
    return sl.astype(s.dtype), sh.astype(s.dtype)



def solve_status_str(hdrtxt, fwiter=4, fpothr=2):
    """Construct header and format details for status display of an
    iterative solver.
    """

    # Field width for all fields other than first depends on precision
    fwothr = fpothr + 6
    # Construct header string from hdrtxt list of column headers
    hdrstr = ("%-*s" % (fwiter+2, hdrtxt[0])) + \
        ((("%%-%ds " % (fwothr+1)) * (len(hdrtxt)-1)) % \
        tuple(hdrtxt[1:]))
    # Construct iteration status format string
    fmtstr = ("%%%dd" % (fwiter)) + ((("  %%%d.%de" % (fwothr, fpothr)) * \
        (len(hdrtxt)-1)))
    # Compute length of separator string
    nsep = fwiter + (fwothr + 2)*(len(hdrtxt)-1)

    return hdrstr, fmtstr, nsep




class Timer(object):
    """Simple timer class."""

    def __init__(self):
        """Initialise timer."""

        self.start()


    def start(self):
        """Reset timer."""

        self.t0 = timer()



    def elapsed(self):
        """Get elapsed time since timer start."""

        return timer() - self.t0




def convdicts():
    """Get a dict associating description strings with example learned
    convolutional dictionaries.
    """

    pth = os.path.join(os.path.dirname(__file__), 'data', 'convdict.npz')
    npz = np.load(pth)
    cdd = {}
    for k in list(npz.keys()):
        cdd[k] = npz[k]
    return cdd




class ExampleImages(object):
    """Example image access class"""

    def __init__(self, scaled=False):
        """Initialise object."""

        self.scaled = scaled
        self.bpth = os.path.join(os.path.dirname(__file__), 'data')
        flst = glob.glob(os.path.join(self.bpth, '') + '*.png')
        self.nlist = []
        for pth in flst:
            self.nlist.append(os.path.basename(os.path.splitext(pth)[0]))


    def names(self):
        """Get list of available names"""

        return self.nlist



    def image(self, name, scaled=None):
        """Get named image"""

        if scaled is None:
            scaled = self.scaled
        pth = os.path.join(self.bpth, name) + '.png'

        try:
            img = misc.imread(pth)
        except IOError:
            raise IOError('Could not access image with name ' + name)


        if scaled:
            img = np.float32(img) / 255.0

        return img




def grid_search(fn, grd, fmin=True, nproc=None):
    """Perform a grid search for optimal parameters of a specified
    function.  In the simplest case the function returns a float value,
    and a single optimum value and corresponding parameter values are
    identified. If the function returns a tuple of values, each of
    these is taken to define a separate function on the search grid,
    with optimum function values and corresponding parameter values
    being identified for each of them.

    **Warning:** This function will hang if `fn` makes use of :mod:`pyfftw`
    with multi-threading enabled (the
    `bug <https://github.com/pyFFTW/pyFFTW/issues/135>`_ has been reported).
    When using the FFT functions in :mod:`sporco.linalg`, multi-threading
    can be disabled by including the following code::

      import sporco.linalg
      sporco.linalg.pyfftw_threads = 1


    Parameters
    ----------
    fn : function
      Function to be evaluated. It should take a tuple of parameter values as
      an argument, and return a float value or a tuple of float values.
    grd : tuple of array_like
      A tuple providing an array of sample points for each axis of the grid
      on which the search is to be performed.
    fmin : boolean, optional (default True)
      Determine whether optimal function values are selected as minima or
      maxima. If `fmin` is True then minima are selected.
    nproc : int or None, optional (default None)
      Number of processes to run in parallel. If None, the number of
      CPUs of the system is used.

    Returns
    -------
    sprm : ndarray
      Optimal parameter values on each axis. If `fn` is multi-valued,
      `sprm` is a matrix with rows corresponding to parameter values
      and columns corresponding to function values.
    sfvl : float or ndarray
      Optimum function value or values
    fvmx : ndarray
      Function value(s) on search grid
    sidx : tuple of int or tuple of ndarray
      Indices of optimal values on parameter grid
    """

    if fmin:
        slct = np.argmin
    else:
        slct = np.argmax
    if nproc is None:
        nproc = mp.cpu_count()
    fprm = itertools.product(*grd)
    pool = mp.Pool(processes=nproc)
    fval = pool.map(fn, fprm)
    if isinstance(fval[0], (tuple, list, np.ndarray)):
        nfnv = len(fval[0])
        fvmx = np.reshape(fval, [a.size for a in grd] + [nfnv,])
        sidx = np.unravel_index(slct(fvmx.reshape((-1,nfnv)), axis=0),
                        fvmx.shape[0:-1]) + (np.array((range(nfnv))),)
        sprm = np.array([grd[k][sidx[k]] for k in range(len(grd))])
        sfvl = tuple(fvmx[sidx])
    else:
        fvmx = np.reshape(fval, [a.size for a in grd])
        sidx = np.unravel_index(slct(fvmx), fvmx.shape)
        sprm = np.array([grd[k][sidx[k]] for k in range(len(grd))])
        sfvl = fvmx[sidx]

    return sprm, sfvl, fvmx, sidx
