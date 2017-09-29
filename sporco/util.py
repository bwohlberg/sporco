# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
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

from timeit import default_timer as timer
import os
import imghdr
import io
import platform
import multiprocessing as mp
import itertools
import collections
import socket
if PY2:
    import urllib2 as urlrequest
    import urllib2 as urlerror
else:
    import urllib.request as urlrequest
    import urllib.error as urlerror
import numpy as np
from scipy import misc
import scipy.ndimage.interpolation as sni

import sporco.linalg as sla
import sporco.plot as spl

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


import warnings

def plot(*args, **kwargs):
    warnings.warn("sporco.util.plot is deprecated: use sporco.plot.plot",
                  PendingDeprecationWarning)
    return spl.plot(*args, **kwargs)

def surf(*args, **kwargs):
    warnings.warn("sporco.util.surf is deprecated: use sporco.plot.surf",
                  PendingDeprecationWarning)
    return spl.surf(*args, **kwargs)

def imview(*args, **kwargs):
    warnings.warn("sporco.util.imview is deprecated: use sporco.plot.imview",
                  PendingDeprecationWarning)
    return spl.imview(*args, **kwargs)



# Python 2/3 unicode literal compatibility
if PY2:
    import codecs
    def u(x):
        """Python 2/3 compatible definition of utf8 literals"""
        return x.decode('utf8')
else:
    def u(x):
        """Python 2/3 compatible definition of utf8 literals"""
        return x




def ntpl2array(ntpl):
    """
    Convert a :func:`collections.namedtuple` object to a :class:`numpy.ndarray`
    object that can be saved using :func:`numpy.savez`.

    Parameters
    ----------
    ntpl : collections.namedtuple object
      Named tuple object to be converted to ndarray

    Returns
    -------
    arr : ndarray
      Array representation of input named tuple
    """

    return np.asarray((np.vstack([col for col in ntpl]), ntpl._fields,
                       ntpl.__class__.__name__))



def array2ntpl(arr):
    """
    Convert a :class:`numpy.ndarray` object constructed by :func:`ntpl2array`
    back to the original :func:`collections.namedtuple` representation.

   Parameters
    ----------
    arr : ndarray
      Array representation of named tuple constructed by :func:`ntpl2array`

    Returns
    -------
    ntpl : collections.namedtuple object
      Named tuple object with the same name and fields as the original named
      typle object provided to :func:`ntpl2array`
    """

    cls = collections.namedtuple(arr[2], arr[1])
    return cls(*tuple(arr[0]))



def transpose_ntpl_list(lst):
    """Transpose a list of named tuple objects (of the same type) into a
    named tuple of lists.

    Parameters
    ----------
    lst : list of collections.namedtuple object
      List of named tuple objects of the same type

    Returns
    -------
    ntpl : collections.namedtuple object
      Named tuple object with each entry consisting of a list of the
      corresponding fields of the named tuple objects in list ``lst``
    """

    cls = collections.namedtuple(lst[0].__class__.__name__, lst[0]._fields)
    if len(lst) == 0:
        return None
    else:
        return cls(*[[lst[k][l] for k in range(len(lst))]
                   for l in range(len(lst[0]))])



def solve_status_str(hdrtxt, fwiter=4, fpothr=2):
    """Construct header and format details for status display of an
    iterative solver.

    Parameters
    ----------
    hdrtxt : tuple of strings
      Tuple of field header strings
    fwiter : int, optional (default 4)
      Number of characters in iteration count integer field
    fpothr : int, optional (default 2)
      Precision of other float field

    Returns
    -------
    hdrstr : string
      Complete header string
    fmtstr : string
      Complete print formatting string for numeric values
    nsep : integer
      Number of characters in separator string
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
    im : ndarray
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
    """Extract all blocks of specified size from an image or list of images.

    Parameters
    ----------
    imgs: array_like or tuple of array_like
      Single image or tuple of images from which to extract blocks
    blksz : tuple of two ints
      Size of the blocks

    Returns
    -------
    blks : ndarray
      Array of extracted blocks
    """

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
    """Convert an RGB image (or images) to grayscale.

    Parameters
    ----------
    rgb : ndarray
      RGB image as Nr x Nc x 3 or Nr x Nc x 3 x K array

    Returns
    -------
    gry : ndarray
      Grayscale image as Nr x Nc or Nr x Nc x K array
    """

    w = sla.atleast_nd(rgb.ndim, np.array([0.299, 0.587, 0.144],
                        dtype=rgb.dtype, ndmin=3))
    return np.sum(w * rgb, axis=2)



def complex_randn(*args):
    """Return a complex array of samples drawn from a standard normal
    distribution.

    Parameters
    ----------
    d0, d1, ..., dn: int
      Dimensions of the random array

    Returns
    -------
    a : ndarray
      Random array of shape (d0, d1, ..., dn)
    """

    return np.random.randn(*args) + 1j*np.random.randn(*args)



def spnoise(s, frc, smn=0.0, smx=1.0):
    """Return image with salt & pepper noise imposed on it.

    Parameters
    ----------
    s : ndarray
      Input image
    frc : float
      Desired fraction of pixels corrupted by noise
    smn : float, optional (default 0.0)
      Lower value for noise (pepper)
    smx : float, optional (default 1.0)
      Upper value for noise (salt)

    Returns
    -------
    sn : ndarray
      Noisy image
    """

    sn = s.copy()
    spm = np.random.uniform(-1.0, 1.0, s.shape)
    sn[spm < frc - 1.0] = smn
    sn[spm > 1.0 - frc] = smx
    return sn



def tikhonov_filter(s, lmbda, npd=16):
    r"""Lowpass filter based on Tikhonov regularization.

    Lowpass filter image(s) and return low and high frequency
    components, consisting of the lowpass filtered image and its
    difference with the input image. The lowpass filter is equivalent to
    Tikhonov regularization with `lmbda` as the regularization parameter
    and a discrete gradient as the operator in the regularization term,
    i.e. the lowpass component is the solution to

    .. math::
      \mathrm{argmin}_\mathbf{x} \; (1/2) \left\|\mathbf{x} - \mathbf{s}
      \right\|_2^2 + (\lambda / 2) \sum_i \| G_i \mathbf{x} \|_2^2 \;\;,

    where :math:`\mathbf{s}` is the input image, :math:`\lambda` is the
    regularization parameter, and :math:`G_i` is an operator that
    computes the discrete gradient along image axis :math:`i`. Once the
    lowpass component :math:`\mathbf{x}` has been computed, the highpass
    component is just :math:`\mathbf{s} - \mathbf{x}`.


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
    sp = np.pad(s, ((npd, npd),)*2 + ((0, 0),)*(s.ndim-2), 'symmetric')
    slp = np.real(sla.ifftn(sla.fftn(sp, axes=(0, 1)) / A, axes=(0, 1)))
    sl = slp[npd:(slp.shape[0]-npd), npd:(slp.shape[1]-npd)]
    sh = s - sl
    return sl.astype(s.dtype), sh.astype(s.dtype)



def idle_cpu_count(mincpu=1):
    """Estimate number of idle CPUs, for use by multiprocessing code
    needing to determine how many processes can be run without excessive
    load. This function uses :func:`os.getloadavg` which is only available
    under a Unix OS.

    Parameters
    ----------
    mincpu : int
      Minimum number of CPUs to report, independent of actual estimate

    Returns
    -------
    idle : int
      Estimate of number of idle CPUs
    """

    if PY2:
        ncpu = mp.cpu_count()
    else:
        ncpu = os.cpu_count()
    idle = int(ncpu - np.floor(os.getloadavg()[0]))
    return max(mincpu, idle)



def grid_search(fn, grd, fmin=True, nproc=None):
    """Perform a grid search for optimal parameters of a specified
    function.  In the simplest case the function returns a float value,
    and a single optimum value and corresponding parameter values are
    identified. If the function returns a tuple of values, each of
    these is taken to define a separate function on the search grid,
    with optimum function values and corresponding parameter values
    being identified for each of them. On all platforms except Windows
    (where ``mp.Pool`` usage has some limitations), the computation
    of the function at the grid points is computed in parallel.

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
    fmin : bool, optional (default True)
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
    fprm = itertools.product(*grd)
    if platform.system() == 'Windows':
        fval = list(map(fn, fprm))
    else:
        if nproc is None:
            nproc = mp.cpu_count()
        pool = mp.Pool(processes=nproc)
        fval = pool.map(fn, fprm)
        pool.close()
        pool.join()
    if isinstance(fval[0], (tuple, list, np.ndarray)):
        nfnv = len(fval[0])
        fvmx = np.reshape(fval, [a.size for a in grd] + [nfnv,])
        sidx = np.unravel_index(slct(fvmx.reshape((-1, nfnv)), axis=0),
                        fvmx.shape[0:-1]) + (np.array((range(nfnv))),)
        sprm = np.array([grd[k][sidx[k]] for k in range(len(grd))])
        sfvl = tuple(fvmx[sidx])
    else:
        fvmx = np.reshape(fval, [a.size for a in grd])
        sidx = np.unravel_index(slct(fvmx), fvmx.shape)
        sprm = np.array([grd[k][sidx[k]] for k in range(len(grd))])
        sfvl = fvmx[sidx]

    return sprm, sfvl, fvmx, sidx



def convdicts():
    """Access a set of example learned convolutional dictionaries.

    Returns
    -------
    cdd : dict
      A dict associating description strings with dictionaries represented
      as ndarrays

    Examples
    --------
    Print the dict keys to obtain the identifiers of the available
    dictionaries

    >>> from sporco import util
    >>> cd = util.convdicts()
    >>> print(cd.keys())
    ['G:12x12x72', 'G:8x8x16,12x12x32,16x16x48', ...]

    Select a specific example dictionary using the corresponding identifier

    >>> D = cd['G:8x8x96']
    """

    pth = os.path.join(os.path.dirname(__file__), 'data', 'convdict.npz')
    npz = np.load(pth)
    cdd = {}
    for k in list(npz.keys()):
        cdd[k] = npz[k]
    return cdd



def netgetdata(url, maxtry=3, timeout=10):
    """
    Get content of a file via a URL.

    Parameters
    ----------
    url : string
      URL of the file to be downloaded
    maxtry : int, optional (default 3)
      Maximum number of download retries
    timeout : int, optional (default 10)
      Timeout in seconds for blocking operations

    Returns
    -------
    str : io.BytesIO
      Buffered I/O stream

    Raises
    ------
    urlerror.URLError (urllib2.URLError in Python 2,
    urllib.error.URLError in Python 3)
      If the file cannot be downloaded
    """

    err = ValueError('maxtry parameter should be greater than zero')
    for ntry in range(maxtry):
        try:
            rspns = urlrequest.urlopen(url, timeout=timeout)
            cntnt = rspns.read()
            break
        except urlerror.URLError as e:
            err = e
            if not isinstance(e.reason, socket.timeout):
                raise
    else:
        raise err

    return io.BytesIO(cntnt)



class ExampleImages(object):
    """Access a set of example images."""

    def __init__(self, scaled=False, dtype=None, zoom=None, gray=False,
                 pth=None):
        """Initialise an ExampleImages object.

        Parameters
        ----------
        scaled : bool, optional (default False)
          Flag indicating whether images should be on the range [0,...,255]
          with np.uint8 dtype (False), or on the range [0,...,1] with
          np.float32 dtype (True)
        dtype : data-type or None, optional (default None)
          Desired data type of images. If `scaled` is True and `dtype` is an
          integer type, the output data type is np.float32
        zoom : float or None, optional (default None)
          Optional support rescaling factor to apply to the images
        gray : bool, optional (default False)
          Flag indicating whether RGB images should be converted to grayscale
        pth : string or None (default None)
          Path to directory containing image files. If the value is None the
          path points to a set of example images that are included with the
          package.
        """

        self.scaled = scaled
        self.dtype = dtype
        self.zoom = zoom
        self.gray = gray
        if pth is None:
            self.bpth = os.path.join(os.path.dirname(__file__), 'data')
        else:
            self.bpth = pth
        self.imglst = []
        self.grpimg = {}
        for dirpath, dirnames, filenames in  os.walk(self.bpth):
            # It would be more robust and portable to use
            # pathlib.PurePath.relative_to
            prnpth = dirpath[len(self.bpth)+1:]
            for f in filenames:
                fpth = os.path.join(dirpath, f)
                if imghdr.what(fpth) is not None:
                    gpth = os.path.join(prnpth, f)
                    self.imglst.append(gpth)
                    if prnpth not in self.grpimg:
                        self.grpimg[prnpth] = []
                    self.grpimg[prnpth].append(gpth)



    def images(self):
        """Get list of available images.

        Returns
        -------
        nlst : list
          A list of names of available images
        """

        return self.imglst



    def groups(self):
        """Get list of available image groups.

        Returns
        -------
        grp : list
          A list of names of available image groups
        """

        return list(self.grpimg.keys())



    def groupimages(self, grp):
        """Get list of available images in specified group.

        Parameters
        ----------
        grp : str
          Name of image group

        Returns
        -------
        nlst : list
          A list of names of available images in the specified group
        """

        return self.grpimg[grp]



    def image(self, fname, group=None, scaled=None, dtype=None, idxexp=None,
              zoom=None, gray=None):
        """Get named image.

        Parameters
        ----------
        fname : string
          Filename of image
        group : string or None, optional (default None)
          Name of image group
        scaled : bool or None, optional (default None)
          Flag indicating whether images should be on the range [0,...,255]
          with np.uint8 dtype (False), or on the range [0,...,1] with
          np.float32 dtype (True). If the value is None, scaling behaviour
          is determined by the `scaling` parameter passed to the object
          initializer, otherwise that selection is overridden.
        dtype : data-type or None, optional (default None)
          Desired data type of images. If `scaled` is True and `dtype` is an
          integer type, the output data type is np.float32. If the value is
          None, the data type is determined by the `dtype` parameter passed to
          the object initializer, otherwise that selection is overridden.
        idxexp :  index expression or None, optional (default None)
          An index expression selecting, for example, a cropped region of
          the requested image. This selection is applied *before* any
          `zoom` rescaling so the expression does not need to be modified when
          the zoom factor is changed.
        zoom : float or None, optional (default None)
          Optional rescaling factor to apply to the images. If the value is
          None, support rescaling behaviour is determined by the `zoom`
          parameter passed to the object initializer, otherwise that selection
          is overridden.
        gray : bool or None, optional (default None)
          Flag indicating whether RGB images should be converted to grayscale.
          If the value is None, behaviour is determined by the `gray`
          parameter passed to the object initializer.

        Returns
        -------
        img : ndarray
          Image array

        Raises
        ------
        IOError
          If the image is not accessible
        """

        if scaled is None:
            scaled = self.scaled
        if dtype is None:
            if self.dtype is None:
                dtype = np.uint8
            else:
                dtype = self.dtype
        if scaled and np.issubdtype(dtype, np.integer):
            dtype = np.float32
        if zoom is None:
            zoom = self.zoom
        if gray is None:
            gray = self.gray
        if group is None:
            pth = os.path.join(self.bpth, fname)
        else:
            pth = os.path.join(self.bpth, group, fname)

        try:
            img = np.asarray(misc.imread(pth), dtype=dtype)
        except IOError:
            raise IOError('Could not access image %s in group %s' %
                          (fname, group))

        if scaled:
            img /= 255.0
        if idxexp is not None:
            img = img[idxexp]
        if zoom is not None:
            if img.ndim == 2:
                img = sni.zoom(img, zoom)
            else:
                img = sni.zoom(img, (zoom,)*2 + (1,)*(img.ndim-2))
        if gray:
            img = rgb2gray(img)

        return img



class Timer(object):
    """Timer class supporting multiple independent labelled timers.

    The timer is based on the relative time returned by
    :func:`timeit.default_timer`.
    """

    def __init__(self, labels=None, dfltlbl='main', alllbl='all'):
        """Initialise timer object.

        Parameters
        ----------
        labels : string or list, optional (default None)
          Specify the label(s) of the timer(s) to be initialised to zero.
        dfltlbl : string, optional (default 'main')
          Set the default timer label to be used when methods are
          called without specifying a label
        alllbl : string, optional (default 'all')
          Set the label string that will be used to denote all timer labels
        """

        # Initialise current and accumulated time dictionaries
        self.t0 = {}
        self.td = {}
        # Record default label and string indicating all labels
        self.dfltlbl = dfltlbl
        self.alllbl = alllbl
        # Initialise dictionary entries for labels to be created
        # immediately
        if labels is not None:
            if not isinstance(labels, (list, tuple)):
                labels = [labels,]
            for lbl in labels:
                self.td[lbl] = 0.0
                self.t0[lbl] = None



    def start(self, labels=None):
        """Start specified timer(s).

        Parameters
        ----------
        labels : string or list, optional (default None)
          Specify the label(s) of the timer(s) to be started. If it is
          ``None``, start the default timer with label specified by the
          ``dfltlbl`` parameter of :meth:`__init__`.
        """

        # Default label is self.dfltlbl
        if labels is None:
            labels = self.dfltlbl
        # If label is not a list or tuple, create a singleton list
        # containing it
        if not isinstance(labels, (list, tuple)):
            labels = [labels,]
        # Iterate over specified label(s)
        t = timer()
        for lbl in labels:
            # On first call to start for a label, set its accumulator to zero
            if lbl not in self.td:
                self.td[lbl] = 0.0
                self.t0[lbl] = None
            # Record the time at which start was called for this lbl if
            # it isn't already running
            if self.t0[lbl] is None:
                self.t0[lbl] = t



    def stop(self, labels=None):
        """Stop specified timer(s).

        Parameters
        ----------
        labels : string or list, optional (default None)
          Specify the label(s) of the timer(s) to be stopped. If it is
          ``None``, stop the default timer with label specified by the
          ``dfltlbl`` parameter of :meth:`__init__`. If it is equal to
          the string specified by the ``alllbl`` parameter of
          :meth:`__init__`, stop all timers.
        """

        # Get current time
        t = timer()
        # Default label is self.dfltlbl
        if labels is None:
            labels = self.dfltlbl
        # All timers are affected if label is equal to self.alllbl,
        # otherwise only the timer(s) specified by label
        if labels == self.alllbl:
            labels = self.t0.keys()
        elif not isinstance(labels, (list, tuple)):
            labels = [labels,]
        # Iterate over specified label(s)
        for lbl in labels:
            if lbl not in self.t0:
                raise KeyError('Unrecognized timer key %s' % lbl)
            # If self.t0[lbl] is None, the corresponding timer is
            # already stopped, so no action is required
            if self.t0[lbl] is not None:
                # Increment time accumulator from the elapsed time
                # since most recent start call
                self.td[lbl] += t - self.t0[lbl]
                # Set start time to None to indicate timer is not running
                self.t0[lbl] = None



    def reset(self, labels=None):
        """Reset specified timer(s).

        Parameters
        ----------
        labels : string or list, optional (default None)
          Specify the label(s) of the timer(s) to be stopped. If it is
          ``None``, stop the default timer with label specified by the
          ``dfltlbl`` parameter of :meth:`__init__`. If it is equal to
          the string specified by the ``alllbl`` parameter of
          :meth:`__init__`, stop all timers.
        """

        # Get current time
        t = timer()
        # Default label is self.dfltlbl
        if labels is None:
            labels = self.dfltlbl
        # All timers are affected if label is equal to self.alllbl,
        # otherwise only the timer(s) specified by label
        if labels == self.alllbl:
            labels = self.t0.keys()
        elif not isinstance(labels, (list, tuple)):
            labels = [labels,]
        # Iterate over specified label(s)
        for lbl in labels:
            if lbl not in self.t0:
                raise KeyError('Unrecognized timer key %s' % lbl)
            # Set start time to None to indicate timer is not running
            self.t0[lbl] = None
            # Set time accumulator to zero
            self.td[lbl] = 0.0



    def elapsed(self, label=None, total=True):
        """Get elapsed time since timer start.

        Parameters
        ----------
        label : string, optional (default None)
          Specify the label of the timer for which the elapsed time is
          required.  If it is ``None``, the default timer with label
          specified by the ``dfltlbl`` parameter of :meth:`__init__`
          is selected.
        total : bool, optional (default True)
          If ``True`` return the total elapsed time since the first
          call of :meth:`start` for the selected timer, otherwise
          return the elapsed time since the most recent call of
          :meth:`start` for which there has not been a corresponding
          call to :meth:`stop`.

        Returns
        -------
        dlt : float
          Elapsed time
        """

        # Get current time
        t = timer()
        # Default label is self.dfltlbl
        if label is None:
            label = self.dfltlbl
            # Return 0.0 if default timer selected and it is not initialised
            if label not in self.t0:
                return 0.0
        # Raise exception if timer with specified label does not exist
        if label not in self.t0:
            raise KeyError('Unrecognized timer key %s' % label)
        # If total flag is True return sum of accumulated time from
        # previous start/stop calls and current start call, otherwise
        # return just the time since the current start call
        te = 0.0
        if self.t0[label] is not None:
            te = t - self.t0[label]
        if total:
            te += self.td[label]

        return te



    def labels(self):
        """Get a list of timer labels.

        Returns
        -------
        lbl : list
          List of timer labels
        """

        return self.t0.keys()



    def __str__(self):
        """Return string representation of object.

        The representation consists of a table with the following columns:

          * Timer label
          * Accumulated time from past start/stop calls
          * Time since current start call, or 'Stopped' if timer is not
            currently running
        """

        # Get current time
        t = timer()
        # Length of label field, calculated from max label length
        lfldln = max([len(lbl) for lbl in self.t0] + [len(self.dfltlbl),]) + 2
        # Header string for table of timers
        s = '%-*s  Accum.       Current\n' % (lfldln, 'Label')
        s += '-' * (lfldln + 25) + '\n'
        # Construct table of timer details
        for lbl in sorted(self.t0):
            td = self.td[lbl]
            if self.t0[lbl] is None:
                ts = ' Stopped'
            else:
                ts = ' %.2e s' % (t - self.t0[lbl])
            s += '%-*s  %.2e s  %s\n' % (lfldln, lbl, td, ts)

        return s




class ContextTimer(object):
    """A wrapper class for :class:`Timer` that enables its use as a
    context manager.

    For example, instead of

    >>> t = Timer()
    >>> t.start()
    >>> do_something()
    >>> t.stop()
    >>> elapsed = t.elapsed()

    one can use

    >>> t = Timer()
    >>> with ContextTimer(t):
    ...   do_something()
    >>> elapsed = t.elapsed()
    """

    def __init__(self, timer=None, label=None, action='StartStop'):
        """Initialise context manager timer wrapper.

        Parameters
        ----------
        timer : class:`Timer` object, optional (default None)
          Specify the timer object to be used as a context manager. If
          ``None``, a new class:`Timer` object is constructed.
        label : string, optional (default None)
          Specify the label of the timer to be used. If it is ``None``,
          start the default timer.
        action : string, optional (default 'StartStop')
          Specify actions to be taken on context entry and exit. If
          the value is 'StartStop', start the timer on entry and stop
          on exit; if it is 'StopStart', stop the timer on entry and
          start it on exit.
        """

        if action not in ['StartStop', 'StopStart']:
            raise ValueError('Unrecognized action %s' % action)
        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer
        self.label = label
        self.action = action


    def __enter__(self):
        """Start the timer and return this ContextTimer instance."""

        if self.action == 'StartStop':
            self.timer.start(self.label)
        else:
            self.timer.stop(self.label)
        return self



    def __exit__(self, type, value, traceback):
        """Stop the timer and return True if no exception was raised within
        the 'with' block, otherwise return False.
        """

        if self.action == 'StartStop':
            self.timer.stop(self.label)
        else:
            self.timer.start(self.label)
        if type:
            return False
        else:
            return True


    def elapsed(self, total=True):
        """Return the elapsed time for the timer.

        Parameters
        ----------
        total : bool, optional (default True)
          If ``True`` return the total elapsed time since the first
          call of :meth:`start` for the selected timer, otherwise
          return the elapsed time since the most recent call of
          :meth:`start` for which there has not been a corresponding
          call to :meth:`stop`.

        Returns
        -------
        dlt : float
          Elapsed time
        """

        return self.timer.elapsed(self.label, total=total)
