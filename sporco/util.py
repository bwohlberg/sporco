#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Utility functions"""

from __future__ import division
from builtins import range
from builtins import object

import numpy as np
from scipy import misc
from timeit import default_timer as timer
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sporco.linalg as sla

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


def plot(dat, title=None, xlbl=None, ylbl=None, lgnd=None, lglc=None,
         ptyp='plot', block=False, fgnm=None, fgsz=(12, 12)):
    """Plot columns of array.

    Parameters
    ----------
    dat : array_like
        Data to plot. Each column is plotted as a separate curve.
    title : string, optional (default=None)
        Figure title.
    xlbl : string, optional (default=None)
        Label for x-axis
    ylbl : string, optional (default=None)
        Label for y-axis
    lgnd : list of strings, optional (default=None)
        List of legend string
    lglc : string, optional (default=None)
        Legend location string
    ptyp : string, optional (default='plot')
        Plot type specification (options are 'plot', 'semilogx', 'semilogy',
        and 'loglog')
    block : boolean, optional (default=False)
        If True, the function only returns when the figure is closed.
    fgnm : integer, optional (default=None)
        Figure number of figure
    fgsz : tuple (width,height), optional (default=(12,12))
        Specify figure dimensions in inches.
    cbar : boolean, optional (default=False)
        Flag indicating whether to display colorbar

    Returns
    -------
    fig : matplotlib.figure.Figure
      Figure object for this figure.
    """

    plttyp = {'plot' : plt.plot, 'semilogx' : plt.semilogx,
              'semilogy' : plt.semilogy, 'loglog' : plt.loglog}
    fig = plt.figure(num=fgnm, figsize=fgsz)
    if ptyp in plttyp:
        plttyp[ptyp](dat)
    else:
        raise ValueError("Invalid plot type '%s'" % ptyp)
    if title is not None:
        plt.title(title)
    if xlbl is not None:
        plt.xlabel(xlbl)
    if ylbl is not None:
        plt.ylabel(ylbl)
    if lgnd is not None:
        plt.legend(lgnd, loc=lglc)
    plt.show(block=block)

    return fig



def imview(img, title=None, block=False, cmap=None, fgnm=None, fgsz=(12, 12),
           cbar=False, axes=None):
    """Display an image.

    Pixel values are displayed when the pointer is over valid image data.
    The figure is closed on key entry 'q'.

    Parameters
    ----------
    img : array_like, shape (Nr, Nc) or (Nr, Nc, 3) or (Nr, Nc, 4)
        Image to display.
    title : string, optional (default=None)
        Figure title.
    block : boolean, optional (default=False)
        If True, the function only returns when the figure is closed.
    cmap : matplotlib.cm colormap, optional (default=None)
        Colour map for image. If none specifed, defaults to cm.Greys_r
        for monochrome image
    fgnm : integer, optional (default=None)
        Figure number of figure
    fgsz : tuple (width,height), optional (default=(12,12))
        Specify figure dimensions in inches.
    cbar : boolean, optional (default=False)
        Flag indicating whether to display colorbar
    axes : matplotlib.axes.Axes object, optional (default=None)
        If specified the new figure shares axes with the specified axes of
        an existing figure so that a zoom is shared across both figures

    Returns
    -------
    fig : matplotlib.figure.Figure
      Figure object for this figure.
    ax : matplotlib.axes.Axes
      Axes for this figure suitable for passing as share parameter of another
      imview call.
    """

    imgd = np.copy(img)

    if cmap is None and img.ndim == 2:
        cmap = cm.Greys_r

    if img.dtype.type == np.uint16 or img.dtype.type == np.int16:
        imgd = np.float16(imgd)

    fig = plt.figure(num=fgnm, figsize=fgsz)
    fig.clf()

    if axes is not None:
        ax = plt.subplot(sharex=axes, sharey=axes)
        axes.set_adjustable('box-forced')
        ax.set_adjustable('box-forced')

    plt.imshow(imgd, cmap=cmap, interpolation="nearest",
               vmin=imgd.min(), vmax=imgd.max())
    if title is not None:
        plt.title(title)
    if cbar:
        orient = 'vertical' if img.shape[0] >= img.shape[1] else 'horizontal'
        plt.colorbar(orientation=orient, shrink=0.8)


    def format_coord(x, y):
        nr, nc = imgd.shape[0:2]
        col = int(x+0.5)
        row = int(y+0.5)
        if col >= 0 and col < nc and row >= 0 and row < nr:
            z = imgd[row, col]
            if imgd.ndim == 2:
                return 'x=%.2f, y=%.2f, z=%.2f' % (x, y, z)
            else:
                return 'x=%.2f, y=%.2f, z=(%.2f,%.2f,%.2f)' % \
                    sum(((x,), (y,), tuple(z)), ())
        else:
            return 'x=%.2f, y=%.2f'%(x, y)

    def press(event):
        if event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', press)

    plt.axis('off')
    ax = plt.gca()
    ax.format_coord = format_coord
    plt.show(block=block)
    return fig, ax



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
    else:
        axisM = 2

    # Construct dictionary atom size vector if not provided
    if sz is None:
        sz = np.tile(np.array(dsz[0:2]).reshape([2, 1]), (1, D.shape[axisM]))
    else:
        sz = np.array(sum(tuple((x[0:2],) * x[2] for x in sz), ())).T

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

    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])



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
    """Construct header and format details for status display of an iterative
    solver
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
    # Compute length of seperator string
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
    convolutional dictionaries"""

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
