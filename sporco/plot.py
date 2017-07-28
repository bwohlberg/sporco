# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Plotting/visualisation functions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import figure, subplot, savefig
from mpl_toolkits.mplot3d import Axes3D
try:
    import mpldatacursor as mpldc
except ImportError:
    have_mpldc = False
else:
    have_mpldc = True


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def plot(dat, x=None, title=None, xlbl=None, ylbl=None, lgnd=None, lglc=None,
         ptyp='plot', lwidth=1.5, lstyle='solid', msize=6.0, mstyle='None',
         block=False, fgrf=None, fgnm=None, fgsz=None):
    """
    Plot columns of array.

    Parameters
    ----------
    dat : array_like
        Data to plot. Each column is plotted as a separate curve.
    x : array_like, optional (default None)
        Values for x-axis of the plot
    title : string, optional (default None)
        Figure title
    xlbl : string, optional (default None)
        Label for x-axis
    ylbl : string, optional (default None)
        Label for y-axis
    lgnd : list of strings, optional (default None)
        List of legend string
    lglc : string, optional (default None)
        Legend location string
    ptyp : string, optional (default 'plot')
        Plot type specification (options are 'plot', 'semilogx',
        'semilogy', and 'loglog')
    lwidth : float, optional (default 1.5)
        Line width
    lstyle : string, optional (default 'solid')
        Line style (see :class:`matplotlib.lines.Line2D`)
    msize : float, optional (default 6.0)
        Marker size
    mstyle : string, optional (default 'None')
        Marker style (see :mod:`matplotlib.markers`)
    block : boolean, optional (default False)
        If True, the function only returns when the figure is closed
    fgrf : :class:`matplotlib.figure.Figure` object, optional (default None)
        Draw in specified figure instead of creating one
    fgnm : integer, optional (default None)
        Figure number of figure
    fgsz : tuple (width,height), optional (default (12,12))
        Specify figure dimensions in inches.
    cbar : boolean, optional (default False)
        Flag indicating whether to display colorbar

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
      Figure object for this figure
    """

    if fgrf is None:
        fig = plt.figure(num=fgnm, figsize=fgsz)
        fig.clf()
    else:
        fig = fgrf

    plttyp = {'plot' : plt.plot, 'semilogx' : plt.semilogx,
              'semilogy' : plt.semilogy, 'loglog' : plt.loglog}
    if ptyp in plttyp:
        if x is None:
            pltln = plttyp[ptyp](dat, linewidth=lwidth, linestyle=lstyle,
                                 marker=mstyle, markersize=msize)
        else:
            pltln = plttyp[ptyp](x, dat, linewidth=lwidth, linestyle=lstyle,
                                 marker=mstyle, markersize=msize)
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

    def press(event):
        if event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', press)

    if have_mpldc:
        mpldc.datacursor(pltln)

    if fgrf is None:
        plt.show(block=block)

    return fig



def surf(z, x=None, y=None, title=None, xlbl=None, ylbl=None, zlbl=None,
         lblpad=8.0, block=False, cmap=None, fgrf=None, axrf=None,
         fgnm=None, fgsz=None):
    """
    Plot columns of array.

    Parameters
    ----------
    z : array_like
        2d array of data to plot
    x : array_like, optional (default None)
        Values for x-axis of the plot
    y : array_like, optional (default None)
        Values for y-axis of the plot
    title : string, optional (default None)
        Figure title
    xlbl : string, optional (default None)
        Label for x-axis
    ylbl : string, optional (default None)
        Label for y-axis
    zlbl : string, optional (default None)
        Label for z-axis
    lblpad : float, optional (default 8.0)
        Label padding
    block : boolean, optional (default False)
        If True, the function only returns when the figure is closed
    cmap : :class:`matplotlib.colors.Colormap`, optional (default None)
        Colour map for surface. If none specifed, defaults to cm.coolwarm
    fgrf : :class:`matplotlib.figure.Figure` object, optional (default None)
        Draw in specified figure instead of creating one
    axrf : :class:`matplotlib.axes.Axes` object, optional (default None)
        Plot in specified axes instead of creating one
    fgnm : integer, optional (default None)
        Figure number of figure
    fgsz : tuple (width,height), optional (default None)
        Specify figure dimensions in inches

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
      Figure object for this figure
    ax : :class:`matplotlib.axes.Axes`
      Axes for this figure
    """

    if fgrf is None:
        fig = plt.figure(num=fgnm, figsize=fgsz)
        fig.clf()
    else:
        fig = fgrf

    if axrf is None:
        ax = plt.axes(projection='3d')
    else:
        ax = axrf

    if x is None:
        x = range(z.shape[1])
    if y is None:
        y = range(z.shape[0])

    if cmap is None:
        cmap = cm.coolwarm

    xg, yg = np.meshgrid(x, y)
    ax.plot_surface(xg, yg, z, rstride=1, cstride=1, cmap=cmap)

    if title is not None:
        plt.title(title)
    if xlbl is not None:
        ax.set_xlabel(xlbl, labelpad=lblpad)
    if ylbl is not None:
        ax.set_ylabel(ylbl, labelpad=lblpad)
    if zlbl is not None:
        ax.set_zlabel(zlbl, labelpad=lblpad)

    def press(event):
        if event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', press)

    if fgrf is None:
        plt.show(block=block)

    return fig, ax



def imview(img, title=None, block=False, copy=True, fltscl=False, fgrf=None,
           fgnm=None, fgsz=(12, 12), norm=None, cmap=None, intrp='nearest',
           cbar=False, axes=None):
    """
    Display an image.

    Pixel values are displayed when the pointer is over valid image
    data.  If a figure object is specified then the image is drawn in
    that figure, and plt.show() is not called.  The figure is closed
    on key entry 'q'.

    Parameters
    ----------
    img : array_like, shape (Nr, Nc) or (Nr, Nc, 3) or (Nr, Nc, 4)
        Image to display
    title : string, optional (default None)
        Figure title
    block : boolean, optional (default False)
        If True, the function only returns when the figure is closed
    copy : boolean, optional (default True)
        If True, create a copy of input `img` as a reference for displayed
        pixel values, ensuring that displayed values do not change when the
        array changes in the calling scope. Set this flag to False if the
        overhead of an additional copy of the input image is not acceptable.
    fltscl : boolean, optional (default False)
        If True, rescale and shift floating point arrays to [0,1]
    fgrf : :class:`matplotlib.figure.Figure` object, optional (default None)
        Draw in specified figure instead of creating one
    fgnm : integer, optional (default None)
        Figure number of figure
    fgsz : tuple (width,height), optional (default (12,12))
        Specify figure dimensions in inches
    norm : :class:`matplotlib.colors.Normalize` object, optional (default None)
        Specify the :class:`matplotlib.colors.Normalize` instance used to
        scale pixel values for input to the colour map
    cmap : :class:`matplotlib.colors.Colormap`, optional (default None)
        Colour map for image. If none specifed, defaults to cm.Greys_r
        for monochrome image
    intrp : string, optional (default 'nearest')
        Specify type of interpolation used to display image (see
        ``interpolation`` parameter of :meth:`matplotlib.axes.Axes.imshow`)
    cbar : boolean, optional (default False)
        Flag indicating whether to display colorbar
    axes : :class:`matplotlib.axes.Axes` object, optional (default None)
        If specified the new figure shares axes with the specified axes of
        an existing figure so that a zoom is shared across both figures

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
      Figure object for this figure
    ax : :class:`matplotlib.axes.Axes`
      Axes for this figure suitable for passing as ``axes`` parameter of
      another imview call
    """

    if img.ndim > 2 and img.shape[2] != 3:
        raise ValueError('Argument img must be an Nr x Nc array or an '
                         'Nr x Nc x 3 array')

    imgd = img.copy()
    if copy:
        # Keep a separate copy of the input image so that the original
        # pixel values can be display rather than the scaled pixel
        # values that are actually plotted.
        img = img.copy()

    if cmap is None and img.ndim == 2:
        cmap = cm.Greys_r

    if np.issubdtype(img.dtype, np.float):
        if fltscl:
            imgd -= imgd.min()
            imgd /= imgd.max()
        if img.ndim > 2:
            imgd = np.clip(imgd, 0.0, 1.0)
    elif img.dtype == np.uint16:
        imgd = np.float16(imgd) / np.iinfo(np.uint16).max
    elif img.dtype == np.int16:
        imgd = np.float16(imgd) - imgd.min()
        imgd /= imgd.max()

    if fgrf is None:
        fig = plt.figure(num=fgnm, figsize=fgsz)
        fig.clf()
    else:
        fig = fgrf

    if axes is not None:
        ax = plt.subplot(sharex=axes, sharey=axes)
        axes.set_adjustable('box-forced')
        ax.set_adjustable('box-forced')

    if norm is None:
        plt.imshow(imgd, cmap=cmap, interpolation=intrp, vmin=imgd.min(),
                   vmax=imgd.max())
    else:
        plt.imshow(imgd, cmap=cmap, interpolation=intrp, norm=norm)

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
            z = img[row, col]
            if imgd.ndim == 2:
                return 'x=%.2f, y=%.2f, z=%.2f' % (x, y, z)
            else:
                return 'x=%.2f, y=%.2f, z=(%.2f,%.2f,%.2f)' % \
                    sum(((x,), (y,), tuple(z)), ())
        else:
            return 'x=%.2f, y=%.2f' % (x, y)

    def press(event):
        if event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', press)

    plt.axis('off')
    ax = plt.gca()
    ax.format_coord = format_coord

    if have_mpldc:
        mpldc.datacursor(display='single')

    if fgrf is None:
        plt.show(block=block)

    return fig, ax



def close(fgrf=None):
    """
    Close figure(s).

    Parameters
    ----------
    fgrf : :class:`matplotlib.figure.Figure` object or integer or None, \
           optional (default None)
        If a figure object reference or figure number is provided, close the
        specified figure, otherwise close all figures
    """

    if fgrf is None:
        plt.close("all")
    else:
        plt.close(fgrf)
