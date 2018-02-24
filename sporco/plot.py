# -*- coding: utf-8 -*-
# Copyright (C) 2015-2018 by Brendt Wohlberg <brendt@ieee.org>
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

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import figure, subplot, subplots, gcf, gca, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
try:
    import mpldatacursor as mpldc
except ImportError:
    have_mpldc = False
else:
    have_mpldc = True


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def attach_keypress(fig):
    """
    Attach a key press event handler that configures keys for closing a
    figure and changing the figure size. Keys 'e' and 'c' respectively
    expand and contract the figure, and key 'q' closes it.

    **Note:** Resizing may not function correctly with all matplotlib
    backends (a
    `bug <https://github.com/matplotlib/matplotlib/issues/10083>`__
    has been reported).

    Parameters
    ----------
    fig : :class:`matplotlib.figure.Figure` object
        Figure to which event handling is to be attached
    """

    def press(event):
        a = 1.1
        if event.key == 'q':
            plt.close(fig)
        elif event.key == 'e':
            fig.set_size_inches(a*fig.get_size_inches(), forward=True)
        elif event.key == 'c':
            fig.set_size_inches(fig.get_size_inches()/a, forward=True)

    # Avoid multiple even handlers attached to the same figure
    if not hasattr(fig, '_sporco_keypress_cid'):
        cid = fig.canvas.mpl_connect('key_press_event', press)
        fig._sporco_keypress_cid = cid



def plot(y, x=None, ptyp='plot', xlbl=None, ylbl=None, title=None,
         lgnd=None, lglc=None, lwidth=1.5, lstyle='solid', msize=6.0,
         mstyle='None', fgsz=None, fgnm=None, fig=None, ax=None):
    """
    Plot points or lines in 2D. If a figure object is specified then the
    plot is drawn in that figure, and fig.show() is not called. The figure
    is closed on key entry 'q'.

    Parameters
    ----------
    y : array_like
        1d or 2d array of data to plot. If a 2d array, each column is
        plotted as a separate curve.
    x : array_like, optional (default None)
        Values for x-axis of the plot
    ptyp : string, optional (default 'plot')
        Plot type specification (options are 'plot', 'semilogx',
        'semilogy', and 'loglog')
    xlbl : string, optional (default None)
        Label for x-axis
    ylbl : string, optional (default None)
        Label for y-axis
    title : string, optional (default None)
        Figure title
    lgnd : list of strings, optional (default None)
        List of legend string
    lglc : string, optional (default None)
        Legend location string
    lwidth : float, optional (default 1.5)
        Line width
    lstyle : string, optional (default 'solid')
        Line style (see :class:`matplotlib.lines.Line2D`)
    msize : float, optional (default 6.0)
        Marker size
    mstyle : string, optional (default 'None')
        Marker style (see :mod:`matplotlib.markers`)
    fgsz : tuple (width,height), optional (default None)
        Specify figure dimensions in inches
    fgnm : integer, optional (default None)
        Figure number of figure
    fig : :class:`matplotlib.figure.Figure` object, optional (default None)
        Draw in specified figure instead of creating one
    ax : :class:`matplotlib.axes.Axes` object, optional (default None)
        Plot in specified axes instead of current axes of figure

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure` object
      Figure object for this figure
    ax : :class:`matplotlib.axes.Axes` object
      Axes object for this plot
    """

    figp = fig
    if fig is None:
        fig = plt.figure(num=fgnm, figsize=fgsz)
        fig.clf()
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()

    if ptyp not in ('plot', 'semilogx', 'semilogy', 'loglog'):
        raise ValueError("Invalid plot type '%s'" % ptyp)
    pltmth = getattr(ax, ptyp)
    if x is None:
        pltln = pltmth(y, linewidth=lwidth, linestyle=lstyle,
                       marker=mstyle, markersize=msize)
    else:
        pltln = pltmth(x, y, linewidth=lwidth, linestyle=lstyle,
                       marker=mstyle, markersize=msize)

    if title is not None:
        ax.set_title(title)
    if xlbl is not None:
        ax.set_xlabel(xlbl)
    if ylbl is not None:
        ax.set_ylabel(ylbl)
    if lgnd is not None:
        ax.legend(lgnd, loc=lglc)

    attach_keypress(fig)

    if have_mpldc:
        mpldc.datacursor(pltln)

    if figp is None:
        fig.show()

    return fig, ax



def surf(z, x=None, y=None, elev=None, azim=None, xlbl=None, ylbl=None,
         zlbl=None, title=None, lblpad=8.0, cntr=None, cmap=None,
         fgsz=None, fgnm=None, fig=None, ax=None):
    """
    Plot a 2D surface in 3D. If a figure object is specified then the
    surface is drawn in that figure, and fig.show() is not called. The
    figure is closed on key entry 'q'.

    Parameters
    ----------
    z : array_like
        2d array of data to plot
    x : array_like, optional (default None)
        Values for x-axis of the plot
    y : array_like, optional (default None)
        Values for y-axis of the plot
    elev : float
        Elevation angle (in degrees) in the z plane
    azim : foat
        Azimuth angle  (in degrees) in the x,y plane
    xlbl : string, optional (default None)
        Label for x-axis
    ylbl : string, optional (default None)
        Label for y-axis
    zlbl : string, optional (default None)
        Label for z-axis
    title : string, optional (default None)
        Figure title
    lblpad : float, optional (default 8.0)
        Label padding
    cntr : int or sequence of ints, optional (default None)
        If not None, plot contours of the surface on the lower end of
        the z-axis. An int specifies the number of contours to plot, and
        a sequence specifies the specific contour levels to plot.
    cmap : :class:`matplotlib.colors.Colormap` object, optional (default None)
        Colour map for surface. If none specifed, defaults to cm.coolwarm
    fgsz : tuple (width,height), optional (default None)
        Specify figure dimensions in inches
    fgnm : integer, optional (default None)
        Figure number of figure
    fig : :class:`matplotlib.figure.Figure` object, optional (default None)
        Draw in specified figure instead of creating one
    ax : :class:`matplotlib.axes.Axes` object, optional (default None)
        Plot in specified axes instead of creating one

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure` object
      Figure object for this figure
    ax : :class:`matplotlib.axes.Axes` object
      Axes object for this plot
    """

    figp = fig
    if fig is None:
        fig = plt.figure(num=fgnm, figsize=fgsz)
        fig.clf()
        ax = plt.axes(projection='3d')
    else:
        if ax is None:
            ax = plt.axes(projection='3d')
        else:
            # See https://stackoverflow.com/a/43563804
            #     https://stackoverflow.com/a/35221116
            if ax.name != '3d':
                ax.remove()
                ax = fig.add_subplot(*ax.get_geometry(), projection='3d')

    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)

    if cmap is None:
        cmap = cm.coolwarm

    if x is None:
        x = range(z.shape[1])
    if y is None:
        y = range(z.shape[0])

    xg, yg = np.meshgrid(x, y)
    ax.plot_surface(xg, yg, z, rstride=1, cstride=1, cmap=cmap)

    if cntr is not None:
        offset = np.around(z.min() - 0.2 * (z.max() - z.min()), 3)
        ax.contour(xg, yg, z, cntr, linewidths=2, cmap=cmap, linestyles="solid",
                   offset=offset)
        ax.set_zlim(offset, ax.get_zlim()[1])

    if title is not None:
        ax.set_title(title)
    if xlbl is not None:
        ax.set_xlabel(xlbl, labelpad=lblpad)
    if ylbl is not None:
        ax.set_ylabel(ylbl, labelpad=lblpad)
    if zlbl is not None:
        ax.set_zlabel(zlbl, labelpad=lblpad)

    attach_keypress(fig)

    if figp is None:
        fig.show()

    return fig, ax



def contour(z, x=None, y=None, v=5, xlbl=None, ylbl=None, title=None,
            cfntsz=10, lfntsz=None, intrp='bicubic', alpha=0.5, cmap=None,
            vmin=None, vmax=None, fgsz=None, fgnm=None, fig=None, ax=None):
    """
    Contour plot of a 2D surface. If a figure object is specified then the
    plot is drawn in that figure, and fig.show() is not called. The figure
    is closed on key entry 'q'.

    Parameters
    ----------
    z : array_like
        2d array of data to plot
    x : array_like, optional (default None)
        Values for x-axis of the plot
    y : array_like, optional (default None)
        Values for y-axis of the plot
    v : int or sequence of ints, optional (default 5)
        An int specifies the number of contours to plot, and a sequence
        specifies the specific contour levels to plot.
    xlbl : string, optional (default None)
        Label for x-axis
    ylbl : string, optional (default None)
        Label for y-axis
    title : string, optional (default None)
        Figure title
    cfntsz : int or None, optional (default 10)
        Contour label font size. No contour labels are displayed if
        set to 0 or None.
    lfntsz : int, optional (default None)
        Axis label font size. The default font size is used if set to None.
    intrp : string, optional (default 'bicubic')
        Specify type of interpolation used to display image underlying
        contours (see ``interpolation`` parameter of
        :meth:`matplotlib.axes.Axes.imshow`)
    alpha : float, optional (default 0.5)
        Underlying image display alpha value
    cmap : :class:`matplotlib.colors.Colormap`, optional (default None)
        Colour map for surface. If none specifed, defaults to cm.coolwarm
    vmin, vmax : float, optional (default None)
        Set upper and lower bounds for the colour map (see the corresponding
        parameters of :meth:`matplotlib.axes.Axes.imshow`)
    fgsz : tuple (width,height), optional (default None)
        Specify figure dimensions in inches
    fgnm : integer, optional (default None)
        Figure number of figure
    fig : :class:`matplotlib.figure.Figure` object, optional (default None)
        Draw in specified figure instead of creating one
    ax : :class:`matplotlib.axes.Axes` object, optional (default None)
        Plot in specified axes instead of current axes of figure

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure` object
      Figure object for this figure
    ax : :class:`matplotlib.axes.Axes` object
      Axes object for this plot
    """

    figp = fig
    if fig is None:
        fig = plt.figure(num=fgnm, figsize=fgsz)
        fig.clf()
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()

    if cmap is None:
        cmap = cm.coolwarm

    if x is None:
        x = np.arange(z.shape[1])
    else:
        x = np.array(x)
    if y is None:
        y = np.arange(z.shape[0])
    else:
        y = np.array(y)
    xg, yg = np.meshgrid(x, y)

    cntr = ax.contour(xg, yg, z, v, colors='black')
    if cfntsz is not None and cfntsz > 0:
        plt.clabel(cntr, inline=True, fontsize=cfntsz)
    im = ax.imshow(z, origin='lower', interpolation=intrp, aspect='auto',
                extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap,
                vmin=vmin, vmax=vmax, alpha=alpha)

    if title is not None:
        ax.set_title(title)
    if xlbl is not None:
        ax.set_xlabel(xlbl, fontsize=lfntsz)
    if ylbl is not None:
        ax.set_ylabel(ylbl, fontsize=lfntsz)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im, ax=ax, cax=cax)

    attach_keypress(fig)

    if have_mpldc:
        mpldc.datacursor()

    if figp is None:
        fig.show()

    return fig, ax



def imview(img, title=None, copy=True, fltscl=False, intrp='nearest',
           norm=None, cbar=False, cmap=None, fgsz=None, fgnm=None,
           fig=None, ax=None):
    """
    Display an image. Pixel values are displayed when the pointer is over
    valid image data.  If a figure object is specified then the image is
    drawn in that figure, and fig.show() is not called. The figure is
    closed on key entry 'q'.

    Parameters
    ----------
    img : array_like, shape (Nr, Nc) or (Nr, Nc, 3) or (Nr, Nc, 4)
        Image to display
    title : string, optional (default None)
        Figure title
    copy : boolean, optional (default True)
        If True, create a copy of input `img` as a reference for displayed
        pixel values, ensuring that displayed values do not change when the
        array changes in the calling scope. Set this flag to False if the
        overhead of an additional copy of the input image is not acceptable.
    fltscl : boolean, optional (default False)
        If True, rescale and shift floating point arrays to [0,1]
    intrp : string, optional (default 'nearest')
        Specify type of interpolation used to display image (see
        ``interpolation`` parameter of :meth:`matplotlib.axes.Axes.imshow`)
    norm : :class:`matplotlib.colors.Normalize` object, optional (default None)
        Specify the :class:`matplotlib.colors.Normalize` instance used to
        scale pixel values for input to the colour map
    cbar : boolean, optional (default False)
        Flag indicating whether to display colorbar
    cmap : :class:`matplotlib.colors.Colormap`, optional (default None)
        Colour map for image. If none specifed, defaults to cm.Greys_r
        for monochrome image
    fgsz : tuple (width,height), optional (default None)
        Specify figure dimensions in inches
    fgnm : integer, optional (default None)
        Figure number of figure
    fig : :class:`matplotlib.figure.Figure` object, optional (default None)
        Draw in specified figure instead of creating one
    ax : :class:`matplotlib.axes.Axes` object, optional (default None)
        Plot in specified axes instead of current axes of figure

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure` object
      Figure object for this figure
    ax : :class:`matplotlib.axes.Axes` object
      Axes object for this plot
    """

    if img.ndim > 2 and img.shape[2] != 3:
        raise ValueError('Argument img must be an Nr x Nc array or an '
                         'Nr x Nc x 3 array')

    figp = fig
    if fig is None:
        fig = plt.figure(num=fgnm, figsize=fgsz)
        fig.clf()
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()

    ax.set_adjustable('box-forced')

    imgd = img.copy()
    if copy:
        # Keep a separate copy of the input image so that the original
        # pixel values can be display rather than the scaled pixel
        # values that are actually plotted.
        img = img.copy()

    if cmap is None and img.ndim == 2:
        cmap = cm.Greys_r

    if np.issubdtype(img.dtype, np.floating):
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

    if norm is None:
        im = ax.imshow(imgd, cmap=cmap, interpolation=intrp, vmin=imgd.min(),
                       vmax=imgd.max())
    else:
        im = ax.imshow(imgd, cmap=cmap, interpolation=intrp, norm=norm)

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if title is not None:
        ax.set_title(title)

    if cbar or cbar is None:
        orient = 'vertical' if img.shape[0] >= img.shape[1] else 'horizontal'
        pos = 'right' if orient == 'vertical' else 'bottom'
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(pos, size="5%", pad=0.2)
        if cbar is None:
            # See http://chris35wills.github.io/matplotlib_axis
            if hasattr(cax, 'set_facecolor'):
                cax.set_facecolor('none')
            else:
                cax.set_axis_bgcolor('none')
            for axis in ['top', 'bottom', 'left', 'right']:
                cax.spines[axis].set_linewidth(0)
            cax.set_xticks([])
            cax.set_yticks([])
        else:
            plt.colorbar(im, ax=ax, cax=cax, orientation=orient)

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

    ax.format_coord = format_coord

    attach_keypress(fig)

    if have_mpldc:
        mpldc.datacursor(display='single')

    if figp is None:
        fig.show()

    return fig, ax



def close(fig=None):
    """
    Close figure(s). If a figure object reference or figure number is
    provided, close the specified figure, otherwise close all figures.

    Parameters
    ----------
    fig : :class:`matplotlib.figure.Figure` object or integer,\
          optional (default None)
        Figure object or number of figure to close
    """

    if fig is None:
        plt.close('all')
    else:
        plt.close(fig)



def set_ipython_plot_backend(backend='qt'):
    """
    Set matplotlib backend within an ipython shell. Ths function has the
    same effect as the line magic ``%matplotlib [backend]`` but is called
    as a function and includes a check to determine whether the code is
    running in an ipython shell, so that it can safely be used within a
    normal python script since it has no effect when not running in an
    ipython shell.

    Parameters
    ----------
    backend : string, optional (default 'qt')
      Name of backend to be passed to the ``%matplotlib`` line magic
      command
    """

    from sporco.util import in_ipython
    if in_ipython():
        # See https://stackoverflow.com/questions/35595766
        get_ipython().run_line_magic('matplotlib', backend)



def set_notebook_plot_backend(backend='inline'):
    """
    Set matplotlib backend within a Jupyter Notebook shell. Ths function
    has the same effect as the line magic ``%matplotlib [backend]`` but is
    called as a function and includes a check to determine whether the code
    is running in a notebook shell, so that it can safely be used within a
    normal python script since it has no effect when not running in a
    notebook shell.

    Parameters
    ----------
    backend : string, optional (default 'inline')
      Name of backend to be passed to the ``%matplotlib`` line magic
      command
    """

    from sporco.util import in_notebook
    if in_notebook():
        # See https://stackoverflow.com/questions/35595766
        get_ipython().run_line_magic('matplotlib', backend)



def config_notebook_plotting():
    """
    Configure plotting functions for inline plotting within a Jupyter
    Notebook shell. This function has no effect when not within a
    notebook shell, and may therefore be used within a normal python
    script.
    """

    # Check whether running within a notebook shell and have
    # not already monkey patched the plot function
    from sporco.util import in_notebook
    module = sys.modules[__name__]
    if in_notebook() and module.plot.__name__ == 'plot':

        # Set inline backend (i.e. %matplotlib inline) if in a notebook shell
        set_notebook_plot_backend()

        # Replace plot function with a wrapper function that discards
        # its return value (within a notebook with inline plotting, plots
        # are duplicated if the return value from the original function is
        # not assigned to a variable)
        plot_original = module.plot
        def plot_wrap(*args, **kwargs):
            plot_original(*args, **kwargs)
        module.plot = plot_wrap

        # Replace surf function with a wrapper function that discards
        # its return value (see comment for plot function)
        surf_original = module.surf
        def surf_wrap(*args, **kwargs):
            surf_original(*args, **kwargs)
        module.surf = surf_wrap

        # Replace contour function with a wrapper function that discards
        # its return value (see comment for plot function)
        contour_original = module.contour
        def contour_wrap(*args, **kwargs):
            contour_original(*args, **kwargs)
        module.contour = contour_wrap

        # Replace imview function with a wrapper function that discards
        # its return value (see comment for plot function)
        imview_original = module.imview
        def imview_wrap(*args, **kwargs):
            imview_original(*args, **kwargs)
        module.imview = imview_wrap

        # Disable figure show method (results in a warning if used within
        # a notebook with inline plotting)
        import matplotlib.figure
        def show_disable(self):
            pass
        matplotlib.figure.Figure.show = show_disable
