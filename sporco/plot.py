# -*- coding: utf-8 -*-
# Copyright (C) 2015-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Plotting/visualisation functions"""

from __future__ import absolute_import, division, print_function
from builtins import range

import sys
import numpy as np
import matplotlib
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



__all__ = ['plot', 'surf', 'contour', 'imview', 'close',
           'set_ipython_plot_backend', 'set_notebook_plot_backend',
           'config_notebook_plotting']



def attach_keypress(fig, scaling=1.1):
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
    scaling : float, optional (default 1.1)
      Scaling factor for figure size changes

    Returns
    -------
    press : function
      Key press event handler function
    """

    def press(event):
        if event.key == 'q':
            plt.close(fig)
        elif event.key == 'e':
            fig.set_size_inches(scaling * fig.get_size_inches(), forward=True)
        elif event.key == 'c':
            fig.set_size_inches(fig.get_size_inches() / scaling, forward=True)

    # Avoid multiple event handlers attached to the same figure
    if not hasattr(fig, '_sporco_keypress_cid'):
        cid = fig.canvas.mpl_connect('key_press_event', press)
        fig._sporco_keypress_cid = cid

    return press



def attach_zoom(ax, scaling=2.0):
    """
    Attach an event handler that supports zooming within a plot using
    the mouse scroll wheel.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes` object
      Axes to which event handling is to be attached
    scaling : float, optional (default 2.0)
      Scaling factor for zooming in and out

    Returns
    -------
    zoom : function
      Mouse scroll wheel event handler function
    """

    # See https://stackoverflow.com/questions/11551049
    def zoom(event):
        # Get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        # Get event location
        xdata = event.xdata
        ydata = event.ydata
        # Return if cursor is not over valid region of plot
        if xdata is None or ydata is None:
            return

        if event.button == 'up':
            # Deal with zoom in
            scale_factor = 1.0 / scaling
        elif event.button == 'down':
            # Deal with zoom out
            scale_factor = scaling

        # Get distance from the cursor to the edge of the figure frame
        x_left = xdata - cur_xlim[0]
        x_right = cur_xlim[1] - xdata
        y_top = ydata - cur_ylim[0]
        y_bottom = cur_ylim[1] - ydata

        # Calculate new x and y limits
        new_xlim = (xdata - x_left * scale_factor,
                    xdata + x_right * scale_factor)
        new_ylim = (ydata - y_top * scale_factor,
                    ydata + y_bottom * scale_factor)

        # Ensure that x limit range is no larger than that of the reference
        if np.diff(new_xlim) > np.diff(zoom.xlim_ref):
            new_xlim *= np.diff(zoom.xlim_ref) / np.diff(new_xlim)
        # Ensure that lower x limit is not less than that of the reference
        if new_xlim[0] < zoom.xlim_ref[0]:
            new_xlim += np.array(zoom.xlim_ref[0] - new_xlim[0])
        # Ensure that upper x limit is not greater than that of the reference
        if new_xlim[1] > zoom.xlim_ref[1]:
            new_xlim -= np.array(new_xlim[1] - zoom.xlim_ref[1])

        # Ensure that ylim tuple has the smallest value first
        if zoom.ylim_ref[1] < zoom.ylim_ref[0]:
            ylim_ref = zoom.ylim_ref[::-1]
            new_ylim = new_ylim[::-1]
        else:
            ylim_ref = zoom.ylim_ref

        # Ensure that y limit range is no larger than that of the reference
        if np.diff(new_ylim) > np.diff(ylim_ref):
            new_ylim *= np.diff(ylim_ref) / np.diff(new_ylim)
        # Ensure that lower y limit is not less than that of the reference
        if new_ylim[0] < ylim_ref[0]:
            new_ylim += np.array(ylim_ref[0] - new_ylim[0])
        # Ensure that upper y limit is not greater than that of the reference
        if new_ylim[1] > ylim_ref[1]:
            new_ylim -= np.array(new_ylim[1] - ylim_ref[1])

        # Return the ylim tuple to its original order
        if zoom.ylim_ref[1] < zoom.ylim_ref[0]:
            new_ylim = new_ylim[::-1]

        # Set new x and y limits
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)

        # Force redraw
        ax.figure.canvas.draw()

    # Record reference x and y limits prior to any zooming
    zoom.xlim_ref = ax.get_xlim()
    zoom.ylim_ref = ax.get_ylim()

    # Get figure for specified axes and attach the event handler
    fig = ax.get_figure()
    fig.canvas.mpl_connect('scroll_event', zoom)

    return zoom



def plot(y, x=None, ptyp='plot', xlbl=None, ylbl=None, title=None,
         lgnd=None, lglc=None, **kwargs):
    """
    Plot points or lines in 2D. If a figure object is specified then the
    plot is drawn in that figure, and ``fig.show()`` is not called. The
    figure is closed on key entry 'q'.

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
    **kwargs :  :class:`matplotlib.lines.Line2D` properties or figure \
    properties, optional
        Keyword arguments specifying :class:`matplotlib.lines.Line2D`
        properties, e.g. ``lw=2.0`` sets a line width of 2, or properties
        of the figure and axes. If not specified, the defaults for line
        width (``lw``) and marker size (``ms``) are 1.5 and 6.0
        respectively. The valid figure and axes keyword arguments are
        listed below:

        .. |mplfg| replace:: :class:`matplotlib.figure.Figure` object
        .. |mplax| replace:: :class:`matplotlib.axes.Axes` object

        .. rst-class:: kwargs

        =====  ==================== ======================================
        kwarg  Accepts              Description
        =====  ==================== ======================================
        fgsz   tuple (width,height) Specify figure dimensions in inches
        fgnm   integer              Figure number of figure
        fig    |mplfg|              Draw in specified figure instead of
                                    creating one
        ax     |mplax|              Plot in specified axes instead of
                                    current axes of figure
        =====  ==================== ======================================


    Returns
    -------
    fig : :class:`matplotlib.figure.Figure` object
      Figure object for this figure
    ax : :class:`matplotlib.axes.Axes` object
      Axes object for this plot
    """

    # Extract kwargs entries that are not related to line properties
    fgsz = kwargs.pop('fgsz', None)
    fgnm = kwargs.pop('fgnm', None)
    fig = kwargs.pop('fig', None)
    ax = kwargs.pop('ax', None)

    figp = fig
    if fig is None:
        fig = plt.figure(num=fgnm, figsize=fgsz)
        fig.clf()
        ax = fig.gca()
    elif ax is None:
        ax = fig.gca()

    # Set defaults for line width and marker size
    if 'lw' not in kwargs and 'linewidth' not in kwargs:
        kwargs['lw'] = 1.5
    if 'ms' not in kwargs and 'markersize' not in kwargs:
        kwargs['ms'] = 6.0

    if ptyp not in ('plot', 'semilogx', 'semilogy', 'loglog'):
        raise ValueError("Invalid plot type '%s'" % ptyp)
    pltmth = getattr(ax, ptyp)
    if x is None:
        pltln = pltmth(y, **kwargs)
    else:
        pltln = pltmth(x, y, **kwargs)

    ax.fmt_xdata = lambda x: "{: .2f}".format(x)
    ax.fmt_ydata = lambda x: "{: .2f}".format(x)

    if title is not None:
        ax.set_title(title)
    if xlbl is not None:
        ax.set_xlabel(xlbl)
    if ylbl is not None:
        ax.set_ylabel(ylbl)
    if lgnd is not None:
        ax.legend(lgnd, loc=lglc)

    attach_keypress(fig)
    attach_zoom(ax)

    if have_mpldc:
        mpldc.datacursor(pltln)

    if figp is None:
        fig.show()

    return fig, ax



def surf(z, x=None, y=None, elev=None, azim=None, xlbl=None, ylbl=None,
         zlbl=None, title=None, lblpad=8.0, alpha=1.0, cntr=None,
         cmap=None, fgsz=None, fgnm=None, fig=None, ax=None):
    """
    Plot a 2D surface in 3D. If a figure object is specified then the
    surface is drawn in that figure, and ``fig.show()`` is not called.
    The figure is closed on key entry 'q'.

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
    alpha : float between 0.0 and 1.0, optional (default 1.0)
        Transparency
    cntr : int or sequence of ints, optional (default None)
        If not None, plot contours of the surface on the lower end of
        the z-axis. An int specifies the number of contours to plot, and
        a sequence specifies the specific contour levels to plot.
    cmap : :class:`matplotlib.colors.Colormap` object, optional (default None)
        Colour map for surface. If none specifed, defaults to cm.YlOrRd
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
                ax = fig.add_subplot(ax.get_subplotspec(), projection='3d')

    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)

    if cmap is None:
        cmap = cm.YlOrRd

    if x is None:
        x = range(z.shape[1])
    if y is None:
        y = range(z.shape[0])

    xg, yg = np.meshgrid(x, y)
    ax.plot_surface(xg, yg, z, rstride=1, cstride=1, alpha=alpha, cmap=cmap)

    if cntr is not None:
        offset = np.around(z.min() - 0.2 * (z.max() - z.min()), 3)
        ax.contour(xg, yg, z, cntr, cmap=cmap, linewidths=2,
                   linestyles="solid", offset=offset)
        ax.set_zlim(offset, ax.get_zlim()[1])

    ax.fmt_xdata = lambda x: "{: .2f}".format(x)
    ax.fmt_ydata = lambda x: "{: .2f}".format(x)
    ax.fmt_zdata = lambda x: "{: .2f}".format(x)

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



def contour(z, x=None, y=None, v=5, xlog=False, ylog=False, xlbl=None,
            ylbl=None, title=None, cfmt=None, cfntsz=10, lfntsz=None,
            alpha=1.0, cmap=None, vmin=None, vmax=None, fgsz=None, fgnm=None,
            fig=None, ax=None):
    """
    Contour plot of a 2D surface. If a figure object is specified then the
    plot is drawn in that figure, and ``fig.show()`` is not called. The
    figure is closed on key entry 'q'.

    Parameters
    ----------
    z : array_like
        2d array of data to plot
    x : array_like, optional (default None)
        Values for x-axis of the plot
    y : array_like, optional (default None)
        Values for y-axis of the plot
    v : int or sequence of floats, optional (default 5)
        An int specifies the number of contours to plot, and a sequence
        specifies the specific contour levels to plot.
    xlog : boolean, optional (default False)
        Set x-axis to log scale
    ylog : boolean, optional (default False)
        Set y-axis to log scale
    xlbl : string, optional (default None)
        Label for x-axis
    ylbl : string, optional (default None)
        Label for y-axis
    title : string, optional (default None)
        Figure title
    cfmt : string, optional (default None)
        Format string for contour labels.
    cfntsz : int or None, optional (default 10)
        Contour label font size. No contour labels are displayed if
        set to 0 or None.
    lfntsz : int, optional (default None)
        Axis label font size. The default font size is used if set to None.
    alpha : float, optional (default 1.0)
        Underlying image display alpha value
    cmap : :class:`matplotlib.colors.Colormap`, optional (default None)
        Colour map for surface. If none specifed, defaults to cm.YlOrRd
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

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    if cmap is None:
        cmap = cm.YlOrRd

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
    kwargs = {}
    if cfntsz is not None and cfntsz > 0:
        kwargs['fontsize'] = cfntsz
    if cfmt is not None:
        kwargs['fmt'] = cfmt
    if kwargs:
        plt.clabel(cntr, inline=True, **kwargs)
    pc = ax.pcolormesh(xg, yg, z, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
                       shading='gouraud', clim=(vmin, vmax))

    if xlog:
        ax.fmt_xdata = lambda x: "{: .2e}".format(x)
    else:
        ax.fmt_xdata = lambda x: "{: .2f}".format(x)
    if ylog:
        ax.fmt_ydata = lambda x: "{: .2e}".format(x)
    else:
        ax.fmt_ydata = lambda x: "{: .2f}".format(x)

    if title is not None:
        ax.set_title(title)
    if xlbl is not None:
        ax.set_xlabel(xlbl, fontsize=lfntsz)
    if ylbl is not None:
        ax.set_ylabel(ylbl, fontsize=lfntsz)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(pc, ax=ax, cax=cax)

    attach_keypress(fig)
    attach_zoom(ax)

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
    drawn in that figure, and ``fig.show()`` is not called. The figure is
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

    # Deal with removal of 'box-forced' adjustable in Matplotlib 2.2.0
    mplv = matplotlib.__version__.split('.')
    if int(mplv[0]) > 2 or (int(mplv[0]) == 2 and int(mplv[1]) >= 2):
        try:
            ax.set_adjustable('box')
        except Exception:
            ax.set_adjustable('datalim')
    else:
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
        imgd = np.float32(imgd) / np.iinfo(np.uint16).max
    elif img.dtype == np.int16:
        imgd = np.float32(imgd) - imgd.min()
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
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < nc and row >= 0 and row < nr:
            z = img[row, col]
            if imgd.ndim == 2:
                return 'x=%6.2f, y=%6.2f, z=%.2f' % (x, y, z)
            else:
                return 'x=%6.2f, y=%6.2f, z=(%.2f,%.2f,%.2f)' % \
                    sum(((x,), (y,), tuple(z)), ())
        else:
            return 'x=%.2f, y=%.2f' % (x, y)

    ax.format_coord = format_coord

    if fig.canvas.toolbar is not None:
        # See https://stackoverflow.com/a/47086132
        def mouse_move(self, event):
            if event.inaxes and event.inaxes.get_navigate():
                s = event.inaxes.format_coord(event.xdata, event.ydata)
                self.set_message(s)

        def mouse_move_patch(arg):
            return mouse_move(fig.canvas.toolbar, arg)

        fig.canvas.toolbar._idDrag = fig.canvas.mpl_connect(
            'motion_notify_event', mouse_move_patch)


    attach_keypress(fig)
    attach_zoom(ax)

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
