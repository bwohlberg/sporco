from __future__ import division
from builtins import object

import pytest
import matplotlib
matplotlib.use('Agg')

import numpy as np
from sporco import plot
from sporco import util


# Monkey patch in_ipython and in_notebook functions to allow testing of
# functions that depend on these tests
def in_ipython():
    return True
util.in_ipython = in_ipython
def in_notebook():
    return True
util.in_notebook = in_notebook

# Dummy get_ipython function to allow testing of code segments that
# are only intended to be run within ipython or a notebook
def get_ipython():
    class ipython_dummy(object):
        def run_line_magic(*args):
            pass
    return ipython_dummy()
plot.get_ipython = get_ipython


@pytest.mark.filterwarnings('ignore:matplotlib is currently using')
class TestSet01(object):

    def setup_method(self, method):
        self.x = np.linspace(-1, 1, 20)[np.newaxis, :]
        self.y = np.linspace(-1, 1, 20)[:, np.newaxis]
        self.z = np.sqrt(self.x**2 + self.y**2)


    def test_01(self):
        x = np.linspace(-1, 1, 20)
        y = x**2
        plot.plot(y, title='Plot Test', xlbl='x', ylbl='y', lgnd=('Legend'))
        plot.close()


    def test_02(self):
        x = np.linspace(-1, 1, 20)
        y = x**2
        fig = plot.figure()
        plot.plot(y, x=x, title='Plot Test', xlbl='x', ylbl='y', fig=fig)
        plot.close()


    def test_03(self):
        fig, ax = plot.subplots(nrows=1, ncols=1)
        plot.surf(self.z, title='Surf Test', xlbl='x', ylbl='y', zlbl='z',
                  elev=0.0, fig=fig, ax=ax)
        plot.close()


    def test_04(self):
        fig = plot.figure()
        plot.surf(self.z, x=self.x, y=self.y, title='Surf Test', xlbl='x',
                  ylbl='y', zlbl='z', cntr=5, fig=fig)
        plot.close()


    def test_05(self):
        plot.contour(self.z, x=self.x, y=self.y, title='Contour Test',
                     xlbl='x', ylbl='y')
        plot.close()


    def test_06(self):
        fig = plot.figure()
        plot.contour(self.z, title='Contour Test', xlbl='x', ylbl='y',
                     fig=fig)
        plot.close()


    def test_07(self):
        plot.imview(self.z.astype(np.float16), title='Imview Test', cbar=True)
        plot.close()


    def test_08(self):
        fig = plot.figure()
        plot.imview(self.z, title='Imview Test', fltscl=True, fig=fig)
        plot.close()


    def test_09(self):
        fg, ax = plot.imview(self.z, title='Imview Test', fltscl=True,
                             cbar=None)
        ax.format_coord(0, 0)
        plot.close(fg)


    def test_10(self):
        fig = plot.figure()
        plot.imview((100.0*self.z).astype(np.int16), title='Imview Test',
                    fltscl=True, fig=fig)
        plot.close()


    def test_11(self):
        fig = plot.figure()
        plot.imview((100.0*self.z).astype(np.uint16), title='Imview Test',
                    fltscl=True, fig=fig)
        plot.close()


    def test_12(self):
        z3 = np.dstack((self.z, 2*self.z, 3*self.z))
        fig = plot.figure()
        plot.imview(z3, title='Imview Test', fig=fig)
        plot.close()


    def test_13(self):
        plot.set_ipython_plot_backend()


    def test_14(self):
        plot.set_notebook_plot_backend()


    def test_15(self):
        plot.config_notebook_plotting()
        assert plot.plot.__name__ == 'plot_wrap'
