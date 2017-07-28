from __future__ import division
from builtins import object

import pytest
import matplotlib
matplotlib.use('Agg')

import numpy as np
from sporco import plot


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
        plot.plot(y, x=x, title='Plot Test', xlbl='x', ylbl='y', fgrf=fig)
        plot.close()


    def test_03(self):
        fig = plot.figure()
        plot.surf(self.z, title='Surf Test', xlbl='x', ylbl='y', zlbl='z')
        plot.close()


    def test_04(self):
        fig = plot.figure()
        plot.surf(self.z, x=self.x, y=self.y, title='Surf Test', xlbl='x',
                  ylbl='y', zlbl='z', fgrf=fig)
        plot.close()


    def test_05(self):
        plot.imview(self.z.astype(np.float16), title='Imview Test', cbar=True)
        plot.close()


    def test_06(self):
        fig = plot.figure()
        plot.imview(self.z, title='Imview Test', fltscl=True, fgrf=fig)
        plot.close()


    def test_07(self):
        fg, ax = plot.imview(self.z, title='Imview Test', fltscl=True)
        plot.imview(self.z, title='Imview Test', fltscl=True, axes=ax)


    def test_08(self):
        fig = plot.figure()
        plot.imview((100.0*self.z).astype(np.int16), title='Imview Test',
                    fltscl=True, fgrf=fig)
        plot.close()


    def test_09(self):
        fig = plot.figure()
        plot.imview((100.0*self.z).astype(np.uint16), title='Imview Test',
                    fltscl=True, fgrf=fig)
        plot.close()


    def test_10(self):
        z3 = np.dstack((self.z, 2*self.z, 3*self.z))
        fig = plot.figure()
        plot.imview(z3, title='Imview Test', fgrf=fig)
        plot.close()
