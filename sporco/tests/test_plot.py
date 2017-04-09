from __future__ import division
from builtins import object

import pytest
import matplotlib
matplotlib.use('Agg')

import numpy as np
from sporco import plot


class TestSet01(object):

    def test_01(self):
        x = np.linspace(-1,1,20)
        y = x**2
        plot.plot(y, title='Plot Test', xlbl='x', ylbl='y', lgnd=('Legend'))
        plot.close()


    def test_02(self):
        x = np.linspace(-1,1,20)
        y = x**2
        fig = plot.figure()
        plot.plot(y, x=x, title='Plot Test', xlbl='x', ylbl='y', fgrf=fig)
        plot.close()


    def test_03(self):
        x = np.linspace(-1,1,20)[np.newaxis,:]
        y = np.linspace(-1,1,20)[:,np.newaxis]
        z = np.sqrt(x**2 + y**2)
        fig = plot.figure()
        plot.surf(z, title='Surf Test', xlbl='x', ylbl='y', zlbl='z')
        plot.close()


    def test_04(self):
        x = np.linspace(-1,1,20)[np.newaxis,:]
        y = np.linspace(-1,1,20)[:,np.newaxis]
        z = np.sqrt(x**2 + y**2)
        fig = plot.figure()
        plot.surf(z, x=x, y=y, title='Surf Test', xlbl='x',
                  ylbl='y', zlbl='z', fgrf=fig)
        plot.close()


    def test_05(self):
        x = np.linspace(-1,1,20)[np.newaxis,:]
        y = np.linspace(-1,1,20)[:,np.newaxis]
        z = np.sqrt(x**2 + y**2)
        plot.imview(np.asarray(z, np.float16), title='Imview Test', cbar=True)
        plot.close()


    def test_06(self):
        x = np.linspace(-1,1,20)[np.newaxis,:]
        y = np.linspace(-1,1,20)[:,np.newaxis]
        z = np.sqrt(x**2 + y**2)
        fig = plot.figure()
        plot.imview(z, title='Imview Test', fltscl=True, fgrf=fig)
        plot.close()


    def test_07(self):
        x = np.linspace(-1,1,20)[np.newaxis,:]
        y = np.linspace(-1,1,20)[:,np.newaxis]
        z = (100.0*np.sqrt(x**2 + y**2)).astype(np.uint16)
        fig = plot.figure()
        plot.imview(z, title='Imview Test', fltscl=True, fgrf=fig)
        plot.close()


    def test_08(self):
        x = np.linspace(-1,1,20)[np.newaxis,:]
        y = np.linspace(-1,1,20)[:,np.newaxis]
        z = np.sqrt(x**2 + y**2)
        z3 = np.dstack((z,2*z,3*z))
        fig = plot.figure()
        plot.imview(z3, title='Imview Test', fgrf=fig)
        plot.close()
