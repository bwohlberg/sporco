from __future__ import division
from builtins import object

import numpy as np
from scipy.ndimage import convolve
from sporco import fft


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        x = np.random.randn(16, 8)
        xf = fft.fftn(x, axes=(0,))
        n1 = np.linalg.norm(x)**2
        n2 = fft.fl2norm2(xf, axis=(0,))
        assert np.abs(n1 - n2) < 1e-12


    def test_02(self):
        x = np.random.randn(16, )
        xf = fft.rfftn(x, axes=(0,))
        n1 = np.linalg.norm(x)**2
        n2 = fft.rfl2norm2(xf, xs=x.shape, axis=(0,))
        assert np.abs(n1 - n2) < 1e-12


    def test_03(self):
        x = np.array([[0, 1], [2, 3]])
        y = np.array([[4, 5], [6, 7]])
        xy = np.array([[38, 36], [30, 28]])
        assert np.allclose(fft.fftconv(x, y, axes=(0, 1)), xy)


    def test_04(self):
        x = np.random.randn(5,)
        y = np.zeros((12,))
        y[4] = 1.0
        xy0 = convolve(y, x)
        xy1 = fft.fftconv(x, y, axes=(0,), origin=(2,))
        assert np.allclose(xy0, xy1)
