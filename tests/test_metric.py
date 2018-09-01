from __future__ import division
from builtins import object

import numpy as np
from sporco import metric



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 16
        x = np.random.randn(N)
        y = x.copy()
        y[0] = 0
        xe = np.abs(x[0])
        e1 = metric.mae(x, y)
        e2 = metric.mse(x, y)
        assert np.abs(e1 - xe / N) < 1e-12
        assert np.abs(e2 - (xe**2) / N) < 1e-12


    def test_02(self):
        N = 16
        x = np.random.randn(N)
        x /= np.sqrt(np.var(x))
        y = x + 1
        assert np.abs(metric.snr(x, y)) < 1e-8


    def test_03(self):
        N = 16
        x = np.random.randn(N)
        x -= x.min()
        x /= x.max()
        y = x + 1
        assert np.abs(metric.psnr(x, y)) < 1e-8


    def test_04(self):
        N = 16
        x = np.random.randn(N)
        y = x + 1
        assert np.abs(metric.psnr(x, y, rng=1.0)) < 1e-8


    def test_05(self):
        N = 16
        x = np.random.randn(N)
        y = np.random.randn(N)
        assert np.abs(metric.isnr(x, y, y)) < 1e-8


    def test_06(self):
        N = 16
        x = np.random.randn(N)
        x /= np.sqrt(np.var(x))
        n = np.random.randn(N)
        n /= np.sqrt(np.var(n))
        y = x + n
        assert np.abs(metric.bsnr(x, y)) < 1e-8


    def test_07(self):
        N = 16
        x = np.random.randn(N, N)
        y = x + 1
        assert metric.pamse(x, y) > 0.0
        assert np.abs(metric.gmsd(x, y)) < 1e-8
