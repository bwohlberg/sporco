from __future__ import division
from builtins import object

import numpy as np

from sporco import interp


class TestSet01(object):

    def test_01(self):
        x = np.arange(0, 11).astype(np.float32)
        m0 = 2.0
        c0 = 1.0
        y0 = m0 * x + c0
        y = y0.copy()
        y[4] += 2.0
        A = np.vstack([x, np.ones(x.shape[0:])]).T
        m1, c1 = interp.lstabsdev(A, y)
        assert np.abs(m0 - m1) < 1e-5
        assert np.abs(c0 - c1) < 1e-5


    def test_02(self):
        x = np.arange(0, 11).astype(np.float32)
        m0 = -1.0
        c0 = 2.0
        y0 = m0 * x + c0
        y = y0.copy()
        y[4] += 2.0
        A = np.vstack([x, np.ones(x.shape[0:])]).T
        m1, c1 = interp.lstabsdev(A, y)
        assert np.abs(m0 - m1) < 1e-5
        assert np.abs(c0 - c1) < 1e-5


    def test_03(self):
        x = np.arange(0, 11).astype(np.float32)
        m0 = 2.0
        c0 = 1.0
        y0 = m0 * x + c0
        y = y0.copy()
        y[0::2] += 2.0
        y[1::2] -= 2.0
        A = np.vstack([x, np.ones(x.shape[0:])]).T
        m1, c1 = interp.lstmaxdev(A, y)
        assert np.abs(m0 - m1) < 1e-5
        assert np.abs(c0 - c1) < 1e-5


    def test_04(self):
        x = np.arange(0, 11).astype(np.float32)
        m0 = -1.0
        c0 = 2.0
        y0 = m0 * x + c0
        y = y0.copy()
        y[0::2] += 2.0
        y[1::2] -= 2.0
        A = np.vstack([x, np.ones(x.shape[0:])]).T
        m1, c1 = interp.lstmaxdev(A, y)
        assert np.abs(m0 - m1) < 1e-5
        assert np.abs(c0 - c1) < 1e-5


    def test_05(self):
        y = interp.lanczos_kernel(np.array([-0.5, 0.0, 0.5, 1.0]), a=2)
        assert np.abs(y[0] - y[2]) < 1e-16
        assert np.abs(y[1] - 1.0) < 1e-16
        assert np.abs(y[3]) < 1e-16


    def test_06(self):
        y = interp.lanczos_filters(4, a=2)
        assert y.shape == (5, 4)


    def test_07(self):
        y = interp.lanczos_filters((3, 4))
        assert y.shape == (7, 7, 12)
        y = interp.lanczos_filters((3, 4), collapse_axes=False)
        assert y.shape == (7, 7, 3, 4)
        y = interp.lanczos_filters((2, (0.1, 0.2, 0.3)))
        assert y.shape == (7, 7, 6)


    def test_08(self):
        x = np.random.randn(9, 8)
        y = interp.bilinear_demosaic(x)
        assert np.allclose(x[1::2, 1::2], y[1::2, 1::2, 0])
        assert np.array_equal(x[0::2, 1::2], y[0::2, 1::2, 1])
        assert np.array_equal(x[1::2, 0::2], y[1::2, 0::2, 1])
        assert np.allclose(x[0::2, 0::2], y[0::2, 0::2, 2])
