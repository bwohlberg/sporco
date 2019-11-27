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


    def test_03(self):
        x = np.random.randn(9, 8)
        y = interp.bilinear_demosaic(x)
        assert np.array_equal(x[1::2, 1::2], y[1::2, 1::2, 0])
        assert np.array_equal(x[0::2, 1::2], y[0::2, 1::2, 1])
        assert np.array_equal(x[1::2, 0::2], y[1::2, 0::2, 1])
        assert np.array_equal(x[0::2, 0::2], y[0::2, 0::2, 2])
