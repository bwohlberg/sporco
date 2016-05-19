from __future__ import division
from builtins import object

import pytest
import numpy as np

from sporco.admm import cbpdndl



class TestSet01(object):

    def setup_method(self, method):
        pass


    def test_01(self):
        N = 16
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, M)
        S = np.random.randn(N, N, K)
        lmbda = 1e-1
        try:
            b = cbpdndl.ConvBPDNDictLearn(D0, S, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_02(self):
        N = 16
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, M)
        S = np.random.randn(N, N, K)
        try:
            b = cbpdndl.ConvBPDNDictLearn(D0, S)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)
