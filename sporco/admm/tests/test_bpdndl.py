from __future__ import division
from builtins import object

import pytest
import numpy as np

from sporco.admm import bpdndl


class TestSet01(object):

    def setup_method(self, method):
        pass


    def test_01(self):
        N = 8
        M = 4
        K = 8
        D0 = np.random.randn(N, M)
        S = np.random.randn(N, K)
        lmbda = 1e-1
        try:
            b = bpdndl.BPDNDictLearn(D0, S, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_02(self):
        N = 8
        M = 4
        K = 8
        D0 = np.random.randn(N, M)
        S = np.random.randn(N, K)
        try:
            b = bpdndl.BPDNDictLearn(D0, S)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)
