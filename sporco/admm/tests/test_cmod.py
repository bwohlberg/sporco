from __future__ import division
from builtins import object

import pytest
import numpy as np

from sporco.admm import cmod
import sporco.linalg as sl



class TestSet01(object):

    def setup_method(self, method):
        pass


    def test_01(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        try:
            b = cmod.CnstrMOD(X, S, (N, M))
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_02(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        try:
            b = cmod.CnstrMOD(X, S)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)
