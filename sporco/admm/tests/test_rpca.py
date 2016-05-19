from builtins import object

import pytest
import numpy as np

from sporco.admm import rpca
import sporco.linalg as sl


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 64
        K = 5
        L = 10
        u = np.random.randn(N, K)
        U = np.dot(u, u.T)
        V = np.random.randn(N, N)
        t = np.sort(np.abs(V).ravel())[V.size-L]
        V[np.abs(V) < t] = 0
        D = U + V
        opt = rpca.RobustPCA.Options({'Verbose' : False, 'gEvalY' : False,
                              'MaxMainIter' : 250,
                              'AutoRho' : {'Enabled' : True}})
        b = rpca.RobustPCA(D, None, opt)
        X, Y = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 321.6189484339) < 1e-6)
        assert(sl.mse(U,X) < 2e-6)
        assert(sl.mse(V,Y) < 1e-8)


    def test_02(self):
        N = 8
        D = np.random.randn(N, N)
        try:
            b = rpca.RobustPCA(D)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)
