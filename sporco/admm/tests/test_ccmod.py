from builtins import object

import pytest
import numpy as np

from sporco.admm import ccmod
import sporco.linalg as sl



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 64
        M = 4
        Nd = 8
        D0 = ccmod.normalise(np.random.randn(Nd, Nd, M), axisN=(0,1))
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(sl.ifftn(sl.fftn(D0, (N, N), (0,1)) *
                   sl.fftn(X, None, (0,1)), None, (0,1)).real, axis=2)
        rho = 1e1
        opt = ccmod.ConvCnstrMOD.Options({'Verbose' : False,
                                          'MaxMainIter' : 500,
                                          'RelStopTol' : 1e-3, 'rho' : rho,
                                          'AutoRho' : {'Enabled' : False}})
        Xr = X.reshape(X.shape[0:2] + (1,1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = ccmod.bcrop(c.Y, D0.shape).squeeze()
        assert(sl.rrs(D0,D1) < 1e-5) 


    def test_02(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        try:
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, M))
            c.solve()
        except Exception as e:
            print(e)
            assert(0)
