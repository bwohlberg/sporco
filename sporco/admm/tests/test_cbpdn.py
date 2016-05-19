from __future__ import division
from builtins import object

import pytest
import numpy as np

from sporco.admm import cbpdn
import sporco.linalg as sl



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 64
        M = 4
        Nd = 8
        D = np.random.randn(Nd, Nd, M)
        X0 = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X0[xp] = np.random.randn(X0[xp].size)
        S = np.sum(sl.ifftn(sl.fftn(D, (N, N), (0,1)) *
                   sl.fftn(X0, None, (0,1)), None, (0,1)).real, axis=2)
        lmbda = 1e-4
        rho = 1e-1
        opt = cbpdn.ConvBPDN.Options({'Verbose' : False, 'MaxMainIter' : 500,
                                      'RelStopTol' : 1e-3, 'rho' : rho,
                                      'AutoRho' : {'Enabled' : False}})
        b = cbpdn.ConvBPDN(D, S, lmbda, opt)
        b.solve()
        X1 = b.Y.squeeze()
        assert(sl.rrs(X0,X1) < 5e-5)
        Sr = b.reconstruct().squeeze()
        assert(sl.rrs(S,Sr) < 1e-4)


    def test_02(self):
        N = 63
        M = 4
        Nd = 8
        D = np.random.randn(Nd, Nd, M)
        X0 = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X0[xp] = np.random.randn(X0[xp].size)
        S = np.sum(sl.ifftn(sl.fftn(D, (N, N), (0,1)) *
                   sl.fftn(X0, None, (0,1)), None, (0,1)).real, axis=2)
        lmbda = 1e-4
        rho = 1e-1
        opt = cbpdn.ConvBPDN.Options({'Verbose' : False, 'MaxMainIter' : 500,
                                      'RelStopTol' : 1e-3, 'rho' : rho,
                                      'AutoRho' : {'Enabled' : False}})
        b = cbpdn.ConvBPDN(D, S, lmbda, opt)
        b.solve()
        X1 = b.Y.squeeze()
        assert(sl.rrs(X0,X1) < 5e-5) 
        Sr = b.reconstruct().squeeze()
        assert(sl.rrs(S,Sr) < 1e-4)


    def test_03(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        lmbda = 1e-1
        try:
            b = cbpdn.ConvBPDN(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_04(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        try:
            b = cbpdn.ConvBPDN(D, s)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)
