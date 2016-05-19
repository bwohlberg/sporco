from __future__ import division
from builtins import range
from builtins import object

import pytest
import numpy as np
from scipy import linalg

from sporco.admm import bpdn
import sporco.linalg as sl


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        rho = 1e-1
        N = 64
        M = 128
        K = 32
        D = np.random.randn(N, M)
        X = np.random.randn(M, K)
        S = D.dot(X)
        Z = (D.T.dot(D).dot(X) + rho*X - D.T.dot(S)) / rho
        lu, piv = bpdn.factorise(D, rho)
        Xslv = bpdn.linsolve(D, rho, lu, piv, D.T.dot(S) + rho*Z)
        assert(sl.rrs(D.T.dot(D).dot(Xslv) + rho*Xslv,
                        D.T.dot(S) + rho*Z) < 1e-11)


    def test_02(self):
        rho = 1e-1
        N = 128
        M = 64
        K = 32
        D = np.random.randn(N, M)
        X = np.random.randn(M, K)
        S = D.dot(X)
        Z = (D.T.dot(D).dot(X) + rho*X - D.T.dot(S)) / rho
        lu, piv = bpdn.factorise(D, rho)
        Xslv = bpdn.linsolve(D, rho, lu, piv, D.T.dot(S) + rho*Z)
        assert(sl.rrs(D.T.dot(D).dot(Xslv) + rho*Xslv,
                        D.T.dot(S) + rho*Z) < 1e-14)


    def test_03(self):
        N = 64
        M = 2*N
        L = 4
        np.random.seed(12345)
        D = np.random.randn(N, M)
        x0 = np.zeros((M, 1))
        si = np.random.permutation(list(range(0, M-1)))
        x0[si[0:L]] = np.random.randn(L, 1)
        s0 = D.dot(x0)
        lmbda = 5e-3
        opt = bpdn.BPDN.Options({'Verbose' : False, 'MaxMainIter' : 500,
                    'RelStopTol' : 1e-3})
        b = bpdn.BPDN(D, s0, lmbda, opt)
        b.solve()
        x1 = b.Y
        assert(np.abs(b.itstat[-1].ObjFun - 1.2016e-2) < 1e-5)
        assert(np.abs(b.itstat[-1].DFid - 1.0025e-5) < 1e-5)
        assert(np.abs(b.itstat[-1].RegL1 - 2.40116) < 1e-5)
        assert(linalg.norm(x1-x0) < 1e-3)


    def test_04(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        lmbda = 1e-1
        try:
            b = bpdn.BPDN(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_05(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        try:
            b = bpdn.BPDN(D, s)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)
