from __future__ import division
from builtins import object

import pytest

import numpy as np
from sporco import linalg



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)



    def test_01(self):
        rho = 1e-1
        N = 64
        M = 32
        K = 8
        D = np.random.randn(N, N, 1, 1, M).astype('complex') + \
            np.random.randn(N, N, 1, 1, M).astype('complex') * 1.0j
        X = np.random.randn(N, N, 1, K, M).astype('complex') + \
            np.random.randn(N, N, 1, K, M).astype('complex') * 1.0j
        S = np.sum(D*X, axis=4, keepdims=True)
        Z = (D.conj()*np.sum(D*X, axis=4, keepdims=True) + \
             rho*X - D.conj()*S) / rho
        Xslv = linalg.solvedbi_sm(D, rho, D.conj()*S + rho*Z)
        assert(linalg.rrs(D.conj()*np.sum(D*Xslv, axis=4, keepdims=True) +
                        rho*Xslv, D.conj()*S + rho*Z) < 1e-11)



    def test_02(self):
        rho = 1e-1
        N = 64
        M = 32
        K = 8
        D = np.random.randn(N, N, 1, 1, M).astype('complex') + \
            np.random.randn(N, N, 1, 1, M).astype('complex') * 1.0j
        X = np.random.randn(N, N, 1, K, M).astype('complex') + \
            np.random.randn(N, N, 1, K, M).astype('complex') * 1.0j
        S = np.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: np.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: np.sum(np.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv = linalg.solvemdbi_ism(X, rho,  XHop(S) + rho*Z, 4, 3)

        assert(linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11)



    def test_03(self):
        rho = 1e-1
        N = 32
        M = 16
        C = 3
        K = 8
        D = np.random.randn(N, N, C, 1, M).astype('complex') + \
            np.random.randn(N, N, C, 1, M).astype('complex') * 1.0j
        X = np.random.randn(N, N, 1, K, M).astype('complex') + \
            np.random.randn(N, N, 1, K, M).astype('complex') * 1.0j
        S = np.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: np.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: np.sum(np.conj(X)* x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv = linalg.solvemdbi_ism(X, rho,  XHop(S) + rho*Z, 4, 3)

        assert(linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11)



    def test_04(self):
        rho = 1e-1
        N = 32
        M = 16
        K = 8
        D = np.random.randn(N, N, 1, 1, M).astype('complex') + \
            np.random.randn(N, N, 1, 1, M).astype('complex') * 1.0j
        X = np.random.randn(N, N, 1, K, M).astype('complex') + \
            np.random.randn(N, N, 1, K, M).astype('complex') * 1.0j
        S = np.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: np.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: np.sum(np.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv = linalg.solvemdbi_rsm(X, rho,  XHop(S) + rho*Z, 3)

        assert(linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11)



    def test_05(self):
        rho = 1e-1
        N = 64
        M = 32
        C = 3
        K = 8
        D = np.random.randn(N, N, C, 1, M).astype('complex') + \
            np.random.randn(N, N, C, 1, M).astype('complex') * 1.0j
        X = np.random.randn(N, N, 1, K, M).astype('complex') + \
            np.random.randn(N, N, 1, K, M).astype('complex') * 1.0j
        S = np.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: np.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: np.sum(np.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv = linalg.solvemdbi_rsm(X, rho,  XHop(S) + rho*Z, 3)

        assert(linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11)



    def test_06(self):
        rho = 1e-1
        N = 32
        M = 16
        K = 8
        D = np.random.randn(N, N, 1, 1, M).astype('complex') + \
            np.random.randn(N, N, 1, 1, M).astype('complex') * 1.0j
        X = np.random.randn(N, N, 1, K, M).astype('complex') + \
            np.random.randn(N, N, 1, K, M).astype('complex') * 1.0j
        S = np.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: np.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: np.sum(np.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv, cgit = linalg.solvemdbi_cg(X, rho, XHop(S)+rho*Z, 4, 3, tol=1e-6)

        assert(linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) <= 1e-6)



    def test_07(self):
        rho = 1e-1
        N = 64
        M = 32
        C = 3
        K = 8
        D = np.random.randn(N, N, C, 1, M).astype('complex') + \
            np.random.randn(N, N, C, 1, M).astype('complex') * 1.0j
        X = np.random.randn(N, N, 1, K, M).astype('complex') + \
            np.random.randn(N, N, 1, K, M).astype('complex') * 1.0j
        S = np.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: np.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: np.sum(np.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv, cgit = linalg.solvemdbi_cg(X, rho, XHop(S)+rho*Z, 4, 3, tol=1e-6)

        assert(linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) <= 1e-6)
