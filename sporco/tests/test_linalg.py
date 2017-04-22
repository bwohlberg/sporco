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
        M = 128
        K = 32
        D = np.random.randn(N, M)
        X = np.random.randn(M, K)
        S = D.dot(X)
        Z = (D.T.dot(D).dot(X) + rho*X - D.T.dot(S)) / rho
        lu, piv = linalg.lu_factor(D, rho)
        Xslv = linalg.lu_solve_ATAI(D, rho, D.T.dot(S) + rho*Z, lu, piv)
        assert(linalg.rrs(D.T.dot(D).dot(Xslv) + rho*Xslv,
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
        lu, piv = linalg.lu_factor(D, rho)
        Xslv = linalg.lu_solve_ATAI(D, rho, D.T.dot(S) + rho*Z, lu, piv)
        assert(linalg.rrs(D.T.dot(D).dot(Xslv) + rho*Xslv,
                        D.T.dot(S) + rho*Z) < 1e-14)



    def test_03(self):
        rho = 1e-1
        N = 64
        M = 128
        K = 32
        D = np.random.randn(N, M)
        X = np.random.randn(M, K)
        S = D.dot(X)
        Z = (D.dot(X).dot(X.T) + rho*D - S.dot(X.T)) / rho
        lu, piv = linalg.lu_factor(X, rho)
        Dslv = linalg.lu_solve_AATI(X, rho, S.dot(X.T) + rho*Z, lu, piv)
        assert(linalg.rrs(Dslv.dot(X).dot(X.T) + rho*Dslv,
                        S.dot(X.T) + rho*Z) < 1e-11)



    def test_04(self):
        rho = 1e-1
        N = 128
        M = 64
        K = 32
        D = np.random.randn(N, M)
        X = np.random.randn(M, K)
        S = D.dot(X)
        Z = (D.dot(X).dot(X.T) + rho*D - S.dot(X.T)) / rho
        lu, piv = linalg.lu_factor(X, rho)
        Dslv = linalg.lu_solve_AATI(X, rho, S.dot(X.T) + rho*Z, lu, piv)
        assert(linalg.rrs(Dslv.dot(X).dot(X.T) + rho*Dslv,
                        S.dot(X.T) + rho*Z) < 1e-11)



    def test_05(self):
        rho = 1e-1
        N = 32
        M = 16
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



    def test_06(self):
        N = 32
        M = 16
        K = 8
        D = np.random.randn(N, N, 1, 1, M).astype('complex') + \
            np.random.randn(N, N, 1, 1, M).astype('complex') * 1.0j
        X = np.random.randn(N, N, 1, K, M).astype('complex') + \
            np.random.randn(N, N, 1, K, M).astype('complex') * 1.0j
        S = np.sum(D*X, axis=4, keepdims=True)
        d = 1e-1 * (np.random.randn(N, N, 1, 1, M).astype('complex') + \
            np.random.randn(N, N, 1, 1, M).astype('complex') * 1.0j)
        Z = (D.conj()*np.sum(D*X, axis=4, keepdims=True) + \
             d*X - D.conj()*S) / d
        Xslv = linalg.solvedbd_sm(D, d, D.conj()*S + d*Z)
        assert(linalg.rrs(D.conj()*np.sum(D*Xslv, axis=4, keepdims=True) +
                        d*Xslv, D.conj()*S + d*Z) < 1e-11)



    def test_07(self):
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
        Dslv = linalg.solvemdbi_ism(X, rho,  XHop(S) + rho*Z, 4, 3)

        assert(linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11)



    def test_08(self):
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



    def test_09(self):
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



    def test_10(self):
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



    def test_11(self):
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



    def test_12(self):
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



    def test_13(self):
        b = np.array([0.0,0.0,2.0])
        s = np.array([0.0,0.0,0.0])
        r = 1.0
        p = linalg.proj_l2ball(b, s, r)
        assert(linalg.rrs(p, np.array([0.0,0.0,1.0])) < 1e-14)



    def test_14(self):
        u = np.array([[0,1],[2,3]])
        v = linalg.roll(u, [1, 1])
        assert(v[0,0] == 3)



    def test_15(self):
        u0 = np.array([[0,1],[2,3]])
        u1 = np.array([[4,5],[6,7]])
        C = linalg.blockcirculant((u0,u1))
        assert(C[3,0] == 6)
        assert(C[3,3] == 3)



    def test_16(self):
        x = np.random.randn(16,8)
        xf = linalg.fftn(x, axes=(0,))
        n1 = np.linalg.norm(x)**2
        n2 = linalg.fl2norm2(xf, axis=(0,))
        assert(np.abs(n1-n2) < 1e-12)



    def test_17(self):
        x = np.random.randn(16,8)
        xf = linalg.rfftn(x, axes=(0,))
        n1 = np.linalg.norm(x)**2
        n2 = linalg.rfl2norm2(xf, xs=x.shape, axis=(0,))
        assert(np.abs(n1-n2) < 1e-12)


    def test_18(self):
        x = np.random.randn(16,8)
        y = np.random.randn(16,8)
        ip1 = np.sum(x * y, axis=0, keepdims=True)
        ip2 = linalg.inner(x, y, axis=0)
        assert(np.linalg.norm(ip1 - ip2) < 1e-13)


    def test_19(self):
        x = np.random.randn(8,8,3,12)
        y = np.random.randn(8,1,1,12)
        ip1 = np.sum(x * y, axis=-1, keepdims=True)
        ip2 = linalg.inner(x, y, axis=-1)
        assert(np.linalg.norm(ip1 - ip2) < 1e-13)
