from __future__ import division
from builtins import object

import numpy as np
import pytest
import platform

from sporco import linalg
from sporco.signal import complex_randn
from sporco.metric import mse


def kronsum(S, B, C):

    A = np.zeros((B.shape[0] * C.shape[0], B.shape[1] * C.shape[1]))
    for i in range(B.shape[2]):
        A += S[i] * np.kron(B[..., i], C[..., i])
    return A


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
        assert linalg.rrs(D.T.dot(D).dot(Xslv) + rho*Xslv,
                          D.T.dot(S) + rho*Z) < 1e-11


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
        assert linalg.rrs(D.T.dot(D).dot(Xslv) + rho*Xslv,
                          D.T.dot(S) + rho*Z) < 1e-14


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
        assert linalg.rrs(Dslv.dot(X).dot(X.T) + rho*Dslv,
                          S.dot(X.T) + rho*Z) < 1e-11


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
        assert linalg.rrs(Dslv.dot(X).dot(X.T) + rho*Dslv,
                          S.dot(X.T) + rho*Z) < 1e-11


    def test_05(self):
        rho = 1e-1
        N = 64
        M = 128
        K = 32
        D = np.random.randn(N, M)
        X = np.random.randn(M, K)
        S = D.dot(X)
        Z = (D.T.dot(D).dot(X) + rho*X - D.T.dot(S)) / rho
        c, lwr = linalg.cho_factor(D, rho)
        Xslv = linalg.cho_solve_ATAI(D, rho, D.T.dot(S) + rho*Z, c, lwr)
        assert linalg.rrs(D.T.dot(D).dot(Xslv) + rho*Xslv,
                          D.T.dot(S) + rho*Z) < 1e-11


    def test_06(self):
        rho = 1e-1
        N = 128
        M = 64
        K = 32
        D = np.random.randn(N, M)
        X = np.random.randn(M, K)
        S = D.dot(X)
        Z = (D.T.dot(D).dot(X) + rho*X - D.T.dot(S)) / rho
        c, lwr = linalg.cho_factor(D, rho)
        Xslv = linalg.cho_solve_ATAI(D, rho, D.T.dot(S) + rho*Z, c, lwr)
        assert linalg.rrs(D.T.dot(D).dot(Xslv) + rho*Xslv,
                          D.T.dot(S) + rho*Z) < 1e-14


    def test_07(self):
        rho = 1e-1
        N = 64
        M = 128
        K = 32
        D = np.random.randn(N, M)
        X = np.random.randn(M, K)
        S = D.dot(X)
        Z = (D.dot(X).dot(X.T) + rho*D - S.dot(X.T)) / rho
        c, lwr = linalg.cho_factor(X, rho)
        Dslv = linalg.cho_solve_AATI(X, rho, S.dot(X.T) + rho*Z, c, lwr)
        assert linalg.rrs(Dslv.dot(X).dot(X.T) + rho*Dslv,
                          S.dot(X.T) + rho*Z) < 1e-11


    def test_08(self):
        rho = 1e-1
        N = 128
        M = 64
        K = 32
        D = np.random.randn(N, M)
        X = np.random.randn(M, K)
        S = D.dot(X)
        Z = (D.dot(X).dot(X.T) + rho*D - S.dot(X.T)) / rho
        c, lwr = linalg.cho_factor(X, rho)
        Dslv = linalg.cho_solve_AATI(X, rho, S.dot(X.T) + rho*Z, c, lwr)
        assert linalg.rrs(Dslv.dot(X).dot(X.T) + rho*Dslv,
                          S.dot(X.T) + rho*Z) < 1e-11


    def test_09(self):
        rho = 1e-1
        N = 32
        M = 16
        K = 8
        D = complex_randn(N, N, 1, 1, M)
        X = complex_randn(N, N, 1, K, M)
        S = np.sum(D*X, axis=4, keepdims=True)
        Z = (D.conj()*np.sum(D*X, axis=4, keepdims=True) + \
             rho*X - D.conj()*S) / rho
        Xslv = linalg.solvedbi_sm(D, rho, D.conj()*S + rho*Z)
        assert linalg.rrs(D.conj()*np.sum(D*Xslv, axis=4, keepdims=True) +
                          rho*Xslv, D.conj()*S + rho*Z) < 1e-11


    def test_10(self):
        N = 32
        M = 16
        K = 8
        D = complex_randn(N, N, 1, 1, M)
        X = complex_randn(N, N, 1, K, M)
        S = np.sum(D*X, axis=4, keepdims=True)
        d = 1e-1 * (np.random.randn(N, N, 1, 1, M).astype('complex') +
            np.random.randn(N, N, 1, 1, M).astype('complex') * 1.0j)
        Z = (D.conj()*np.sum(D*X, axis=4, keepdims=True) +
             d*X - D.conj()*S) / d
        Xslv = linalg.solvedbd_sm(D, d, D.conj()*S + d*Z)
        assert linalg.rrs(D.conj()*np.sum(D*Xslv, axis=4, keepdims=True) +
                          d*Xslv, D.conj()*S + d*Z) < 1e-11


    def test_11(self):
        rho = 1e-1
        N = 32
        M = 16
        K = 8
        D = complex_randn(N, N, 1, 1, M)
        X = complex_randn(N, N, 1, K, M)
        S = np.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: np.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: np.sum(np.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv = linalg.solvemdbi_ism(X, rho, XHop(S) + rho*Z, 4, 3)
        assert linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11


    def test_12(self):
        rho = 1e-1
        N = 32
        M = 16
        C = 3
        K = 8
        D = complex_randn(N, N, C, 1, M)
        X = complex_randn(N, N, 1, K, M)
        S = np.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: np.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: np.sum(np.conj(X)* x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv = linalg.solvemdbi_ism(X, rho, XHop(S) + rho*Z, 4, 3)
        assert linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11


    def test_13(self):
        rho = 1e-1
        N = 32
        M = 16
        K = 8
        D = complex_randn(N, N, 1, 1, M)
        X = complex_randn(N, N, 1, K, M)
        S = np.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: np.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: np.sum(np.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv = linalg.solvemdbi_rsm(X, rho, XHop(S) + rho*Z, 3)
        assert linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11


    def test_14(self):
        rho = 1e-1
        N = 64
        M = 32
        C = 3
        K = 8
        D = complex_randn(N, N, C, 1, M)
        X = complex_randn(N, N, 1, K, M)
        S = np.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: np.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: np.sum(np.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv = linalg.solvemdbi_rsm(X, rho, XHop(S) + rho*Z, 3)
        assert linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11


    @pytest.mark.skipif(platform.system() == 'Windows',
                        reason='Feature not supported under Windows')
    def test_15(self):
        rho = 1e-1
        N = 32
        M = 16
        K = 8
        D = complex_randn(N, N, 1, 1, M)
        X = complex_randn(N, N, 1, K, M)
        S = np.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: np.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: np.sum(np.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv, cgit = linalg.solvemdbi_cg(X, rho, XHop(S)+rho*Z, 4, 3, tol=1e-6)
        assert linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) <= 1e-6


    @pytest.mark.skipif(platform.system() == 'Windows',
                        reason='Feature not supported under Windows')
    def test_16(self):
        rho = 1e-1
        N = 64
        M = 32
        C = 3
        K = 8
        D = complex_randn(N, N, C, 1, M)
        X = complex_randn(N, N, 1, K, M)
        S = np.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: np.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: np.sum(np.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv, cgit = linalg.solvemdbi_cg(X, rho, XHop(S)+rho*Z, 4, 3, tol=1e-6)
        assert linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) <= 1e-6


    @pytest.mark.skip(reason="Function linalg.proj_l2ball to be deprecated")
    def test_17(self):
        b = np.array([0.0, 0.0, 2.0])
        s = np.array([0.0, 0.0, 0.0])
        r = 1.0
        p = linalg.proj_l2ball(b, s, r)
        assert linalg.rrs(p, np.array([0.0, 0.0, 1.0])) < 1e-14


    def test_18(self):
        u0 = np.array([[0, 1], [2, 3]])
        u1 = np.array([[4, 5], [6, 7]])
        C = linalg.block_circulant((u0, u1))
        assert C[3, 0] == 6
        assert C[3, 3] == 3


    def test_19(self):
        x = np.random.randn(16, 8)
        y = np.random.randn(16, 8)
        ip1 = np.sum(x * y, axis=0, keepdims=True)
        ip2 = linalg.inner(x, y, axis=0)
        assert np.linalg.norm(ip1 - ip2) < 1e-13


    def test_20(self):
        x = np.random.randn(8, 8, 3, 12)
        y = np.random.randn(8, 1, 1, 12)
        ip1 = np.sum(x * y, axis=-1, keepdims=True)
        ip2 = linalg.inner(x, y, axis=-1)
        assert np.linalg.norm(ip1 - ip2) < 1e-13


    def test_21(self):
        a = np.random.randn(7, 8)
        b = np.random.randn(8, 12)
        c1 = a.dot(b)
        c2 = linalg.dot(a, b)
        assert np.linalg.norm(c1 - c2) < 1e-14


    def test_22(self):
        a = np.random.randn(7, 8)
        b = np.random.randn(3, 4, 8, 12)
        c1 = np.zeros((3, 4, 7, 12))
        for i0 in range(c1.shape[0]):
            for i1 in range(c1.shape[1]):
                c1[i0, i1] = a.dot(b[i0, i1])
        c2 = linalg.dot(a, b)
        assert np.linalg.norm(c1 - c2) < 2e-14


    def test_23(self):
        a = np.random.randn(7, 8)
        b = np.random.randn(3, 8, 4, 12)
        c1 = np.zeros((3, 7, 4, 12))
        for i0 in range(c1.shape[0]):
            for i1 in range(c1.shape[3]):
                c1[i0, ..., i1] = a.dot(b[i0, ..., i1])
        c2 = linalg.dot(a, b, axis=1)
        assert np.linalg.norm(c1 - c2) < 2e-14


    def test_24(self):
        U = np.random.randn(5, 10)
        B, S, C = linalg.pca(U, centre=False)
        assert np.linalg.norm(B.dot(B.T) - np.eye(U.shape[0])) < 1e-10


    def test_25(self):
        U = np.random.randn(5, 10)
        B, S, C = linalg.pca(U, centre=True)
        assert np.linalg.norm(B.dot(B.T) - np.eye(U.shape[0])) < 1e-10


    def test_26(self):
        B0 = np.arange(1, 7).reshape(3, 2)
        C0 = np.arange(1, 5).reshape(2, 2)
        A0 = np.kron(B0, C0)
        B, C = linalg.nkp(A0, B0.shape, C0.shape)
        assert mse(A0, np.kron(B, C)) < 1e-12
        r = B0[0, 0] / B[0, 0]
        B *= r
        C /= r
        assert mse(B0, B) < 1e-12
        assert mse(C0, C) < 1e-12


    def test_27(self):
        B0 = np.random.randn(3, 5, 2)
        C0 = np.random.randn(4, 6, 2)
        A0 = kronsum(np.ones((2,)), B0, C0)
        S, B, C = linalg.kpsvd(A0, B0.shape[0:2], C0.shape[0:2])
        A = kronsum(S, B[..., 0:2], C[..., 0:2])
        assert mse(A0, A) < 1e-12


    def test_28(self):
        alpha = np.random.randn()
        N = 7
        M = 10
        X0 = np.random.randn(N, M)
        A = np.random.randn(N, 1)
        B = np.random.randn(1, M)
        C = A * X0 * B + alpha * X0
        X1 = linalg.solve_symmetric_sylvester(A, B, C, alpha)
        assert mse(X0, X1) < 1e-14
        A = A[:, 0]
        B = B[0]
        X1 = linalg.solve_symmetric_sylvester(A, B, C, alpha)
        assert mse(X0, X1) < 1e-14


    def test_29(self):
        alpha = np.random.randn()
        N = 7
        M = 10
        X0 = np.random.randn(N, M)
        A = np.random.randn(N, 1)
        Bg = np.random.randn(M, 4)
        B = Bg.dot(Bg.T)
        C = A * X0.dot(B) + alpha * X0
        X1 = linalg.solve_symmetric_sylvester(A, B, C, alpha)
        assert mse(X0, X1) < 1e-12
        Lambda, Q = np.linalg.eigh(B)
        X1 = linalg.solve_symmetric_sylvester(A, (Lambda, Q), C, alpha)
        assert mse(X0, X1) < 1e-12


    def test_30(self):
        alpha = np.random.randn()
        N = 7
        M = 10
        X0 = np.random.randn(N, M)
        Ag = np.random.randn(4, N)
        A = Ag.T.dot(Ag)
        B = np.random.randn(1, M)
        C = A.dot(X0) * B + alpha * X0
        X1 = linalg.solve_symmetric_sylvester(A, B, C, alpha)
        assert mse(X0, X1) < 1e-12
        Lambda, Q = np.linalg.eigh(A)
        X1 = linalg.solve_symmetric_sylvester((Lambda, Q), B, C, alpha)
        assert mse(X0, X1) < 1e-12


    def test_31(self):
        alpha = np.random.randn()
        N = 8
        M = 11
        X0 = np.random.randn(N, M)
        Ag = np.random.randn(4, N)
        A = Ag.T.dot(Ag)
        Bg = np.random.randn(M, 4)
        B = Bg.dot(Bg.T)
        C = A.dot(X0).dot(B) + alpha * X0
        X1 = linalg.solve_symmetric_sylvester(A, B, C, alpha)
        assert mse(X0, X1) < 1e-12
        LambdaA, QA = np.linalg.eigh(A)
        LambdaB, QB = np.linalg.eigh(B)
        X1 = linalg.solve_symmetric_sylvester((LambdaA, QA), (LambdaB, QB),
                                              C, alpha)
        assert mse(X0, X1) < 1e-12


    def test_32(self):
        Amx = np.random.randn(7, 6)
        ATmx = Amx.T.copy()
        A = lambda x : np.matmul(Amx, x)
        AT = lambda x : np.matmul(ATmx, x)
        assert linalg.valid_adjoint(A, AT, (Amx.shape[1],), (Amx.shape[0],))
        ATmx[0, 1] *= 2
        AT = lambda x : np.matmul(ATmx, x)
        assert not linalg.valid_adjoint(A, AT, (Amx.shape[1],),
                                        (Amx.shape[0],))

