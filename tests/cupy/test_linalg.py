from __future__ import division
from builtins import object

import pytest
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).compute_capability
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("GPU device inaccessible", allow_module_level=True)
except ImportError:
    pytest.skip("cupy not installed", allow_module_level=True)


import cupy as cp
from sporco.cupy import linalg
from sporco.cupy import signal



class TestSet01(object):

    def setup_method(self, method):
        cp.random.seed(12345)


    def test_05(self):
        rho = 1e-1
        N = 64
        M = 128
        K = 32
        D = cp.random.randn(N, M)
        X = cp.random.randn(M, K)
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
        D = cp.random.randn(N, M)
        X = cp.random.randn(M, K)
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
        D = cp.random.randn(N, M)
        X = cp.random.randn(M, K)
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
        D = cp.random.randn(N, M)
        X = cp.random.randn(M, K)
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
        D = signal.complex_randn(N, N, 1, 1, M)
        X = signal.complex_randn(N, N, 1, K, M)
        S = cp.sum(D*X, axis=4, keepdims=True)
        Z = (D.conj()*cp.sum(D*X, axis=4, keepdims=True) + \
             rho*X - D.conj()*S) / rho
        Xslv = linalg.solvedbi_sm(D, rho, D.conj()*S + rho*Z)
        assert linalg.rrs(D.conj()*cp.sum(D*Xslv, axis=4, keepdims=True) +
                          rho * Xslv, D.conj() * S + rho*Z) < 1e-11



    def test_10(self):
        N = 32
        M = 16
        K = 8
        D = signal.complex_randn(N, N, 1, 1, M)
        X = signal.complex_randn(N, N, 1, K, M)
        S = cp.sum(D*X, axis=4, keepdims=True)
        d = 1e-1 * (cp.random.randn(N, N, 1, 1, M).astype('complex') +
            cp.random.randn(N, N, 1, 1, M).astype('complex') * 1.0j)
        Z = (D.conj()*cp.sum(D*X, axis=4, keepdims=True) +
             d*X - D.conj()*S) / d
        Xslv = linalg.solvedbd_sm(D, d, D.conj()*S + d*Z)
        assert linalg.rrs(D.conj()*cp.sum(D*Xslv, axis=4, keepdims=True) +
                          d*Xslv, D.conj()*S + d*Z) < 1e-11



    def test_11(self):
        rho = 1e-1
        N = 32
        M = 16
        K = 8
        D = signal.complex_randn(N, N, 1, 1, M)
        X = signal.complex_randn(N, N, 1, K, M)
        S = cp.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: cp.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: cp.sum(cp.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv = linalg.solvemdbi_ism(X, rho, XHop(S) + rho*Z, 4, 3)

        assert linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11



    def test_12(self):
        rho = 1e-1
        N = 32
        M = 16
        C = 3
        K = 8
        D = signal.complex_randn(N, N, C, 1, M)
        X = signal.complex_randn(N, N, 1, K, M)
        S = cp.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: cp.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: cp.sum(cp.conj(X)* x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv = linalg.solvemdbi_ism(X, rho, XHop(S) + rho*Z, 4, 3)

        assert linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11



    def test_13(self):
        rho = 1e-1
        N = 32
        M = 16
        K = 8
        D = signal.complex_randn(N, N, 1, 1, M)
        X = signal.complex_randn(N, N, 1, K, M)
        S = cp.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: cp.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: cp.sum(cp.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv = linalg.solvemdbi_rsm(X, rho, XHop(S) + rho*Z, 3)

        assert linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11



    def test_14(self):
        rho = 1e-1
        N = 64
        M = 32
        C = 3
        K = 8
        D = signal.complex_randn(N, N, C, 1, M)
        X = signal.complex_randn(N, N, 1, K, M)
        S = cp.sum(D*X, axis=4, keepdims=True)

        Xop = lambda x: cp.sum(X * x, axis=4, keepdims=True)
        XHop = lambda x: cp.sum(cp.conj(X) * x, axis=3, keepdims=True)
        Z = (XHop(Xop(D)) + rho*D - XHop(S)) / rho
        Dslv = linalg.solvemdbi_rsm(X, rho, XHop(S) + rho*Z, 3)

        assert linalg.rrs(XHop(Xop(Dslv)) + rho*Dslv, XHop(S) + rho*Z) < 1e-11
