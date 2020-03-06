from builtins import object

import pytest
import numpy as np
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).compute_capability
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("GPU device inaccessible", allow_module_level=True)
except ImportError:
    pytest.skip("cupy not installed", allow_module_level=True)

from sporco.cupy.admm import rpca
import sporco.cupy.metric as sm


class TestSet01(object):

    def setup_method(self, method):
        cp.random.seed(12345)


    def test_01(self):
        N = 64
        K = 5
        L = 10
        u = cp.random.randn(N, K)
        U = cp.dot(u, u.T)
        V = cp.random.randn(N, N)
        t = cp.sort(cp.abs(V).ravel())[V.size-L]
        V[cp.abs(V) < t] = 0
        D = U + V
        opt = rpca.RobustPCA.Options({'Verbose': False, 'gEvalY': False,
                                      'MaxMainIter': 250,
                                      'AutoRho': {'Enabled': True}})
        b = rpca.RobustPCA(D, None, opt)
        X, Y = b.solve()
        assert sm.mse(U, X) < 5e-6
        assert sm.mse(V, Y) < 1e-8


    def test_02(self):
        N = 8
        D = cp.random.randn(N, N)
        try:
            b = rpca.RobustPCA(D)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        N = 8
        D = cp.random.randn(N, N)
        dt = cp.float16
        opt = rpca.RobustPCA.Options({'Verbose': False, 'MaxMainIter': 20,
                            'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = rpca.RobustPCA(D, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_04(self):
        N = 8
        D = cp.random.randn(N, N)
        dt = cp.float32
        opt = rpca.RobustPCA.Options({'Verbose': False, 'MaxMainIter': 20,
                            'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = rpca.RobustPCA(D, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_05(self):
        N = 8
        D = cp.random.randn(N, N)
        dt = cp.float64
        opt = rpca.RobustPCA.Options({'Verbose': False, 'MaxMainIter': 20,
                            'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = rpca.RobustPCA(D, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt
