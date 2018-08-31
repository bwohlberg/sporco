from builtins import object

import numpy as np

from sporco.admm import rpca
import sporco.metric as sm


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
        opt = rpca.RobustPCA.Options({'Verbose': False, 'gEvalY': False,
                              'MaxMainIter': 250,
                              'AutoRho': {'Enabled': True}})
        b = rpca.RobustPCA(D, None, opt)
        X, Y = b.solve()
        assert np.abs(b.itstat[-1].ObjFun - 321.493968419) < 1e-6
        assert sm.mse(U, X) < 5e-6
        assert sm.mse(V, Y) < 1e-8


    def test_02(self):
        N = 8
        D = np.random.randn(N, N)
        try:
            b = rpca.RobustPCA(D)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        N = 8
        D = np.random.randn(N, N)
        dt = np.float16
        opt = rpca.RobustPCA.Options({'Verbose': False, 'MaxMainIter': 20,
                            'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = rpca.RobustPCA(D, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_04(self):
        N = 8
        D = np.random.randn(N, N)
        dt = np.float32
        opt = rpca.RobustPCA.Options({'Verbose': False, 'MaxMainIter': 20,
                            'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = rpca.RobustPCA(D, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_05(self):
        N = 8
        D = np.random.randn(N, N)
        dt = np.float64
        opt = rpca.RobustPCA.Options({'Verbose': False, 'MaxMainIter': 20,
                            'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = rpca.RobustPCA(D, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt
