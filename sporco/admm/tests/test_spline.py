from builtins import object

import numpy as np

from sporco.admm import spline
import sporco.metric as sm


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 64
        L = 20
        x = np.cos(np.linspace(0, np.pi, N))[np.newaxis, :]
        y = np.cos(np.linspace(0, np.pi, N))[:, np.newaxis]
        U = x*y
        V = np.random.randn(N, N)
        t = np.sort(np.abs(V).ravel())[V.size-L]
        V[np.abs(V) < t] = 0
        D = U + V
        lmbda = 0.1
        opt = spline.SplineL1.Options({'Verbose': False, 'gEvalY': False,
                              'MaxMainIter': 250, 'RelStopTol': 5e-4,
                              'DFidWeight': V == 0,
                              'AutoRho': {'Enabled': True}})
        b = spline.SplineL1(D, lmbda, opt)
        X = b.solve()
        assert np.abs(b.itstat[-1].ObjFun - 0.333606246) < 1e-6
        assert sm.mse(U, X) < 1e-6


    def test_02(self):
        N = 8
        D = np.random.randn(N, N)
        lmbda = 0.1
        try:
            b = spline.SplineL1(D, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        N = 8
        D = np.random.randn(N, N)
        lmbda = 0.1
        dt = np.float32
        opt = spline.SplineL1.Options({'Verbose': False, 'MaxMainIter': 20,
                              'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = spline.SplineL1(D, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_04(self):
        N = 8
        D = np.random.randn(N, N)
        lmbda = 0.1
        dt = np.float64
        opt = spline.SplineL1.Options({'Verbose': False, 'MaxMainIter': 20,
                              'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = spline.SplineL1(D, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt
