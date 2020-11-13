from __future__ import division
from builtins import object
from past.utils import old_div

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


from sporco.cupy.admm import tvl1
import sporco.cupy.metric as sm


class TestSet01(object):

    def setup_method(self, method):
        cp.random.seed(12345)
        self.D = cp.random.randn(16, 15)
        self.Dc = cp.random.randn(16, 15) + 1j * cp.random.randn(16, 15)


    def test_01(self):
        lmbda = 3
        try:
            b = tvl1.TVL1Denoise(self.D, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_01cplx(self):
        lmbda = 3
        try:
            b = tvl1.TVL1Denoise(self.Dc, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        lmbda = 3
        try:
            b = tvl1.TVL1Deconv(cp.ones((1, 1)), self.D, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02cplx(self):
        lmbda = 3
        try:
            b = tvl1.TVL1Deconv(cp.ones((1, 1)), self.Dc, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        lmbda = 3
        dt = cp.float16
        opt = tvl1.TVL1Denoise.Options(
            {'Verbose': False, 'MaxMainIter': 20, 'AutoRho':
             {'Enabled': True}, 'DataType': dt})
        b = tvl1.TVL1Denoise(self.D, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_04(self):
        lmbda = 3
        dt = cp.float32
        opt = tvl1.TVL1Denoise.Options(
            {'Verbose': False, 'MaxMainIter': 20, 'AutoRho':
             {'Enabled': True}, 'DataType': dt})
        b = tvl1.TVL1Denoise(self.D, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_05(self):
        lmbda = 3
        dt = cp.float64
        opt = tvl1.TVL1Denoise.Options(
            {'Verbose': False, 'MaxMainIter': 20, 'AutoRho':
             {'Enabled': True}, 'DataType': dt})
        b = tvl1.TVL1Denoise(self.D, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_06(self):
        lmbda = 3
        dt = cp.float32
        opt = tvl1.TVL1Deconv.Options(
            {'Verbose': False, 'MaxMainIter': 20, 'AutoRho':
             {'Enabled': True}, 'DataType': dt})
        b = tvl1.TVL1Deconv(cp.ones((1, )), self.D, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_07(self):
        lmbda = 3
        dt = cp.float64
        opt = tvl1.TVL1Deconv.Options(
            {'Verbose': False, 'MaxMainIter': 20, 'AutoRho':
             {'Enabled': True}, 'DataType': dt})
        b = tvl1.TVL1Deconv(cp.ones((1, )), self.D, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_08(self):
        lmbda = 3
        opt = tvl1.TVL1Denoise.Options({'MaxMainIter': 20})
        b = tvl1.TVL1Denoise(self.D, lmbda, opt)
        b.solve()
        opt['Y0'] = b.Y
        try:
            c = tvl1.TVL1Denoise(self.D, lmbda, opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_09(self):
        lmbda = 3
        opt = tvl1.TVL1Deconv.Options({'MaxMainIter': 20})
        b = tvl1.TVL1Deconv(cp.ones((1, )), self.D, lmbda, opt)
        b.solve()
        opt['Y0'] = b.Y
        try:
            c = tvl1.TVL1Deconv(cp.ones((1, )), self.D, lmbda, opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0




class TestSet02(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 64
        L = 20
        self.U = cp.ones((N, N))
        self.U[:, 0:(old_div(N, 2))] = -1
        self.V = cp.asarray(np.random.randn(N, N))
        t = cp.sort(cp.abs(self.V).ravel())[self.V.size - L]
        self.V[cp.abs(self.V) < t] = 0
        self.D = self.U + self.V


    def test_01(self):
        lmbda = 3
        opt = tvl1.TVL1Denoise.Options({'Verbose': False, 'gEvalY': False,
                                        'MaxMainIter': 250})
        b = tvl1.TVL1Denoise(self.D, lmbda, opt)
        X = b.solve()
        assert cp.abs(b.itstat[-1].ObjFun - 447.78101756451662) < 1e-6
        assert sm.mse(self.U, X) < 1e-6


    def test_02(self):
        lmbda = 3
        opt = tvl1.TVL1Deconv.Options({'Verbose': False, 'gEvalY': False,
                                       'MaxMainIter': 250, 'rho': 10.0})
        b = tvl1.TVL1Deconv(cp.ones((1, )), self.D, lmbda, opt)
        X = b.solve()
        assert cp.abs(b.itstat[-1].ObjFun - 831.88219947939172) < 1e-5
        assert sm.mse(self.U, X) < 1e-4




class TestSet03(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 32
        L = 20
        self.U = cp.ones((N, N, N))
        self.U[:, 0:(old_div(N, 2))] = -1
        self.V = cp.asarray(np.random.randn(N, N, N))
        t = cp.sort(cp.abs(self.V).ravel())[self.V.size - L]
        self.V[cp.abs(self.V) < t] = 0
        self.D = self.U + self.V


    def test_01(self):
        lmbda = 3
        opt = tvl1.TVL1Denoise.Options({'Verbose': False, 'gEvalY': False,
                                        'MaxMainIter': 250})
        b = tvl1.TVL1Denoise(self.D, lmbda, opt, axes=(0, 1))
        X = b.solve()
        assert cp.abs(b.itstat[-1].ObjFun - 6219.3241727233126) < 1e-6
        assert sm.mse(self.U, X) < 1e-6


    def test_02(self):
        lmbda = 3
        opt = tvl1.TVL1Deconv.Options({'Verbose': False, 'gEvalY': False,
                                       'MaxMainIter': 250, 'rho': 10.0})
        b = tvl1.TVL1Deconv(cp.ones((1, )), self.D, lmbda, opt, axes=(0, 1))
        X = b.solve()
        assert cp.abs(b.itstat[-1].ObjFun - 12364.029061174046) < 1e-5
        assert sm.mse(self.U, X) < 1e-4




class TestSet04(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 32
        L = 20
        self.U = cp.ones((N, N, N))
        self.U[:, 0:(old_div(N, 2))] = -1
        self.V = cp.asarray(np.random.randn(N, N, N))
        t = cp.sort(cp.abs(self.V).ravel())[self.V.size - L]
        self.V[cp.abs(self.V) < t] = 0
        self.D = self.U + self.V


    def test_01(self):
        lmbda = 3
        opt = tvl1.TVL1Denoise.Options({'Verbose': False, 'gEvalY': False,
                                        'MaxMainIter': 250})
        b = tvl1.TVL1Denoise(self.D, lmbda, opt, axes=(0, 1, 2))
        X = b.solve()
        assert cp.abs(b.itstat[-1].ObjFun - 6219.6209699337605) < 1e-6
        assert sm.mse(self.U, X) < 1e-6


    def test_02(self):
        lmbda = 3
        opt = tvl1.TVL1Deconv.Options({'Verbose': False, 'gEvalY': False,
                                       'MaxMainIter': 250, 'rho': 10.0})
        b = tvl1.TVL1Deconv(cp.ones((1, )), self.D, lmbda, opt, axes=(0, 1, 2))
        X = b.solve()
        assert cp.abs(b.itstat[-1].ObjFun - 12363.969118576981) < 1e-5
        assert sm.mse(self.U, X) < 1e-4
