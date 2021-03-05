from __future__ import division
from builtins import object
from past.utils import old_div

import numpy as np

from sporco.admm import tvl2
import sporco.metric as sm


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)
        self.D = np.random.randn(16, 15)
        self.Dc = np.random.randn(16, 15) + 1j * np.random.randn(16, 15)


    def test_01(self):
        lmbda = 3
        try:
            b = tvl2.TVL2Denoise(self.D, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_01cplx(self):
        lmbda = 3
        try:
            b = tvl2.TVL2Denoise(self.Dc, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        lmbda = 3
        try:
            b = tvl2.TVL2Deconv(np.ones((1, 1)), self.D, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02cplx(self):
        lmbda = 3
        try:
            b = tvl2.TVL2Deconv(np.ones((1, 1)), self.Dc, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        lmbda = 3
        dt = np.float16
        opt = tvl2.TVL2Denoise.Options({'Verbose': False, 'MaxMainIter': 20,
                            'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = tvl2.TVL2Denoise(self.D, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_04(self):
        lmbda = 3
        dt = np.float32
        opt = tvl2.TVL2Denoise.Options({'Verbose': False, 'MaxMainIter': 20,
                            'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = tvl2.TVL2Denoise(self.D, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_05(self):
        lmbda = 3
        dt = np.float64
        opt = tvl2.TVL2Denoise.Options({'Verbose': False, 'MaxMainIter': 20,
                            'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = tvl2.TVL2Denoise(self.D, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_06(self):
        lmbda = 3
        dt = np.float32
        opt = tvl2.TVL2Deconv.Options({'Verbose': False, 'MaxMainIter': 20,
                            'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = tvl2.TVL2Deconv(np.ones((1, 1)), self.D, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_07(self):
        lmbda = 3
        dt = np.float64
        opt = tvl2.TVL2Deconv.Options({'Verbose': False, 'MaxMainIter': 20,
                            'AutoRho': {'Enabled': True}, 'DataType': dt})
        b = tvl2.TVL2Deconv(np.ones((1, 1)), self.D, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt





class TestSet02(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 64
        self.U = np.ones((N, N))
        self.U[:, 0:(old_div(N, 2))] = -1
        self.V = 1e-1 * np.random.randn(N, N)
        self.D = self.U + self.V


    def test_01(self):
        lmbda = 1e-1
        opt = tvl2.TVL2Denoise.Options({'Verbose': False, 'gEvalY': False,
                                        'MaxMainIter': 300, 'rho': 75*lmbda})
        b = tvl2.TVL2Denoise(self.D, lmbda, opt)
        X = b.solve()
        assert np.abs(b.itstat[-1].ObjFun - 32.875710674129564) < 1e-3
        assert sm.mse(self.U, X) < 1e-3


    def test_02(self):
        lmbda = 1e-1
        opt = tvl2.TVL2Deconv.Options({'Verbose': False, 'gEvalY': False,
                                       'MaxMainIter': 250})
        b = tvl2.TVL2Deconv(np.ones((1)), self.D, lmbda, opt)
        X = b.solve()
        assert np.abs(b.itstat[-1].ObjFun - 45.45958573088) < 1e-3
        assert sm.mse(self.U, X) < 1e-3




class TestSet03(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 32
        self.U = np.ones((N, N, N))
        self.U[:, 0:(old_div(N, 2)), :] = -1
        self.V = 1e-1 * np.random.randn(N, N, N)
        self.D = self.U + self.V


    def test_01(self):
        lmbda = 1e-1
        opt = tvl2.TVL2Denoise.Options({'Verbose': False, 'gEvalY': False,
                                        'MaxMainIter': 250, 'rho': 10*lmbda})
        b = tvl2.TVL2Denoise(self.D, lmbda, opt, axes=(0, 1))
        X = b.solve()
        assert np.abs(b.itstat[-1].ObjFun - 363.0802047) < 1e-3
        assert sm.mse(self.U, X) < 1e-3


    def test_02(self):
        lmbda = 1e-1
        opt = tvl2.TVL2Deconv.Options({'Verbose': False, 'gEvalY': False,
                                       'MaxMainIter': 250})
        b = tvl2.TVL2Deconv(np.ones((1)), self.D, lmbda, opt, axes=(0, 1))
        X = b.solve()
        assert np.abs(b.itstat[-1].ObjFun - 564.1586542) < 1e-3
        assert sm.mse(self.U, X) < 1e-3




class TestSet04(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 32
        self.U = np.ones((N, N, N))
        self.U[:, 0:(old_div(N, 2)), :] = -1
        self.V = 1e-1 * np.random.randn(N, N, N)
        self.D = self.U + self.V


    def test_01(self):
        lmbda = 1e-1
        opt = tvl2.TVL2Denoise.Options({'Verbose': False, 'gEvalY': False,
                                        'MaxMainIter': 250, 'rho': 10*lmbda})
        b = tvl2.TVL2Denoise(self.D, lmbda, opt, axes=(0, 1, 2))
        X = b.solve()
        assert np.abs(b.itstat[-1].ObjFun - 366.04267554965134) < 1e-3
        assert sm.mse(self.U, X) < 1e-3


    def test_02(self):
        lmbda = 1e-1
        opt = tvl2.TVL2Deconv.Options({'Verbose': False, 'gEvalY': False,
                                       'MaxMainIter': 250})
        b = tvl2.TVL2Deconv(np.ones((1)), self.D, lmbda, opt, axes=(0, 1, 2))
        X = b.solve()
        assert np.abs(b.itstat[-1].ObjFun - 567.72425227) < 1e-3
        assert sm.mse(self.U, X) < 1e-3
