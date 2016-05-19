from __future__ import division
from builtins import object
from past.utils import old_div

import pytest
import numpy as np

from sporco.admm import tvl1
import sporco.linalg as sl


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 16
        self.D = np.random.randn(N,N)


    def test_01(self):
        lmbda = 3
        try:
            b = tvl1.TVL1Denoise(self.D, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_02(self):
        lmbda = 3
        try:
            b = tvl1.TVL1Deconv(np.ones((1,1)), self.D, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_03(self):
        lmbda = 3
        opt = tvl1.TVL1Denoise.Options({'MaxMainIter' : 20})
        b = tvl1.TVL1Denoise(self.D, lmbda, opt)
        b.solve()
        opt['Y0'] = b.Y
        try:
            c = tvl1.TVL1Denoise(self.D, lmbda, opt)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_04(self):
        lmbda = 3
        opt = tvl1.TVL1Deconv.Options({'MaxMainIter' : 20})
        b = tvl1.TVL1Deconv(np.ones((1,1)), self.D, lmbda, opt)
        b.solve()
        opt['Y0'] = b.Y
        try:
            c = tvl1.TVL1Deconv(np.ones((1,1)), self.D, lmbda, opt)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)




class TestSet02(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 64
        L = 20
        self.U = np.ones((N,N))
        self.U[:, 0:(old_div(N,2))] = -1
        self.V = np.random.randn(N,N)
        t = np.sort(np.abs(self.V).ravel())[self.V.size-L]
        self.V[np.abs(self.V) < t] = 0
        self.D = self.U + self.V


    def test_01(self):
        lmbda = 3
        opt = tvl1.TVL1Denoise.Options({'Verbose' : False, 'gEvalY' : False,
                                        'MaxMainIter' : 250})
        b = tvl1.TVL1Denoise(self.D, lmbda, opt)
        X = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 447.78101756451662) < 1e-6)
        assert(sl.mse(self.U,X) < 1e-6)


    def test_02(self):
        lmbda = 3
        opt = tvl1.TVL1Deconv.Options({'Verbose' : False, 'gEvalY' : False,
                                       'MaxMainIter' : 250, 'rho' : 10.0})
        b = tvl1.TVL1Deconv(np.ones((1,1)), self.D, lmbda, opt)
        X = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 831.88219947939172) < 1e-5)
        assert(sl.mse(self.U,X) < 1e-4)




class TestSet03(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 32
        L = 20
        self.U = np.ones((N,N,N))
        self.U[:, 0:(old_div(N,2))] = -1
        self.V = np.random.randn(N,N,N)
        t = np.sort(np.abs(self.V).ravel())[self.V.size-L]
        self.V[np.abs(self.V) < t] = 0
        self.D = self.U + self.V


    def test_01(self):
        lmbda = 3
        opt = tvl1.TVL1Denoise.Options({'Verbose' : False, 'gEvalY' : False,
                                        'MaxMainIter' : 250})
        b = tvl1.TVL1Denoise(self.D, lmbda, opt, axes=(0,1))
        X = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 6219.3241727233126) < 1e-6)
        assert(sl.mse(self.U,X) < 1e-6)


    def test_02(self):
        lmbda = 3
        opt = tvl1.TVL1Deconv.Options({'Verbose' : False, 'gEvalY' : False,
                                       'MaxMainIter' : 250, 'rho' : 10.0})
        b = tvl1.TVL1Deconv(np.ones((1,1)), self.D, lmbda, opt, axes=(0,1))
        X = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 12364.029061174046) < 1e-5)
        assert(sl.mse(self.U,X) < 1e-4)




class TestSet04(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 32
        L = 20
        self.U = np.ones((N,N,N))
        self.U[:, 0:(old_div(N,2))] = -1
        self.V = np.random.randn(N,N,N)
        t = np.sort(np.abs(self.V).ravel())[self.V.size-L]
        self.V[np.abs(self.V) < t] = 0
        self.D = self.U + self.V


    def test_01(self):
        lmbda = 3
        opt = tvl1.TVL1Denoise.Options({'Verbose' : False, 'gEvalY' : False,
                                        'MaxMainIter' : 250})
        b = tvl1.TVL1Denoise(self.D, lmbda, opt, axes=(0,1,2))
        X = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 6219.6209699337605) < 1e-6)
        assert(sl.mse(self.U,X) < 1e-6)


    def test_02(self):
        lmbda = 3
        opt = tvl1.TVL1Deconv.Options({'Verbose' : False, 'gEvalY' : False,
                                       'MaxMainIter' : 250, 'rho' : 10.0})
        b = tvl1.TVL1Deconv(np.ones((1,1)), self.D, lmbda, opt, axes=(0,1,2))
        X = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 12363.969118576981) < 1e-5)
        assert(sl.mse(self.U,X) < 1e-4)
