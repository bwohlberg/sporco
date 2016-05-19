from __future__ import division
from builtins import object
from past.utils import old_div

import pytest
import numpy as np

from sporco.admm import tvl2
import sporco.linalg as sl


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 16
        self.D = np.random.randn(N,N)


    def test_01(self):
        lmbda = 3
        try:
            b = tvl2.TVL2Denoise(self.D, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_02(self):
        lmbda = 3
        try:
            b = tvl2.TVL2Deconv(np.ones((1,1)), self.D, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)




class TestSet02(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 64
        self.U = np.ones((N,N))
        self.U[:, 0:(old_div(N,2))] = -1
        self.V = 1e-1 * np.random.randn(N, N)
        self.D = self.U + self.V


    def test_01(self):
        lmbda = 1e-1
        opt = tvl2.TVL2Denoise.Options({'Verbose' : False, 'gEvalY' : False,
                                        'MaxMainIter' : 300, 'rho' : 75*lmbda})
        b = tvl2.TVL2Denoise(self.D, lmbda, opt)
        X = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 32.875710674129564) < 1e-3)
        assert(sl.mse(self.U,X) < 1e-3)


    def test_02(self):
        lmbda = 1e-1
        opt = tvl2.TVL2Deconv.Options({'Verbose' : False, 'gEvalY' : False,
                                       'MaxMainIter' : 250})
        b = tvl2.TVL2Deconv(np.ones((1)), self.D, lmbda, opt)
        X = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 45.441537456677) < 1e-3)
        assert(sl.mse(self.U,X) < 1e-3)




class TestSet03(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 32
        self.U = np.ones((N,N,N))
        self.U[:, 0:(old_div(N,2)), :] = -1
        self.V = 1e-1 * np.random.randn(N,N,N)
        self.D = self.U + self.V


    def test_01(self):
        lmbda = 1e-1
        opt = tvl2.TVL2Denoise.Options({'Verbose' : False, 'gEvalY' : False,
                                        'MaxMainIter' : 250, 'rho' : 10*lmbda})
        b = tvl2.TVL2Denoise(self.D, lmbda, opt, axes=(0,1))
        X = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 363.03797264834986) < 1e-3)
        assert(sl.mse(self.U,X) < 1e-3)


    def test_02(self):
        lmbda = 1e-1
        opt = tvl2.TVL2Deconv.Options({'Verbose' : False, 'gEvalY' : False,
                                       'MaxMainIter' : 250})
        b = tvl2.TVL2Deconv(np.ones((1)), self.D, lmbda, opt, axes=(0,1))
        X = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 563.96797296313059) < 1e-3)
        assert(sl.mse(self.U,X) < 1e-3)




class TestSet04(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 32
        self.U = np.ones((N,N,N))
        self.U[:, 0:(old_div(N,2)), :] = -1
        self.V = 1e-1 * np.random.randn(N,N,N)
        self.D = self.U + self.V


    def test_01(self):
        lmbda = 1e-1
        opt = tvl2.TVL2Denoise.Options({'Verbose' : False, 'gEvalY' : False,
                                        'MaxMainIter' : 250, 'rho' : 10*lmbda})
        b = tvl2.TVL2Denoise(self.D, lmbda, opt, axes=(0,1,2))
        X = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 366.04267554965134) < 1e-3)
        assert(sl.mse(self.U,X) < 1e-3)


    def test_02(self):
        lmbda = 1e-1
        opt = tvl2.TVL2Deconv.Options({'Verbose' : False, 'gEvalY' : False,
                                       'MaxMainIter' : 250})
        b = tvl2.TVL2Deconv(np.ones((1)), self.D, lmbda, opt, axes=(0,1,2))
        X = b.solve()
        assert(np.abs(b.itstat[-1].ObjFun - 567.49092052210574) < 1e-3)
        assert(sl.mse(self.U,X) < 1e-3)
