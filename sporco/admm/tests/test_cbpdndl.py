from __future__ import division
from builtins import object

import pytest
import numpy as np

from sporco.admm import cbpdndl



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 16
        Nd = 5
        M = 4
        K = 3
        self.D0 = np.random.randn(Nd, Nd, M)
        self.S = np.random.randn(N, N, K)


    def test_01(self):
        lmbda = 1e-1
        try:
            b = cbpdndl.ConvBPDNDictLearn(self.D0, self.S[...,0], lmbda,
                                          dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_02(self):
        lmbda = 1e-1
        try:
            b = cbpdndl.ConvBPDNDictLearn(self.D0, self.S, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_03(self):
        try:
            b = cbpdndl.ConvBPDNDictLearn(self.D0, self.S)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_04(self):
        lmbda = 1e-1
        opt = cbpdndl.ConvBPDNDictLearn.Options({'AccurateDFid' : True,
                                                 'MaxMainIter' : 10})
        try:
            b = cbpdndl.ConvBPDNDictLearn(self.D0, self.S, lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)
