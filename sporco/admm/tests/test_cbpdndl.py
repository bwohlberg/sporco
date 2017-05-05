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
        opt = cbpdndl.ConvBPDNDictLearn.Options({'MaxMainIter' : 10})
        try:
            b = cbpdndl.ConvBPDNDictLearn(self.D0, self.S[...,0], lmbda,
                                          opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_02(self):
        lmbda = 1e-1
        opt = cbpdndl.ConvBPDNDictLearn.Options({'MaxMainIter' : 10})
        try:
            b = cbpdndl.ConvBPDNDictLearn(self.D0, self.S, lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_03(self):
        opt = cbpdndl.ConvBPDNDictLearn.Options({'MaxMainIter' : 10})
        try:
            b = cbpdndl.ConvBPDNDictLearn(self.D0, self.S, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_04(self):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, Nc, M)
        S = np.random.randn(N, N, Nc, K)
        lmbda = 1e-1
        opt = cbpdndl.ConvBPDNDictLearn.Options({'MaxMainIter': 10})
        try:
            b = cbpdndl.ConvBPDNDictLearn(D0, S, lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_05(self):
        lmbda = 1e-1
        opt = cbpdndl.ConvBPDNDictLearn.Options({'AccurateDFid' : True,
                                                 'MaxMainIter' : 10})
        try:
            b = cbpdndl.ConvBPDNDictLearn(self.D0, self.S, lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)
