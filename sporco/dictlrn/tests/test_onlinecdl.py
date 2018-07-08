from __future__ import division
from builtins import object

import pytest
import numpy as np

from sporco.dictlrn import onlinecdl



class TestSet01(object):

    def setup_method(self, method):
        N = 16
        Nd = 5
        M = 4
        K = 3
        np.random.seed(12345)
        self.D0 = np.random.randn(Nd, Nd, M)
        self.S = np.random.randn(N, N, K)


    def test_01(self):
        lmbda = 1e-1
        opt = onlinecdl.OnlineConvBPDNDictLearn.Options({'MaxMainIter': 10})
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, self.S[...,[0]],
                                                  lmbda, opt=opt)
            for it in range(opt['MaxMainIter']):
                b.solve(self.S[..., [0]])
        except Exception as e:
            print(e)
            assert(0)


    def test_02(self):
        lmbda = 1e-1
        opt = onlinecdl.OnlineConvBPDNDictLearn.Options({'MaxMainIter': 10})
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, self.S[...,[0]],
                                                  lmbda, opt=opt)
            for it in range(opt['MaxMainIter']):
                img_index = np.random.randint(0, self.S.shape[-1])
                b.solve(self.S[..., [img_index]])
        except Exception as e:
            print(e)
            assert(0)


    def test_03(self):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, Nc, M)
        S = np.random.randn(N, N, Nc, K)
        lmbda = 1e-1
        opt = onlinecdl.OnlineConvBPDNDictLearn.Options({'MaxMainIter': 10})
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, S[...,[0]], lmbda,
                                                  opt=opt)
            for it in range(opt['MaxMainIter']):
                img_index = np.random.randint(0, S.shape[-1])
                b.solve(S[..., [img_index]])
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
        opts = onlinecdl.OnlineConvBPDNDictLearn.Options(
            {'MaxMainIter': 10,
             'CBPDN': {'RelaxParam': 1.0, 'AutoRho': {'Enabled': False}}})
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, S[...,[0]], lmbda,
                                                  opt=opts)
            for it in range(opts['MaxMainIter']):
                img_index = np.random.randint(0, S.shape[-1])
                b.solve(S[..., [img_index]])
        except Exception as e:
            print(e)
            assert(0)


    def test_05(self):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, Nc, M)
        S = np.random.randn(N, N, Nc, K)
        lmbda = 1e-1
        opts = onlinecdl.OnlineConvBPDNDictLearn.Options(
            {'MaxMainIter': 10,
             'CBPDN': {'RelaxParam': 1.8, 'AutoRho': {'Enabled': False}}})
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, S[...,[0]], lmbda,
                                                  opt=opts)
            for it in range(opts['MaxMainIter']):
                img_index = np.random.randint(0, S.shape[-1])
                b.solve(S[..., [img_index]])
        except Exception as e:
            print(e)
            assert(0)

