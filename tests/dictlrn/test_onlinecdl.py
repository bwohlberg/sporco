from __future__ import division
from builtins import object

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
        opt = onlinecdl.OnlineConvBPDNDictLearn.Options(
            {'CBPDN': {'MaxMainIter': 10}})
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, lmbda, opt=opt)
            for it in range(10):
                img_index = np.random.randint(0, self.S.shape[-1])
                b.solve(self.S[..., img_index])
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        lmbda = 1e-1
        opt = onlinecdl.OnlineConvBPDNDictLearn.Options(
            {'CBPDN': {'MaxMainIter': 10}})
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, lmbda, opt=opt)
            for it in range(10):
                img_index = np.random.randint(0, self.S.shape[-1])
                b.solve(self.S[..., [img_index]])
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        lmbda = 1e-1
        W = np.random.randn(*self.S.shape[0:2])
        opt = onlinecdl.OnlineConvBPDNMaskDictLearn.Options(
            {'CBPDN': {'MaxMainIter': 10}})
        try:
            b = onlinecdl.OnlineConvBPDNMaskDictLearn(self.D0, lmbda, opt=opt)
            for it in range(10):
                img_index = np.random.randint(0, self.S.shape[-1])
                b.solve(self.S[..., img_index], W)
        except Exception as e:
            print(e)
            assert 0



class TestSet02(object):

    def setup_method(self, method):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        np.random.seed(12345)
        self.D0 = np.random.randn(Nd, Nd, Nc, M)
        self.S = np.random.randn(N, N, Nc, K)


    def test_01(self):
        lmbda = 1e-1
        opt = onlinecdl.OnlineConvBPDNDictLearn.Options(
            {'CBPDN': {'MaxMainIter': 10}})
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, lmbda, opt=opt)
            for it in range(10):
                img_index = np.random.randint(0, self.S.shape[-1])
                b.solve(self.S[..., img_index])
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        lmbda = 1e-1
        opts = onlinecdl.OnlineConvBPDNDictLearn.Options(
            {'CBPDN': {'MaxMainIter': 10, 'AutoRho': {'Enabled': False}}})
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, lmbda, opt=opts)
            for it in range(10):
                img_index = np.random.randint(0, self.S.shape[-1])
                b.solve(self.S[..., [img_index]])
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        lmbda = 1e-1
        W = np.random.randn(*self.S.shape[0:3])
        opt = onlinecdl.OnlineConvBPDNMaskDictLearn.Options(
            {'CBPDN': {'MaxMainIter': 10}})
        try:
            b = onlinecdl.OnlineConvBPDNMaskDictLearn(self.D0, lmbda, opt=opt)
            for it in range(10):
                img_index = np.random.randint(0, self.S.shape[-1])
                b.solve(self.S[..., img_index], W)
        except Exception as e:
            print(e)
            assert 0



class TestSet03(object):

    def setup_method(self, method):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        np.random.seed(12345)
        self.D0 = np.random.randn(Nd, Nd, 1, M)
        self.S = np.random.randn(N, N, Nc, K)


    def test_01(self):
        lmbda = 1e-1
        opt = onlinecdl.OnlineConvBPDNDictLearn.Options(
            {'CBPDN': {'MaxMainIter': 10}})
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, lmbda, opt=opt)
            for it in range(10):
                img_index = np.random.randint(0, self.S.shape[-1])
                b.solve(self.S[..., img_index])
        except Exception as e:
            print(e)
            assert 0
