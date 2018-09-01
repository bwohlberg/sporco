from __future__ import division
from builtins import object

import pytest
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).compute_capability
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("GPU device inaccessible", allow_module_level=True)
except ImportError:
    pytest.skip("cupy not installed", allow_module_level=True)


from sporco.cupy.dictlrn import onlinecdl



class TestSet01(object):

    def setup_method(self, method):
        N = 16
        Nd = 5
        M = 4
        K = 3
        cp.random.seed(12345)
        self.D0 = cp.random.randn(Nd, Nd, M)
        self.S = cp.random.randn(N, N, K)


    def test_01(self):
        lmbda = 1e-1
        opt = onlinecdl.OnlineConvBPDNDictLearn.Options()
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, lmbda, opt=opt)
            for it in range(5):
                img_index = cp.random.randint(0, self.S.shape[-1])
                b.solve(self.S[..., img_index])
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        lmbda = 1e-1
        opt = onlinecdl.OnlineConvBPDNDictLearn.Options()
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, lmbda, opt=opt)
            for it in range(5):
                img_index = int(cp.random.randint(0, self.S.shape[-1]))
                b.solve(self.S[..., [img_index]])
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        lmbda = 1e-1
        W = cp.random.randn(*self.S.shape[0:2])
        opt = onlinecdl.OnlineConvBPDNMaskDictLearn.Options()
        try:
            b = onlinecdl.OnlineConvBPDNMaskDictLearn(self.D0, lmbda, opt=opt)
            for it in range(5):
                img_index = cp.random.randint(0, self.S.shape[-1])
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
        cp.random.seed(12345)
        self.D0 = cp.random.randn(Nd, Nd, Nc, M)
        self.S = cp.random.randn(N, N, Nc, K)


    def test_01(self):
        lmbda = 1e-1
        opt = onlinecdl.OnlineConvBPDNDictLearn.Options()
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, lmbda, opt=opt)
            for it in range(5):
                img_index = cp.random.randint(0, self.S.shape[-1])
                b.solve(self.S[..., img_index])
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        lmbda = 1e-1
        opts = onlinecdl.OnlineConvBPDNDictLearn.Options(
            {'CBPDN': {'MaxMainIter': 20, 'AutoRho': {'Enabled': False}}})
        try:
            b = onlinecdl.OnlineConvBPDNDictLearn(self.D0, lmbda, opt=opts)
            for it in range(5):
                img_index = int(cp.random.randint(0, self.S.shape[-1]))
                b.solve(self.S[..., [img_index]])
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        lmbda = 1e-1
        W = cp.random.randn(*self.S.shape[0:3])
        opt = onlinecdl.OnlineConvBPDNMaskDictLearn.Options()
        try:
            b = onlinecdl.OnlineConvBPDNMaskDictLearn(self.D0, lmbda, opt=opt)
            for it in range(5):
                img_index = cp.random.randint(0, self.S.shape[-1])
                b.solve(self.S[..., img_index], W)
        except Exception as e:
            print(e)
            assert 0
