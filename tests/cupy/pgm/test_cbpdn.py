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


from sporco.cupy.pgm import cbpdn
from sporco.cupy.pgm.momentum import MomentumLinear, MomentumGenLinear
from sporco.cupy.pgm.stepsize import StepSizePolicyBB, StepSizePolicyCauchy
from sporco.cupy.pgm.backtrack import BacktrackStandard
from sporco.cupy.array import list2array
from sporco.cupy.fft import complex_dtype, fftn, ifftn
from sporco.cupy.linalg import rrs


class TestSet01(object):

    def setup_method(self, method):
        cp.random.seed(12345)


    def test_01(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda, dimK=0)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 0


    def test_02(self):
        N = 16
        Nd = 5
        Cs = 3
        K = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs, K)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 1


    def test_03(self):
        N = 16
        Nd = 5
        Cd = 3
        M = 4
        D = cp.random.randn(Nd, Nd, Cd, M)
        s = cp.random.randn(N, N, Cd)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 0


    def test_04(self):
        N = 16
        Nd = 5
        Cd = 3
        K = 5
        M = 4
        D = cp.random.randn(Nd, Nd, Cd, M)
        s = cp.random.randn(N, N, Cd, K)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 1


    def test_05(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, K)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 0
        assert b.cri.dimK == 1


    def test_06(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, K)
        dt = cp.float32
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 20,
                                      'Backtrack': BacktrackStandard(),
                                      'DataType': dt})
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Xf.dtype == complex_dtype(dt)
        assert b.Yf.dtype == complex_dtype(dt)


    def test_07(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, K)
        dt = cp.float64
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 20,
                                      'Backtrack': BacktrackStandard(),
                                      'DataType': dt})
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Xf.dtype == complex_dtype(dt)
        assert b.Yf.dtype == complex_dtype(dt)


    def test_08(self):
        N = 16
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N)
        try:
            b = cbpdn.ConvBPDN(D, s)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_09(self):
        N = 16
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N)
        try:
            b = cbpdn.ConvBPDN(D, s)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_10(self):
        N = 64
        M = 4
        Nd = 8
        D = cp.random.randn(Nd, Nd, M)
        X0 = cp.zeros((N, N, M))
        xr = cp.random.randn(N, N, M)
        xp = cp.abs(xr) > 3
        X0[xp] = cp.random.randn(X0[xp].size)
        S = cp.sum(ifftn(fftn(D, (N, N), (0, 1)) *
                            fftn(X0, None, (0, 1)), None, (0, 1)).real,
                   axis=2)
        lmbda = 1e-2
        L = 1e3
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 2000,
                                      'RelStopTol': 1e-9, 'L': L,
                                      'Backtrack': BacktrackStandard()})
        b = cbpdn.ConvBPDN(D, S, lmbda, opt)
        b.solve()
        X1 = b.X.squeeze()
        assert rrs(X0, X1) < 5e-4
        Sr = b.reconstruct().squeeze()
        assert rrs(S, Sr) < 3e-4


    def test_11(self):
        N = 63
        M = 4
        Nd = 8
        D = cp.random.randn(Nd, Nd, M)
        X0 = cp.zeros((N, N, M))
        xr = cp.random.randn(N, N, M)
        xp = cp.abs(xr) > 3
        X0[xp] = cp.random.randn(X0[xp].size)
        S = cp.sum(ifftn(fftn(D, (N, N), (0, 1)) *
                         fftn(X0, None, (0, 1)), None, (0, 1)).real,
                   axis=2)
        lmbda = 1e-2
        L = 1e3
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 2000,
                                      'RelStopTol': 1e-9, 'L': L,
                                      'Backtrack': BacktrackStandard()})
        b = cbpdn.ConvBPDN(D, S, lmbda, opt)
        b.solve()
        X1 = b.X.squeeze()
        assert rrs(X0, X1) < 5e-4
        Sr = b.reconstruct().squeeze()
        assert rrs(S, Sr) < 2e-4


    def test_12(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs)
        lmbda = 1e-1
        L = 1e3
        opt = cbpdn.ConvBPDN.Options({'L': L})
        b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt, dimK=0)
        b.solve()
        assert list2array(b.getitstat().Rsdl)[-1] < 1e-3


    def test_13(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs)
        lmbda = 1e-1
        L = 1e3
        try:
            opt = cbpdn.ConvBPDN.Options({'L': L})
            b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert list2array(b.getitstat().Rsdl)[-1] < 1e-3


    def test_14(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs)
        lmbda = 1e-1
        L = 1e3
        try:
            opt = cbpdn.ConvBPDN.Options(
                {'Momentum': MomentumLinear(), 'L': L})
            b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert list2array(b.getitstat().Rsdl)[-1] < 1e-3


    def test_15(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs)
        lmbda = 1e-1
        L = 1e3
        try:
            opt = cbpdn.ConvBPDN.Options(
                {'Momentum': MomentumGenLinear(), 'L': L})
            b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert list2array(b.getitstat().Rsdl)[-1] < 1e-3


    def test_16(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs)
        lmbda = 1e-1
        L = 1e3
        try:
            opt = cbpdn.ConvBPDN.Options(
                {'StepSizePolicy': StepSizePolicyCauchy(), 'L': L})
            b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert list2array(b.getitstat().Rsdl)[-1] < 1e-3


    def test_17(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs)
        lmbda = 1e-1
        L = 1e3
        try:
            opt = cbpdn.ConvBPDN.Options(
                {'StepSizePolicy': StepSizePolicyBB(), 'L': L})
            b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert list2array(b.getitstat().Rsdl)[-1] < 1e-3


    def test_18(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs)
        lmbda = 1e-1
        L = 1e3
        try:
            opt = cbpdn.ConvBPDN.Options({'Monotone': True, 'L': L})
            b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert list2array(b.getitstat().Rsdl)[-1] < 1e-3


    def test_19(self):
        N = 16
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N)
        lmbda = 1e-1
        try:
            b = cbpdn.ConvBPDNMask(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_20(self):
        N = 16
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N)
        w = cp.ones(s.shape)
        lmbda = 1e-1
        try:
            b = cbpdn.ConvBPDNMask(D, s, lmbda, w)
            b.solve()
            b.reconstruct()
        except Exception as e:
            print(e)
            assert 0
