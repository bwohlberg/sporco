from builtins import object

import numpy as np

from sporco.pgm import ccmod
from sporco.fft import complex_dtype, fftn, ifftn
from sporco.linalg import rrs
import sporco.cnvrep as cr

from sporco.pgm.momentum import MomentumLinear, MomentumGenLinear
from sporco.pgm.stepsize import StepSizePolicyBB, StepSizePolicyCauchy
from sporco.pgm.backtrack import BacktrackStandard, BacktrackRobust


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        try:
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, M))
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        N = 16
        M = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        try:
            c = ccmod.ConvCnstrMOD(X, S, ((4, 4, 4), (8, 8, 4)))
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        L = 2e3
        try:
            opt = ccmod.ConvCnstrMOD.Options({'Verbose': False,
                                              'MaxMainIter': 100, 'L': L})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(c.getitstat().Rsdl)[-1] < 5e-3


    def test_04(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        dt = np.float32
        opt = ccmod.ConvCnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 20,
             'Backtrack': BacktrackStandard(),
             'DataType': dt})
        c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, M), opt=opt)
        c.solve()
        assert c.X.dtype == dt
        assert c.Xf.dtype == complex_dtype(dt)
        assert c.Yf.dtype == complex_dtype(dt)


    def test_05(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        dt = np.float64
        opt = ccmod.ConvCnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 20,
             'Backtrack': BacktrackStandard(),
             'DataType': dt})
        c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, M), opt=opt)
        c.solve()
        assert c.X.dtype == dt
        assert c.Xf.dtype == complex_dtype(dt)
        assert c.Yf.dtype == complex_dtype(dt)


    def test_06(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N)
        try:
            opt = ccmod.ConvCnstrMOD.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, 1, M),
                                   opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_07(self):
        N = 16
        K = 3
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, K, M)
        S = np.random.randn(N, N, K)
        try:
            opt = ccmod.ConvCnstrMOD.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, 1, M),
                                   opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_08(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        try:
            opt = ccmod.ConvCnstrMOD.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, 1, M),
                                   opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_09(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        try:
            opt = ccmod.ConvCnstrMOD.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, 1, M),
                                   opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_10(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        try:
            opt = ccmod.ConvCnstrMOD.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, Nc, M),
                                   opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_11(self):
        N = 32
        M = 4
        Nd = 5
        D0 = cr.normalise(cr.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(ifftn(fftn(D0, (N, N), (0, 1)) * fftn(
            X, None, (0, 1)), None, (0, 1)).real, axis=2)
        L = 2.5
        opt = ccmod.ConvCnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 3000, 'ZeroMean': True,
             'RelStopTol': 0., 'L': L, 'Backtrack': BacktrackStandard()})
        Xr = X.reshape(X.shape[0:2] + (1, 1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.X, D0.shape).squeeze()
        assert rrs(D0, D1) < 1e-4
        assert np.array(c.getitstat().Rsdl)[-1] < 1e-5


    def test_12(self):
        N = 32
        M = 4
        Nd = 5
        D0 = cr.normalise(cr.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(ifftn(fftn(D0, (N, N), (0, 1)) * fftn(
            X, None, (0, 1)), None, (0, 1)).real, axis=2)
        L = 0.5
        opt = ccmod.ConvCnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 3000, 'ZeroMean': True,
             'RelStopTol': 0, 'L': L, 'Backtrack': BacktrackStandard()})
        Xr = X.reshape(X.shape[0:2] + (1, 1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.X, D0.shape).squeeze()
        assert rrs(D0, D1) < 1e-4
        assert np.array(c.getitstat().Rsdl)[-1] < 1e-5


    def test_13(self):
        N = 64
        M = 4
        Nd = 8
        D0 = cr.normalise(cr.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(ifftn(fftn(D0, (N, N), (0, 1)) * fftn(
            X, None, (0, 1)), None, (0, 1)).real, axis=2)
        L = 0.5
        opt = ccmod.ConvCnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 3000, 'ZeroMean': True,
             'RelStopTol': 0., 'L': L, 'Backtrack': BacktrackStandard()})
        Xr = X.reshape(X.shape[0:2] + (1, 1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.X, D0.shape).squeeze()

        assert rrs(D0, D1) < 1e-4
        assert np.array(c.getitstat().Rsdl)[-1] < 1e-5


    def test_14(self):
        N = 32
        M = 4
        Nd = 5
        D0 = cr.normalise(cr.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(ifftn(fftn(D0, (N, N), (0, 1)) * fftn(
            X, None, (0, 1)), None, (0, 1)).real, axis=2)
        L = 2.5
        opt = ccmod.ConvCnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 3000, 'ZeroMean': True,
             'RelStopTol': 0., 'L': L, 'Momentum': MomentumLinear()})
        Xr = X.reshape(X.shape[0:2] + (1, 1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.X, D0.shape).squeeze()
        assert rrs(D0, D1) < 1e-4
        assert np.array(c.getitstat().Rsdl)[-1] < 1e-5


    def test_15(self):
        N = 32
        M = 4
        Nd = 5
        D0 = cr.normalise(cr.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(ifftn(fftn(D0, (N, N), (0, 1)) * fftn(
            X, None, (0, 1)), None, (0, 1)).real, axis=2)
        L = 2.5
        opt = ccmod.ConvCnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 3000, 'ZeroMean': True,
             'RelStopTol': 0., 'L': L, 'Momentum': MomentumGenLinear()})
        Xr = X.reshape(X.shape[0:2] + (1, 1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.X, D0.shape).squeeze()
        assert rrs(D0, D1) < 1e-4
        assert np.array(c.getitstat().Rsdl)[-1] < 1e-5


    def test_16(self):
        N = 64
        M = 4
        Nd = 8
        D0 = cr.normalise(cr.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(ifftn(fftn(D0, (N, N), (0, 1)) * fftn(
            X, None, (0, 1)), None, (0, 1)).real, axis=2)
        L = 0.5
        opt = ccmod.ConvCnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 3000, 'ZeroMean': True,
             'RelStopTol': 0., 'L': L, 'StepSizePolicy': StepSizePolicyBB()})
        Xr = X.reshape(X.shape[0:2] + (1, 1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.X, D0.shape).squeeze()

        assert rrs(D0, D1) < 1e-4
        assert np.array(c.getitstat().Rsdl)[-1] < 1e-5


    def test_17(self):
        N = 64
        M = 4
        Nd = 8
        D0 = cr.normalise(cr.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(ifftn(fftn(D0, (N, N), (0, 1)) * fftn(
            X, None, (0, 1)), None, (0, 1)).real, axis=2)
        L = 0.5
        opt = ccmod.ConvCnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 3000,
             'ZeroMean': True, 'RelStopTol': 0., 'L': L,
             'StepSizePolicy': StepSizePolicyCauchy()})
        Xr = X.reshape(X.shape[0:2] + (1, 1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.X, D0.shape).squeeze()

        assert rrs(D0, D1) < 1e-4
        assert np.array(c.getitstat().Rsdl)[-1] < 1e-5


    def test_18(self):
        N = 64
        M = 4
        Nd = 8
        D0 = cr.normalise(cr.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(ifftn(fftn(D0, (N, N), (0, 1)) * fftn(
            X, None, (0, 1)), None, (0, 1)).real, axis=2)
        L = 50.0
        opt = ccmod.ConvCnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 3000, 'ZeroMean': True,
             'RelStopTol': 0., 'L': L, 'Monotone': True})
        Xr = X.reshape(X.shape[0:2] + (1, 1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.X, D0.shape).squeeze()

        assert rrs(D0, D1) < 1e-4
        assert np.array(c.getitstat().Rsdl)[-1] < 1e-5


    def test_19(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N)
        W = np.array([1.0])
        try:
            opt = ccmod.ConvCnstrMODMask.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMask(X, S, W, (Nd, Nd, 1, M), opt=opt,
                                       dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_20(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N)
        W = np.random.randn(N, N)
        try:
            opt = ccmod.ConvCnstrMODMask.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMask(X, S, W, (Nd, Nd, 1, M), opt=opt,
                                       dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_21(self):
        N = 16
        K = 3
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, K, M)
        S = np.random.randn(N, N, K)
        W = np.random.randn(N, N)
        try:
            opt = ccmod.ConvCnstrMODMask.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMask(X, S, W, (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_22(self):
        N = 16
        K = 3
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, K, M)
        S = np.random.randn(N, N, K)
        W = np.random.randn(N, N, K)
        try:
            opt = ccmod.ConvCnstrMODMask.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMask(X, S, W, (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_23(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        W = np.random.randn(N, N, Nc)
        try:
            opt = ccmod.ConvCnstrMODMask.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMask(X, S, W, (Nd, Nd, 1, M), opt=opt,
                                       dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_24(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, Nc, K)
        try:
            opt = ccmod.ConvCnstrMODMask.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMask(X, S, W, (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_25(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, Nc)
        try:
            opt = ccmod.ConvCnstrMODMask.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMask(X, S, W, (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_26(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, 1, K)
        try:
            opt = ccmod.ConvCnstrMODMask.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMask(X, S, W, (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_27(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, Nc, K)
        try:
            opt = ccmod.ConvCnstrMODMask.Options(
                {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMask(X, S, W, (Nd, Nd, Nc, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_28(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        W = np.random.randn(N, N)
        L = 5e3
        try:
            opt = ccmod.ConvCnstrMODMask.Options(
                {'Verbose': False, 'MaxMainIter': 200, 'L': L})
            c = ccmod.ConvCnstrMODMask(X, S, W, (Nd, Nd, 1, M),
                                       opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(c.getitstat().Rsdl)[-1] < 5e-3


    def test_29(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        W = np.random.randn(N, N)
        L = 5e1
        try:
            opt = ccmod.ConvCnstrMODMask.Options(
                {'Verbose': False, 'MaxMainIter': 200, 'L': L,
                 'Backtrack': BacktrackRobust()})
            c = ccmod.ConvCnstrMODMask(X, S, W, (Nd, Nd, 1, M),
                                       opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(c.getitstat().Rsdl)[-1] < 5e-3
