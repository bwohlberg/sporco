from builtins import object

import pytest
import numpy as np

from sporco.fista import ccmod
import sporco.linalg as sl
import sporco.cnvrep as cr



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)
    
    def test_01(self):
        N = 32
        M = 4
        Nd = 5
        D0 = cr.normalise(cr.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(sl.ifftn(sl.fftn(D0, (N, N), (0,1)) *
                   sl.fftn(X, None, (0,1)), None, (0,1)).real, axis=2)
        L = 3e1
        opt = ccmod.ConvCnstrMOD.Options({'Verbose': False,
                    'MaxMainIter': 3000,
                    'ZeroMean': True, 'RelStopTol': 1e-6, 'L': L,
                    'BackTrack': {'Enabled': False}})
        Xr = X.reshape(X.shape[0:2] + (1,1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.X, D0.shape).squeeze()
        assert(sl.rrs(D0, D1) < 1e-4)
        assert(np.array(c.getitstat().Rsdl)[-1] < 1e-5)


    def test_02(self):
        N = 32
        M = 4
        Nd = 5
        D0 = cr.normalise(cr.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(sl.ifftn(sl.fftn(D0, (N, N), (0,1)) *
                   sl.fftn(X, None, (0,1)), None, (0,1)).real, axis=2)
        L = 1e1
        opt = ccmod.ConvCnstrMOD.Options({'Verbose': False,
                    'MaxMainIter': 3000,
                    'ZeroMean': True, 'RelStopTol': 1e-6, 'L': L,
                    'BackTrack': {'Enabled': True}})
        Xr = X.reshape(X.shape[0:2] + (1,1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.X, D0.shape).squeeze()
        assert(sl.rrs(D0, D1) < 1e-4)
        assert(np.array(c.getitstat().Rsdl)[-1] < 1e-5)


    def test_03(self):
        N = 64
        M = 4
        Nd = 8
        D0 = cr.normalise(cr.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(sl.ifftn(sl.fftn(D0, (N, N), (0,1)) *
                   sl.fftn(X, None, (0,1)), None, (0,1)).real, axis=2)
        L = 1e1
        opt = ccmod.ConvCnstrMOD.Options({'Verbose': False,
                    'MaxMainIter': 3000,
                    'ZeroMean': True, 'RelStopTol': 1e-6, 'L': L,
                    'BackTrack': {'Enabled': True}})
        Xr = X.reshape(X.shape[0:2] + (1,1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.X, D0.shape).squeeze()
        assert(sl.rrs(D0,D1) < 1e-4)
        assert(np.array(c.getitstat().Rsdl)[-1] < 1e-5)


    def test_04(self):
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
            assert(0)


    def test_05(self):
        N = 16
        M = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        try:
            c = ccmod.ConvCnstrMOD(X, S, ((4, 4, 4),(8, 8, 4)))
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_06(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        L = 2e3
        try:
            opt = ccmod.ConvCnstrMODOptions({'Verbose': False,
                            'MaxMainIter': 100, 'L' : L})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)
        assert(np.array(c.getitstat().Rsdl)[-1] < 5e-3)


    def test_07(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        dt = np.float32
        opt = ccmod.ConvCnstrMODOptions(
            {'Verbose': False, 'MaxMainIter': 20,
             'BackTrack': {'Enabled': True},
             'DataType': dt})
        c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, M), opt=opt)
        c.solve()
        assert(c.X.dtype == dt)
        assert(c.Xf.dtype == sl.complex_dtype(dt))
        assert(c.Yf.dtype == sl.complex_dtype(dt))


    def test_08(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        dt = np.float64
        opt = ccmod.ConvCnstrMODOptions(
            {'Verbose': False, 'MaxMainIter': 20,
             'BackTrack': {'Enabled': True},
             'DataType': dt})
        c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, M), opt=opt)
        c.solve()
        assert(c.X.dtype == dt)
        assert(c.Xf.dtype == sl.complex_dtype(dt))
        assert(c.Yf.dtype == sl.complex_dtype(dt))


    def test_09(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N)
        try:
            opt = ccmod.ConvCnstrMODOptions(
            {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, 1, M),
                opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_10(self):
        N = 16
        K = 3
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, K, M)
        S = np.random.randn(N, N, K)
        try:
            opt = ccmod.ConvCnstrMODOptions(
            {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, 1, M),
                opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_11(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        try:
            opt = ccmod.ConvCnstrMODOptions(
            {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, 1, M),
                opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_12(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        try:
            opt = ccmod.ConvCnstrMODOptions(
            {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, 1, M),
                opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_13(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        try:
            opt = ccmod.ConvCnstrMODOptions(
            {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, Nc, M),
                opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_14(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N)
        W = np.array([1.0])
        try:
            opt = ccmod.ConvCnstrMODMaskDcpl.Options(
                           {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMaskDcpl(X, S, W,
                    (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_15(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N)
        W = np.random.randn(N, N)
        try:
            opt = ccmod.ConvCnstrMODMaskDcpl.Options(
                           {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMaskDcpl(X, S, W,
                    (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_16(self):
        N = 16
        K = 3
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, K, M)
        S = np.random.randn(N, N, K)
        W = np.random.randn(N, N)
        try:
            opt = ccmod.ConvCnstrMODMaskDcpl.Options(
                           {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMaskDcpl(X, S, W,
                    (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_17(self):
        N = 16
        K = 3
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, K, M)
        S = np.random.randn(N, N, K)
        W = np.random.randn(N, N, K)
        try:
            opt = ccmod.ConvCnstrMODMaskDcpl.Options(
                           {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMaskDcpl(X, S, W,
                    (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_18(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        W = np.random.randn(N, N, Nc)
        try:
            opt = ccmod.ConvCnstrMODMaskDcpl.Options(
                           {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMaskDcpl(X, S, W,
                    (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_19(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, Nc, K)
        try:
            opt = ccmod.ConvCnstrMODMaskDcpl.Options(
                           {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMaskDcpl(X, S, W,
                    (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_20(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, Nc)
        try:
            opt = ccmod.ConvCnstrMODMaskDcpl.Options(
                           {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMaskDcpl(X, S, W,
                    (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_21(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, 1, K)
        try:
            opt = ccmod.ConvCnstrMODMaskDcpl.Options(
                           {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMaskDcpl(X, S, W,
                    (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_22(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, Nc, K)
        try:
            opt = ccmod.ConvCnstrMODMaskDcpl.Options(
                           {'Verbose': False, 'MaxMainIter': 20})
            c = ccmod.ConvCnstrMODMaskDcpl(X, S, W,
                    (Nd, Nd, Nc, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_23(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        W = np.random.randn(N, N)
        L = 5e3
        try:
            opt = ccmod.ConvCnstrMODMaskDcpl.Options({'Verbose': False,
                            'MaxMainIter': 200, 'L' : L})
            c = ccmod.ConvCnstrMODMaskDcpl(X, S, W, (Nd, Nd, 1, M),
                                             opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)
        assert(np.array(c.getitstat().Rsdl)[-1] < 5e-3)


