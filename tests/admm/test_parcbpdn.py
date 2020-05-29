from __future__ import division
from builtins import object

import pickle
import numpy as np

from sporco.admm import parcbpdn
from sporco.fft import fftn, ifftn
import sporco.linalg as sl


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, Cs)
        lmbda = 1e-1
        b = parcbpdn.ParConvBPDN(D, s, lmbda, dimK=0)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 0


    def test_02(self):
        N = 16
        Nd = 5
        Cs = 3
        K = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, Cs, K)
        lmbda = 1e-1
        b = parcbpdn.ParConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 1


    def test_03(self):
        N = 16
        Nd = 5
        Cd = 3
        M = 4
        D = np.random.randn(Nd, Nd, Cd, M)
        s = np.random.randn(N, N, Cd)
        lmbda = 1e-1
        b = parcbpdn.ParConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 0


    def test_04(self):
        N = 16
        Nd = 5
        Cd = 3
        K = 5
        M = 4
        D = np.random.randn(Nd, Nd, Cd, M)
        s = np.random.randn(N, N, Cd, K)
        lmbda = 1e-1
        b = parcbpdn.ParConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 1


    def test_05(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        lmbda = 1e-1
        b = parcbpdn.ParConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 0
        assert b.cri.dimK == 1


    def test_06(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        dt = np.float32
        opt = parcbpdn.ParConvBPDN.Options({'Verbose': False,
                            'MaxMainIter': 20, 'AutoRho': {'Enabled':
                            True}, 'DataType': dt})
        lmbda = 1e-1
        b = parcbpdn.ParConvBPDN(D, s, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_07(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        dt = np.float64
        opt = parcbpdn.ParConvBPDN.Options({'Verbose': False,
                            'MaxMainIter': 20, 'AutoRho': {'Enabled':
                            True}, 'DataType': dt})
        lmbda = 1e-1
        b = parcbpdn.ParConvBPDN(D, s, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_08(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        try:
            b = parcbpdn.ParConvBPDN(D, s)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_09(self):
        N = 64
        M = 4
        Nd = 8
        D = np.random.randn(Nd, Nd, M)
        X0 = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X0[xp] = np.random.randn(X0[xp].size)
        S = np.sum(ifftn(fftn(D, (N, N), (0, 1)) *
                   fftn(X0, None, (0, 1)), None, (0, 1)).real, axis=2)
        lmbda = 1e-4
        rho = 3e-3
        alpha = 6
        opt = parcbpdn.ParConvBPDN.Options({'Verbose': False,
                        'MaxMainIter': 1000, 'RelStopTol': 1e-3, 'rho': rho,
                        'alpha': alpha, 'AutoRho': {'Enabled': False}})
        b = parcbpdn.ParConvBPDN(D, S, lmbda, opt=opt)
        b.solve()
        X1 = b.Y.squeeze()
        assert sl.rrs(X0, X1) < 5e-5
        Sr = b.reconstruct().squeeze()
        assert sl.rrs(S, Sr) < 1e-4


    def test_10(self):
        N = 63
        M = 4
        Nd = 8
        D = np.random.randn(Nd, Nd, M)
        X0 = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X0[xp] = np.random.randn(X0[xp].size)
        S = np.sum(ifftn(fftn(D, (N, N), (0, 1)) *
                   fftn(X0, None, (0, 1)), None, (0, 1)).real, axis=2)
        lmbda = 1e-4
        alpha = 6
        rho = 3e-3
        opt = parcbpdn.ParConvBPDN.Options({'Verbose': False,
                            'MaxMainIter': 1000, 'RelStopTol': 1e-3,
                            'rho': rho, 'alpha': alpha, 'AutoRho':
                            {'Enabled': False}})
        b = parcbpdn.ParConvBPDN(D, S, lmbda, opt=opt)
        b.solve()
        X1 = b.Y.squeeze()
        assert sl.rrs(X0, X1) < 5e-5
        Sr = b.reconstruct().squeeze()
        assert sl.rrs(S, Sr) < 1e-4


    def test_11(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        lmbda = 1e-1
        try:
            b = parcbpdn.ParConvBPDN(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_12(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, Cs)
        lmbda = 1e-1
        try:
            b = parcbpdn.ParConvBPDN(D, s, lmbda, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_13(self):
        N = 16
        Nd = 5
        Cs = 3
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, Cs, K)
        lmbda = 1e-1
        try:
            b = parcbpdn.ParConvBPDN(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_14(self):
        N = 16
        Nd = 5
        Cd = 3
        M = 4
        D = np.random.randn(Nd, Nd, Cd, M)
        s = np.random.randn(N, N, Cd)
        lmbda = 1e-1
        try:
            b = parcbpdn.ParConvBPDN(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_15(self):
        N = 16
        Nd = 5
        Cd = 3
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, Cd, M)
        s = np.random.randn(N, N, Cd, K)
        lmbda = 1e-1
        try:
            b = parcbpdn.ParConvBPDN(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_16(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        dt = np.float32
        opt = parcbpdn.ParConvBPDN.Options({'Verbose': False,
                        'LinSolveCheck': True, 'MaxMainIter': 20,
                        'AutoRho': {'Enabled': True}, 'DataType': dt})
        lmbda = 1e-1
        b = parcbpdn.ParConvBPDN(D, s, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_17(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        w = np.ones(s.shape)
        lmbda = 1e-1
        try:
            b = parcbpdn.ParConvBPDN(D, s, lmbda, W=w)
            b.solve()
            b.reconstruct()
        except Exception as e:
            print(e)
            assert 0


    def test_18(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        w = np.ones(s.shape)
        dt = np.float32
        opt = parcbpdn.ParConvBPDN.Options({'Verbose': False,
                        'MaxMainIter': 20, 'AutoRho': {'Enabled':
                        True}, 'DataType': dt})
        lmbda = 1e-1
        b = parcbpdn.ParConvBPDN(D, s, lmbda, W=w, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt



    def test_19(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        lmbda = 1e-1
        opt = parcbpdn.ParConvBPDN.Options({'Verbose': False,
                                            'MaxMainIter': 10})
        b = parcbpdn.ParConvBPDN(D, s, lmbda, opt=opt)
        bp = pickle.dumps(b)
        c = pickle.loads(bp)
        Xb = b.solve()
        Xc = c.solve()
        assert np.linalg.norm(Xb - Xc) == 0.0
