from builtins import object

import numpy as np

import sporco.cnvrep as cr
from sporco.admm import ccmod
from sporco.linalg import rrs
from sporco.fft import fftconv



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 32
        M = 4
        Nd = 5
        self.D0 = cr.normalise(cr.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        self.X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        self.X[xp] = np.random.randn(self.X[xp].size)
        self.S = np.sum(fftconv(self.X, self.D0, axes=(0, 1)).real, axis=2)
        d0c = np.random.randn(Nd, Nd, M) + 1j * np.random.randn(Nd, Nd, M)
        self.D0c = cr.normalise(cr.zeromean(d0c, (Nd, Nd, M), dimN=2), dimN=2)
        self.Xc = np.zeros((N, N, M)) + 1j * np.zeros((N, N, M))
        self.Xc[xp] = (np.random.randn(self.Xc[xp].size) +
                      1j * np.random.randn(self.Xc[xp].size))
        self.Sc = np.sum(fftconv(self.Xc, self.D0c, axes=(0, 1)), axis=2)


    def test_01(self):
        rho = 1e-1
        opt = ccmod.ConvCnstrMOD_IterSM.Options({'Verbose': False,
                    'MaxMainIter': 500, 'LinSolveCheck': True,
                    'ZeroMean': True, 'RelStopTol': 1e-5, 'rho': rho,
                    'AutoRho': {'Enabled': False}})
        Xr = self.X.reshape(self.X.shape[0:2] + (1, 1,) + self.X.shape[2:])
        Sr = self.S.reshape(self.S.shape + (1,))
        c = ccmod.ConvCnstrMOD_IterSM(Xr, Sr, self.D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.Y, self.D0.shape).squeeze()
        assert rrs(self.D0, D1) < 1e-5
        assert np.array(c.getitstat().XSlvRelRes).max() < 1e-5


    def test_01cplx(self):
        rho = 1e-1
        opt = ccmod.ConvCnstrMOD_IterSM.Options({'Verbose': False,
                    'MaxMainIter': 500, 'LinSolveCheck': True,
                    'ZeroMean': True, 'RelStopTol': 1e-5, 'rho': rho,
                    'AutoRho': {'Enabled': False}})
        Xr = self.Xc.reshape(self.Xc.shape[0:2] + (1, 1,) + self.Xc.shape[2:])
        Sr = self.Sc.reshape(self.Sc.shape + (1,))
        c = ccmod.ConvCnstrMOD_IterSM(Xr, Sr, self.D0c.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.Y, self.D0c.shape).squeeze()
        assert rrs(self.D0c, D1) < 1e-4
        assert np.array(c.getitstat().XSlvRelRes).max() < 1e-5


    def test_02(self):
        rho = 1e-1
        opt = ccmod.ConvCnstrMOD_CG.Options({'Verbose': False,
                    'MaxMainIter': 500, 'LinSolveCheck': True,
                    'ZeroMean': True, 'RelStopTol': 1e-5, 'rho': rho,
                    'AutoRho': {'Enabled': False},
                    'CG': {'StopTol': 1e-5}})
        Xr = self.X.reshape(self.X.shape[0:2] + (1, 1,) + self.X.shape[2:])
        Sr = self.S.reshape(self.S.shape + (1,))
        c = ccmod.ConvCnstrMOD_CG(Xr, Sr, self.D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.Y, self.D0.shape).squeeze()
        assert rrs(self.D0, D1) < 1e-4
        assert np.array(c.getitstat().XSlvRelRes).max() < 1e-3


    def test_02cplx(self):
        rho = 1e-1
        opt = ccmod.ConvCnstrMOD_CG.Options({'Verbose': False,
                    'MaxMainIter': 500, 'LinSolveCheck': True,
                    'ZeroMean': True, 'RelStopTol': 1e-5, 'rho': rho,
                    'AutoRho': {'Enabled': False},
                    'CG': {'StopTol': 1e-5}})
        Xr = self.Xc.reshape(self.Xc.shape[0:2] + (1, 1,) + self.Xc.shape[2:])
        Sr = self.Sc.reshape(self.Sc.shape + (1,))
        c = ccmod.ConvCnstrMOD_CG(Xr, Sr, self.D0c.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.Y, self.D0c.shape).squeeze()
        assert rrs(self.D0c, D1) < 1e-3
        assert np.array(c.getitstat().XSlvRelRes).max() < 1e-3


    def test_03(self):
        rho = 1e-1
        opt = ccmod.ConvCnstrMOD_Consensus.Options({'Verbose': False,
                    'MaxMainIter': 500, 'LinSolveCheck': True,
                    'ZeroMean': True, 'RelStopTol': 1e-4, 'rho': rho,
                    'AutoRho': {'Enabled': False}})
        Xr = self.X.reshape(self.X.shape[0:2] + (1, 1,) + self.X.shape[2:])
        Sr = self.S.reshape(self.S.shape + (1,))
        c = ccmod.ConvCnstrMOD_Consensus(Xr, Sr, self.D0.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.Y, self.D0.shape).squeeze()
        assert rrs(self.D0, D1) < 1e-5
        assert np.array(c.getitstat().XSlvRelRes).max() < 1e-5


    def test_03cplx(self):
        rho = 1e-1
        opt = ccmod.ConvCnstrMOD_Consensus.Options({'Verbose': False,
                    'MaxMainIter': 500, 'LinSolveCheck': True,
                    'ZeroMean': True, 'RelStopTol': 1e-4, 'rho': rho,
                    'AutoRho': {'Enabled': False}})
        Xr = self.Xc.reshape(self.Xc.shape[0:2] + (1, 1,) + self.Xc.shape[2:])
        Sr = self.Sc.reshape(self.Sc.shape + (1,))
        c = ccmod.ConvCnstrMOD_Consensus(Xr, Sr, self.D0c.shape, opt)
        c.solve()
        D1 = cr.bcrop(c.Y, self.D0c.shape).squeeze()
        assert rrs(self.D0c, D1) < 1e-4
        assert np.array(c.getitstat().XSlvRelRes).max() < 1e-5


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
            assert 0


    def test_05(self):
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


    def test_06(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        try:
            opt = ccmod.ConvCnstrMODOptions({'Verbose': False,
                            'MaxMainIter': 20, 'LinSolveCheck': True})
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(c.getitstat().XSlvRelRes).max() < 1e-5


    def test_07(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        dt = np.float32
        opt = ccmod.ConvCnstrMODOptions(
            {'Verbose': False, 'MaxMainIter': 20,
             'AutoRho': {'Enabled': True},
             'DataType': dt})
        c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, M), opt=opt)
        c.solve()
        assert c.X.dtype == dt
        assert c.Y.dtype == dt
        assert c.U.dtype == dt


    def test_08(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        dt = np.float64
        opt = ccmod.ConvCnstrMODOptions(
            {'Verbose': False, 'MaxMainIter': 20,
             'AutoRho': {'Enabled': True},
             'DataType': dt})
        c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, M), opt=opt)
        c.solve()
        assert c.X.dtype == dt
        assert c.Y.dtype == dt
        assert c.U.dtype == dt


    def test_09(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N)
        try:
            opt = ccmod.ConvCnstrMOD_Consensus.Options({'Verbose': False,
                            'MaxMainIter': 20, 'LinSolveCheck': True})
            c = ccmod.ConvCnstrMOD_Consensus(X, S, (Nd, Nd, 1, M),
                                             opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_10(self):
        N = 16
        K = 3
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, K, M)
        S = np.random.randn(N, N, K)
        try:
            opt = ccmod.ConvCnstrMOD_Consensus.Options({'Verbose': False,
                            'MaxMainIter': 20, 'LinSolveCheck': True})
            c = ccmod.ConvCnstrMOD_Consensus(X, S, (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_11(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        try:
            opt = ccmod.ConvCnstrMOD_Consensus.Options({'Verbose': False,
                            'MaxMainIter': 20, 'LinSolveCheck': True})
            c = ccmod.ConvCnstrMOD_Consensus(X, S, (Nd, Nd, 1, M),
                                             opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_12(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        try:
            opt = ccmod.ConvCnstrMOD_Consensus.Options({'Verbose': False,
                            'MaxMainIter': 20, 'LinSolveCheck': True})
            c = ccmod.ConvCnstrMOD_Consensus(X, S, (Nd, Nd, 1, M),
                                             opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_13(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        try:
            opt = ccmod.ConvCnstrMOD_Consensus.Options({'Verbose': False,
                            'MaxMainIter': 20, 'LinSolveCheck': True})
            c = ccmod.ConvCnstrMOD_Consensus(X, S, (Nd, Nd, Nc, M),
                                             opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_14(self):
        opt = ccmod.ConvCnstrMODBase.Options({'AuxVarObj': False})
        assert opt['fEvalX'] is True and opt['gEvalY'] is False
        opt['AuxVarObj'] = True
        assert opt['fEvalX'] is False and opt['gEvalY'] is True


    def test_15(self):
        opt = ccmod.ConvCnstrMODBase.Options({'AuxVarObj': True})
        assert opt['fEvalX'] is False and opt['gEvalY'] is True
        opt['AuxVarObj'] = False
        assert opt['fEvalX'] is True and opt['gEvalY'] is False
