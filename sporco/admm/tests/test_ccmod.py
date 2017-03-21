from builtins import object

import pytest
import numpy as np

from sporco.admm import ccmod
import sporco.linalg as sl



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 64
        M = 4
        Nd = 8
        D0 = ccmod.normalise(ccmod.zeromean(
            np.random.randn(Nd, Nd, M), (Nd, Nd, M), dimN=2), dimN=2)
        X = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X[xp] = np.random.randn(X[xp].size)
        S = np.sum(sl.ifftn(sl.fftn(D0, (N, N), (0,1)) *
                   sl.fftn(X, None, (0,1)), None, (0,1)).real, axis=2)
        rho = 1e1
        opt = ccmod.ConvCnstrMOD.Options({'Verbose' : False,
                                          'MaxMainIter' : 500,
                                          'LinSolveCheck' : True,
                                          'ZeroMean' : True,
                                          'RelStopTol' : 1e-3, 'rho' : rho,
                                          'AutoRho' : {'Enabled' : False}})
        Xr = X.reshape(X.shape[0:2] + (1,1,) + X.shape[2:])
        Sr = S.reshape(S.shape + (1,))
        c = ccmod.ConvCnstrMOD(Xr, Sr, D0.shape, opt)
        c.solve()
        D1 = ccmod.bcrop(c.Y, D0.shape).squeeze()
        assert(sl.rrs(D0,D1) < 1e-5)


    def test_02(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        opt = ccmod.ConvCnstrMOD.Options({'ZeroMean' : True})
        try:
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, M))
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_03(self):
        N = 16
        M = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        opt = ccmod.ConvCnstrMOD.Options({'ZeroMean' : True})
        try:
            c = ccmod.ConvCnstrMOD(X, S, ((4, 4, 4),(8, 8, 4)), opt)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_04(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        opt = ccmod.ConvCnstrMOD.Options({'ZeroMean' : True})
        try:
            c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, 1, M), opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_05(self):
        N = 16
        X = np.random.randn(N, N, 3, 1, 10)
        S = np.random.randn(N, N, 3)
        dsz = (
            (
                (3, 3, 1, 6),
                (4, 4, 1, 6),
                (5, 5, 1, 6)
            ),
            (6, 6, 3, 4)
        )
        opt = ccmod.ConvCnstrMOD.Options({'ZeroMean' : True})
        try:
            c = ccmod.ConvCnstrMOD(X, S, dsz, opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_06(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        dt = np.float32
        opt = ccmod.ConvCnstrMOD.Options(
            {'Verbose' : False, 'MaxMainIter' : 20,
             'AutoRho' : {'Enabled' : True},
             'DataType' : dt})
        c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, M), opt=opt)
        c.solve()
        assert(c.X.dtype == dt)
        assert(c.Y.dtype == dt)
        assert(c.U.dtype == dt)


    def test_07(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N, 1)
        dt = np.float64
        opt = ccmod.ConvCnstrMOD.Options(
            {'Verbose' : False, 'MaxMainIter' : 20,
             'AutoRho' : {'Enabled' : True},
             'DataType' : dt})
        c = ccmod.ConvCnstrMOD(X, S, (Nd, Nd, M), opt=opt)
        c.solve()
        assert(c.X.dtype == dt)
        assert(c.Y.dtype == dt)
        assert(c.U.dtype == dt)


    def test_09(self):
        opt = ccmod.ConvCnstrMOD.Options({'AuxVarObj' : False})
        assert(opt['fEvalX'] is True and opt['gEvalY'] is False)
        opt['AuxVarObj'] = True
        assert(opt['fEvalX'] is False and opt['gEvalY'] is True)


    def test_10(self):
        opt = ccmod.ConvCnstrMOD.Options({'AuxVarObj' : True})
        assert(opt['fEvalX'] is False and opt['gEvalY'] is True)
        opt['AuxVarObj'] = False
        assert(opt['fEvalX'] is True and opt['gEvalY'] is False)
