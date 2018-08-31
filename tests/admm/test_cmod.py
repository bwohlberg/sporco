from __future__ import division
from builtins import object

import numpy as np

from sporco.admm import cmod



class TestSet01(object):

    def setup_method(self, method):
        pass


    def test_01(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        try:
            b = cmod.CnstrMOD(X, S, (N, M))
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        try:
            b = cmod.CnstrMOD(X, S)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        dt = np.float16
        opt = cmod.CnstrMOD.Options({'Verbose': False, 'MaxMainIter': 20,
                                     'AutoRho': {'Enabled': True},
                                     'DataType': dt})
        b = cmod.CnstrMOD(X, S, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_04(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        dt = np.float32
        opt = cmod.CnstrMOD.Options({'Verbose': False, 'MaxMainIter': 20,
                                     'AutoRho': {'Enabled': True},
                                     'DataType': dt})
        b = cmod.CnstrMOD(X, S, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_05(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        dt = np.float64
        opt = cmod.CnstrMOD.Options({'Verbose': False, 'MaxMainIter': 20,
                                     'AutoRho': {'Enabled': True},
                                     'DataType': dt})
        b = cmod.CnstrMOD(X, S, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_06(self):
        opt = cmod.CnstrMOD.Options({'AuxVarObj': False})
        assert opt['fEvalX'] is True and opt['gEvalY'] is False
        opt['AuxVarObj'] = True
        assert opt['fEvalX'] is False and opt['gEvalY'] is True


    def test_07(self):
        opt = cmod.CnstrMOD.Options({'AuxVarObj': True})
        assert opt['fEvalX'] is False and opt['gEvalY'] is True
        opt['AuxVarObj'] = False
        assert opt['fEvalX'] is True and opt['gEvalY'] is False
