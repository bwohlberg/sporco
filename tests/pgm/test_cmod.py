from __future__ import division
from builtins import object

import numpy as np

from sporco.pgm import cmod

from sporco.pgm.momentum import MomentumLinear, MomentumGenLinear
from sporco.pgm.stepsize import StepSizePolicyBB, StepSizePolicyCauchy


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


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
        opt = cmod.CnstrMOD.Options({'Verbose': False, 'MaxMainIter': 200,
                                     'RelStopTol': 1e-4, 'L': 100.0})
        b = cmod.CnstrMOD(X, S, (N, M), opt=opt)
        b.solve()
        assert np.array(b.getitstat().Rsdl).min() < 1e-4


    def test_04(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        dt = np.float16
        opt = cmod.CnstrMOD.Options({'Verbose': False, 'MaxMainIter': 20,
                                     'DataType': dt})
        b = cmod.CnstrMOD(X, S, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt


    def test_05(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        dt = np.float32
        opt = cmod.CnstrMOD.Options({'Verbose': False, 'MaxMainIter': 20,
                                     'DataType': dt})
        b = cmod.CnstrMOD(X, S, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt


    def test_06(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        dt = np.float64
        opt = cmod.CnstrMOD.Options({'Verbose': False,
                                     'MaxMainIter': 20, 'DataType': dt})
        b = cmod.CnstrMOD(X, S, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt


    def test_07(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        opt = cmod.CnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 200, 'RelStopTol': 1e-4,
             'L': 100.0, 'Momentum': MomentumLinear()})
        b = cmod.CnstrMOD(X, S, (N, M), opt=opt)
        b.solve()
        assert np.array(b.getitstat().Rsdl).min() < 1e-4


    def test_08(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        opt = cmod.CnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 200, 'RelStopTol': 1e-4,
             'L': 100.0, 'Momentum': MomentumGenLinear()})
        b = cmod.CnstrMOD(X, S, (N, M), opt=opt)
        b.solve()
        assert np.array(b.getitstat().Rsdl).min() < 1e-4


    def test_09(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        opt = cmod.CnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 50, 'RelStopTol': 1e-4,
             'L': 100.0, 'StepSizePolicy': StepSizePolicyBB()})
        b = cmod.CnstrMOD(X, S, (N, M), opt=opt)
        b.solve()
        assert np.array(b.getitstat().Rsdl).min() < 1e-4


    def test_10(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        opt = cmod.CnstrMOD.Options({'Verbose': False,
                                     'MaxMainIter': 100,
                                     'RelStopTol': 1e-4,
                                     'L': 10.0,
                                     'StepSizePolicy': StepSizePolicyCauchy()})
        try:
            b = cmod.CnstrMOD(X, S, (N, M), opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_11(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        opt = cmod.CnstrMOD.Options({'Verbose': False,
                                     'MaxMainIter': 500,
                                     'RelStopTol': 1e-4,
                                     'L': 250.0,
                                     'Monotone': True})
        b = cmod.CnstrMOD(X, S, (N, M), opt=opt)
        b.solve()
        assert np.array(b.getitstat().Rsdl).min() < 2e-4


    def test_12(self):
        N = 16
        M = 4
        K = 8
        X = np.random.randn(M, K)
        S = np.random.randn(N, K)
        W = np.abs(np.random.randn(N, K))
        opt = cmod.WeightedCnstrMOD.Options(
            {'Verbose': False, 'MaxMainIter': 200, 'RelStopTol': 1e-4,
             'L': 100.0})
        b = cmod.WeightedCnstrMOD(X, S, W=W, dsz=(N, M), opt=opt)
        b.solve()
        assert np.array(b.getitstat().Rsdl).min() < 1e-4
