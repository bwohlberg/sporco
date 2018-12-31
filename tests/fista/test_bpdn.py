from __future__ import division
from builtins import range
from builtins import object

import pickle
import numpy as np

from sporco.fista import bpdn



def CallbackTest(obj):
    return bool(obj.k > 5)


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        lmbda = 1e-1
        try:
            b = bpdn.BPDN(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        try:
            b = bpdn.BPDN(D, s)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        try:
            opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 80,
                                     'BackTrack': {'Enabled': True},
                                     'RelStopTol': 1e-4})
            b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(b.getitstat().Rsdl).min() < 2.e-4


    def test_04(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        try:
            opt = bpdn.BPDN.Options(
                {'Verbose': False, 'MaxMainIter': 50, 'BackTrack':
                 {'Enabled': True, 'Robust': True}, 'RelStopTol': 1e-4})
            b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(b.getitstat().Rsdl).min() < 2.e-4


    def test_05(self):
        N = 8
        M = 8
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        try:
            opt = bpdn.BPDN.Options(
                {'FastSolve': True, 'Verbose': False, 'MaxMainIter': 10,
                 'BackTrack': {'Enabled': False}})
            b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_06(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        dt = np.float16
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 20,
                                 'BackTrack': {'Enabled': True},
                                 'DataType': dt})
        b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt


    def test_07(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        dt = np.float32
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 20,
                                 'BackTrack': {'Enabled': True},
                                 'DataType': dt})
        b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt


    def test_08(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        dt = np.float64
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 20,
                                 'BackTrack': {'Enabled': True},
                                 'DataType': dt})
        b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt


    def test_09(self):
        N = 64
        M = 2 * N
        L = 4
        np.random.seed(12345)
        D = np.random.randn(N, M)
        x0 = np.zeros((M, 1))
        si = np.random.permutation(list(range(0, M - 1)))
        x0[si[0:L]] = np.random.randn(L, 1)
        s0 = D.dot(x0)
        lmbda = 5e-3
        opt = bpdn.BPDN.Options(
            {'Verbose': False, 'MaxMainIter': 1000, 'BackTrack':
             {'Enabled': True, 'Robust': True}, 'RelStopTol': 5e-8})
        b = bpdn.BPDN(D, s0, lmbda, opt)
        b.solve()
        x1 = b.Y
        assert np.abs(b.itstat[-1].ObjFun - 0.012009) < 1e-5
        assert np.abs(b.itstat[-1].DFid - 1.9636082e-06) < 1e-5
        assert np.abs(b.itstat[-1].RegL1 - 2.401446) < 2e-4
        assert np.linalg.norm(x1 - x0) < 1e-3


    def test_10(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        lmbda = 1e-1
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 10})
        b = bpdn.BPDN(D, s, lmbda, opt)
        bp = pickle.dumps(b)
        c = pickle.loads(bp)
        Xb = b.solve()
        Xc = c.solve()
        assert np.linalg.norm(Xb - Xc) == 0.0
