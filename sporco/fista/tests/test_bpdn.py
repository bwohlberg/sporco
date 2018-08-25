from __future__ import division
from builtins import range
from builtins import object

import pytest
import numpy as np
from scipy import linalg
import pickle

from sporco.fista import bpdn
import sporco.linalg as sl



def CallbackTest(obj):
    if obj.k > 5:
        return True
    else:
        return False


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
            assert(0)


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
            assert(0)


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
            assert(0)
        assert(np.array(b.getitstat().Rsdl).min() < 2.e-4)


    def test_05(self):
        N = 8
        M = 8
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        try:
            opt = bpdn.BPDN.Options({'FastSolve': True, 'Verbose': False,
                'MaxMainIter': 10, 'BackTrack': {'Enabled': False}})
            b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


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
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)


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
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)


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
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)


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
        assert(linalg.norm(Xb-Xc)==0.0)
