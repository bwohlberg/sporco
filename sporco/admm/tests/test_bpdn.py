from __future__ import division
from builtins import range
from builtins import object

import pytest
import numpy as np
from scipy import linalg
import pickle

from sporco.admm import bpdn
import sporco.linalg as sl


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
            opt = bpdn.BPDN.Options({'Verbose' : True, 'MaxMainIter' : 20,
                                     'AutoRho' : {'StdResiduals' : True}})
            b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_04(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        dt = np.float16
        opt = bpdn.BPDN.Options({'Verbose' : False, 'MaxMainIter' : 20,
                                 'AutoRho' : {'Enabled' : True},
                                 'DataType' : dt})
        b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
        b.solve()
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)
        assert(b.U.dtype == dt)


    def test_05(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        dt = np.float32
        opt = bpdn.BPDN.Options({'Verbose' : False, 'MaxMainIter' : 20,
                                 'AutoRho' : {'Enabled' : True},
                                 'DataType' : dt})
        b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
        b.solve()
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)
        assert(b.U.dtype == dt)


    def test_06(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        dt = np.float64
        opt = bpdn.BPDN.Options({'Verbose' : False, 'MaxMainIter' : 20,
                                 'AutoRho' : {'Enabled' : True},
                                 'DataType' : dt})
        b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
        b.solve()
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)
        assert(b.U.dtype == dt)


    def test_07(self):
        N = 64
        M = 2*N
        L = 4
        np.random.seed(12345)
        D = np.random.randn(N, M)
        x0 = np.zeros((M, 1))
        si = np.random.permutation(list(range(0, M-1)))
        x0[si[0:L]] = np.random.randn(L, 1)
        s0 = D.dot(x0)
        lmbda = 5e-3
        opt = bpdn.BPDN.Options({'Verbose' : False, 'MaxMainIter' : 500,
                                 'RelStopTol' : 5e-4})
        b = bpdn.BPDN(D, s0, lmbda, opt)
        b.solve()
        x1 = b.Y
        assert(np.abs(b.itstat[-1].ObjFun - 0.012009) < 1e-5)
        assert(np.abs(b.itstat[-1].DFid - 1.9636082e-06) < 1e-5)
        assert(np.abs(b.itstat[-1].RegL1 - 2.401446) < 1e-5)
        assert(linalg.norm(x1-x0) < 1e-3)


    def test_08(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        lmbda = 1e-1
        mu = 1e-2
        try:
            b = bpdn.BPDNJoint(D, s, lmbda, mu)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_09(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        dt = np.float16
        opt = bpdn.BPDNJoint.Options({'Verbose' : False, 'MaxMainIter' : 20,
                                 'AutoRho' : {'Enabled' : True},
                                 'DataType' : dt})
        b = bpdn.BPDNJoint(D, s, lmbda=1.0, mu=0.1, opt=opt)
        b.solve()
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)
        assert(b.U.dtype == dt)


    def test_10(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        lmbda = 1e-1
        mu = 1e-2
        try:
            b = bpdn.ElasticNet(D, s, lmbda, mu)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_11(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        dt = np.float16
        opt = bpdn.ElasticNet.Options({'Verbose' : False, 'MaxMainIter' : 20,
                                 'AutoRho' : {'Enabled' : True},
                                 'DataType' : dt})
        b = bpdn.ElasticNet(D, s, lmbda=1.0, mu=0.1, opt=opt)
        b.solve()
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)
        assert(b.U.dtype == dt)


    def test_16(self):
        N = 8
        M = 16
        D = np.random.randn(N, M)
        s = np.random.randn(N, 1)
        lmbda = 1e-1
        opt = bpdn.BPDN.Options({'Verbose' : False, 'MaxMainIter' : 10})
        b = bpdn.BPDN(D, s, lmbda, opt)
        bp = pickle.dumps(b)
        c = pickle.loads(bp)
        Xb = b.solve()
        Xc = c.solve()
        assert(linalg.norm(Xb-Xc)==0.0)


    def test_17(self):
        opt = bpdn.GenericBPDN.Options({'AuxVarObj' : False})
        assert(opt['fEvalX'] is True and opt['gEvalY'] is False)
        opt['AuxVarObj'] = True
        assert(opt['fEvalX'] is False and opt['gEvalY'] is True)


    def test_18(self):
        opt = bpdn.GenericBPDN.Options({'AuxVarObj' : True})
        assert(opt['fEvalX'] is False and opt['gEvalY'] is True)
        opt['AuxVarObj'] = False
        assert(opt['fEvalX'] is True and opt['gEvalY'] is False)
