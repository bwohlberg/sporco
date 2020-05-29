from __future__ import division
from builtins import object

import pickle
import pytest
import numpy as np
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).compute_capability
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("GPU device inaccessible", allow_module_level=True)
except ImportError:
    pytest.skip("cupy not installed", allow_module_level=True)


from sporco.cupy.admm import bpdn
from sporco.cupy.array import list2array



def CallbackTest(obj):
    return bool(obj.k > 5)



class TestSet01(object):

    def setup_method(self, method):
        cp.random.seed(12345)


    def test_01(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
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
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        try:
            b = bpdn.BPDN(D, s)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        try:
            opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 20,
                                     'LinSolveCheck': True,
                                     'AutoRho': {'StdResiduals': True}})
            b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert list2array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_04(self):
        N = 8
        M = 8
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        try:
            opt = bpdn.BPDN.Options(
                {'FastSolve': True, 'Verbose': False, 'MaxMainIter': 10,
                 'AutoRho': {'Enabled': False}})
            b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_05(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        dt = cp.float16
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 20,
                                 'AutoRho': {'Enabled': True},
                                 'DataType': dt})
        b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_06(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        dt = cp.float32
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 20,
                                 'AutoRho': {'Enabled': True},
                                 'DataType': dt})
        b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_07(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        dt = cp.float64
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 20,
                                 'AutoRho': {'Enabled': True},
                                 'DataType': dt})
        b = bpdn.BPDN(D, s, lmbda=1.0, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_08(self):
        N = 64
        M = 2 * N
        L = 4
        np.random.seed(12345)
        D = cp.array(np.random.randn(N, M))
        x0 = cp.zeros((M, 1))
        si = cp.array(np.random.permutation(M))
        x0[si[0:L]] = cp.array(np.random.randn(L, 1))
        s0 = D.dot(x0)
        lmbda = 5e-3
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 500,
                                 'RelStopTol': 5e-4})
        b = bpdn.BPDN(D, s0, lmbda, opt)
        b.solve()
        x1 = b.Y
        assert cp.abs(b.itstat[-1].ObjFun - 0.012009) < 1e-5
        assert cp.abs(b.itstat[-1].DFid - 1.9636082e-06) < 1e-5
        assert cp.abs(b.itstat[-1].RegL1 - 2.401446) < 1e-5
        assert cp.linalg.norm(x1 - x0) < 1e-3


    def test_09(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        lmbda = 1e-1
        mu = 1e-2
        try:
            b = bpdn.BPDNJoint(D, s, lmbda, mu)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_10(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        dt = cp.float16
        opt = bpdn.BPDNJoint.Options({'Verbose': False, 'MaxMainIter': 20,
                                      'AutoRho': {'Enabled': True},
                                      'DataType': dt})
        b = bpdn.BPDNJoint(D, s, lmbda=1.0, mu=0.1, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_11(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        lmbda = 1e-1
        mu = 1e-2
        try:
            b = bpdn.ElasticNet(D, s, lmbda, mu)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_12(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        lmbda = 1e-1
        mu = 1e-2
        try:
            opt = bpdn.ElasticNet.Options(
                {'Verbose': False, 'MaxMainIter': 20, 'LinSolveCheck': True,
                 'AutoRho': {'StdResiduals': True}})
            b = bpdn.ElasticNet(D, s, lmbda, mu, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert list2array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_13(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        dt = cp.float16
        opt = bpdn.ElasticNet.Options({'Verbose': False, 'MaxMainIter': 20,
                                       'AutoRho': {'Enabled': True},
                                       'DataType': dt})
        b = bpdn.ElasticNet(D, s, lmbda=1.0, mu=0.1, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_14(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        gamma = 1e0
        try:
            b = bpdn.BPDNProjL1(D, s, gamma)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_15(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        dt = cp.float32
        opt = bpdn.BPDNProjL1.Options({'Verbose': False, 'MaxMainIter': 20,
                                       'AutoRho': {'Enabled': True},
                                       'DataType': dt})
        b = bpdn.BPDNProjL1(D, s, gamma=1.0, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_16(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        epsilon = 1e-1
        try:
            b = bpdn.MinL1InL2Ball(D, s, epsilon)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_17(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        dt = cp.float16
        opt = bpdn.MinL1InL2Ball.Options({'Verbose': False, 'MaxMainIter': 20,
                                          'AutoRho': {'Enabled': True},
                                          'DataType': dt})
        b = bpdn.MinL1InL2Ball(D, s, epsilon=1.0, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_18(self):
        N = 64
        M = 2 * N
        L = 8
        cp.random.seed(12345)
        D = cp.random.randn(N, M)
        x0 = cp.zeros((M, 1))
        si = cp.random.permutation(M)
        x0[si[0:L]] = cp.random.randn(L, 1)
        s = D.dot(x0)
        lmbda = 5e-2
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 300,
                                 'RelStopTol': 1e-5, 'AutoRho':
                                 {'Enabled': False}})
        bp = bpdn.BPDN(D, s, lmbda=lmbda, opt=opt)
        Xp = bp.solve()
        epsilon = cp.linalg.norm(D.dot(Xp) - s)
        opt = bpdn.MinL1InL2Ball.Options(
            {'Verbose': False, 'MaxMainIter': 300, 'RelStopTol': 1e-5,
             'rho': 2e1, 'AutoRho': {'Enabled': False}})
        bc = bpdn.MinL1InL2Ball(D, s, epsilon=epsilon, opt=opt)
        Xc = bc.solve()
        assert cp.linalg.norm(Xp - Xc) / cp.linalg.norm(Xp) < 1e-3
        assert cp.abs(cp.linalg.norm(Xp, 1) - cp.linalg.norm(Xc, 1)) < 1e-3


    def test_19(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        lmbda = 1e-1
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 10})
        b = bpdn.BPDN(D, s, lmbda, opt)
        bp = pickle.dumps(b)
        c = pickle.loads(bp)
        Xb = b.solve()
        Xc = c.solve()
        assert cp.linalg.norm(Xb - Xc) == 0.0


    def test_20(self):
        opt = bpdn.GenericBPDN.Options({'AuxVarObj': False})
        assert opt['fEvalX'] is True and opt['gEvalY'] is False
        opt['AuxVarObj'] = True
        assert opt['fEvalX'] is False and opt['gEvalY'] is True


    def test_21(self):
        opt = bpdn.GenericBPDN.Options({'AuxVarObj': True})
        assert opt['fEvalX'] is False and opt['gEvalY'] is True
        opt['AuxVarObj'] = False
        assert opt['fEvalX'] is True and opt['gEvalY'] is False


    @pytest.mark.filterwarnings('ignore:admm.ADMM.runtime')
    def test_22(self):
        N = 8
        M = 16
        D = cp.random.randn(N, M)
        s = cp.random.randn(N, 1)
        lmbda = 1e-1
        opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 10,
                                 'Callback': CallbackTest, 'RelaxParam': 1.0})
        b = bpdn.BPDN(D, s, lmbda, opt=opt)
        assert b.getitstat() is None
        b.solve()
        assert b.runtime > 0.0
        assert b.k == 7
        assert b.var_x() is not None
        assert b.var_y() is not None
        assert b.var_u() is not None
