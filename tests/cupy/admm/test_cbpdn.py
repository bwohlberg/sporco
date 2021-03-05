from __future__ import division
from builtins import object

import pickle
import pytest
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).compute_capability
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("GPU device inaccessible", allow_module_level=True)
except ImportError:
    pytest.skip("cupy not installed", allow_module_level=True)


from sporco.cupy.admm import cbpdn
from sporco.cupy.array import list2array
from sporco.cupy.fft import fftn, ifftn, fftconv
from sporco.cupy.linalg import rrs


class TestSet01(object):

    def setup_method(self, method):
        cp.random.seed(12345)


    def test_01(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda, dimK=0)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 0


    def test_02(self):
        N = 16
        Nd = 5
        Cs = 3
        K = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs, K)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 1


    def test_03(self):
        N = 16
        Nd = 5
        Cd = 3
        M = 4
        D = cp.random.randn(Nd, Nd, Cd, M)
        s = cp.random.randn(N, N, Cd)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 0


    def test_04(self):
        N = 16
        Nd = 5
        Cd = 3
        K = 5
        M = 4
        D = cp.random.randn(Nd, Nd, Cd, M)
        s = cp.random.randn(N, N, Cd, K)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 1
        assert b.cri.dimK == 1


    def test_05(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, K)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda)
        assert b.cri.dimC == 0
        assert b.cri.dimK == 1


    def test_06(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, K)
        dt = cp.float32
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 20,
                                      'AutoRho': {'Enabled': True},
                                      'DataType': dt})
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_07(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, K)
        dt = cp.float64
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 20,
                                      'AutoRho': {'Enabled': True},
                                      'DataType': dt})
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_08(self):
        Nr = 16
        Nc = 17
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(Nr, Nc)
        lmbda = 1e-1
        try:
            opt = cbpdn.ConvBPDN.Options({'LinSolveCheck': True})
            b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert list2array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_09(self):
        N = 16
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N)
        try:
            b = cbpdn.ConvBPDN(D, s)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_10(self):
        N = 64
        M = 4
        Nd = 8
        D = cp.random.randn(Nd, Nd, M)
        X0 = cp.zeros((N, N, M))
        xr = cp.random.randn(N, N, M)
        xp = cp.abs(xr) > 3
        X0[xp] = cp.random.randn(X0[xp].size)
        S = cp.sum(fftconv(D, X0, axes=(0, 1)), axis=2)
        lmbda = 1e-4
        rho = 1e-1
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 500,
                                      'RelStopTol': 1e-3, 'rho': rho,
                                      'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDN(D, S, lmbda, opt)
        b.solve()
        X1 = b.Y.squeeze()
        assert rrs(X0, X1) < 5e-5
        Sr = b.reconstruct().squeeze()
        assert rrs(S, Sr) < 1e-4


    def test_10cplx(self):
        N = 64
        M = 4
        Nd = 8
        D = cp.random.randn(Nd, Nd, M) + 1j * cp.random.randn(Nd, Nd, M)
        X0 = cp.zeros((N, N, M)) + 1j * cp.zeros((N, N, M))
        xr = cp.random.randn(N, N, M)
        xp = cp.abs(xr) > 3
        X0[xp] = (cp.random.randn(X0[xp].size) +
                  1j * cp.random.randn(X0[xp].size))
        S = cp.sum(fftconv(D, X0, axes=(0, 1)), axis=2)
        lmbda = 1e-4
        rho = 1e-1
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 500,
                                      'RelStopTol': 1e-3, 'rho': rho,
                                      'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDN(D, S, lmbda, opt)
        b.solve()
        X1 = b.Y.squeeze()
        assert rrs(X0, X1) < 5e-5
        Sr = b.reconstruct().squeeze()
        assert rrs(S, Sr) < 1e-4


    def test_11(self):
        N = 63
        M = 4
        Nd = 8
        D = cp.random.randn(Nd, Nd, M)
        X0 = cp.zeros((N, N, M))
        xr = cp.random.randn(N, N, M)
        xp = cp.abs(xr) > 3
        X0[xp] = cp.random.randn(X0[xp].size)
        S = cp.sum(ifftn(fftn(D, (N, N), (0, 1)) *
                         fftn(X0, None, (0, 1)), None, (0, 1)).real,
                   axis=2)
        lmbda = 1e-4
        rho = 1e-1
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 500,
                                      'RelStopTol': 1e-3, 'rho': rho,
                                      'AutoRho': {'Enabled': False}})
        b = cbpdn.ConvBPDN(D, S, lmbda, opt)
        b.solve()
        X1 = b.Y.squeeze()
        assert rrs(X0, X1) < 5e-5
        Sr = b.reconstruct().squeeze()
        assert rrs(S, Sr) < 1e-4


    def test_12(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs)
        lmbda = 1e-1
        try:
            opt = cbpdn.ConvBPDN.Options({'LinSolveCheck': True})
            b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert list2array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_13(self):
        N = 16
        Nd = 5
        Cd = 3
        M = 4
        D = cp.random.randn(Nd, Nd, Cd, M)
        s = cp.random.randn(N, N, Cd)
        lmbda = 1e-1
        try:
            opt = cbpdn.ConvBPDN.Options({'LinSolveCheck': True})
            b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert list2array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_14(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs)
        lmbda = 1e-1
        try:
            opt = cbpdn.ConvBPDNJoint.Options({'LinSolveCheck': True})
            b = cbpdn.ConvBPDNJoint(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert list2array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_15(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, K)
        dt = cp.float32
        opt = cbpdn.ConvBPDNJoint.Options(
            {'Verbose': False, 'MaxMainIter': 20, 'AutoRho': {'Enabled': True},
             'DataType': dt})
        lmbda = 1e-1
        mu = 1e-2
        b = cbpdn.ConvBPDNJoint(D, s, lmbda, mu, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_16(self):
        Nr = 16
        Nc = 17
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(Nr, Nc)
        lmbda = 1e-1
        mu = 1e-2
        try:
            b = cbpdn.ConvElasticNet(D, s, lmbda, mu)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_17(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, K)
        dt = cp.float32
        opt = cbpdn.ConvElasticNet.Options(
            {'Verbose': False, 'LinSolveCheck': True, 'MaxMainIter': 20,
             'AutoRho': {'Enabled': True}, 'DataType': dt})
        lmbda = 1e-1
        mu = 1e-2
        b = cbpdn.ConvElasticNet(D, s, lmbda, mu, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_18(self):
        Nr = 16
        Nc = 17
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(Nr, Nc)
        lmbda = 1e-1
        mu = 1e-2
        try:
            b = cbpdn.ConvBPDNGradReg(D, s, lmbda, mu)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_19(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, K)
        dt = cp.float32
        opt = cbpdn.ConvBPDNGradReg.Options(
            {'Verbose': False, 'LinSolveCheck': True, 'MaxMainIter': 20,
             'AutoRho': {'Enabled': True}, 'DataType': dt})
        lmbda = 1e-1
        mu = 1e-2
        b = cbpdn.ConvBPDNGradReg(D, s, lmbda, mu, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_20(self):
        N = 16
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N)
        epsilon = 1e0
        try:
            b = cbpdn.ConvMinL1InL2Ball(D, s, epsilon)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_21(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, K)
        dt = cp.float32
        opt = cbpdn.ConvMinL1InL2Ball.Options(
            {'Verbose': False, 'MaxMainIter': 20, 'AutoRho':
             {'Enabled': True}, 'DataType': dt})
        epsilon = 1e0
        b = cbpdn.ConvMinL1InL2Ball(D, s, epsilon, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_22(self):
        N = 32
        M = 4
        Nd = 8
        D = cp.random.randn(Nd, Nd, M)
        D /= cp.sqrt(cp.sum(D**2, axis=(0, 1)))
        X0 = cp.zeros((N, N, M))
        xr = cp.random.randn(N, N, M)
        xp = cp.abs(xr) > 3
        X0[xp] = cp.random.randn(X0[xp].size)
        S = cp.sum(fftconv(D, X0, axes=(0, 1)), axis=2)
        lmbda = 1e-3
        opt = cbpdn.ConvBPDN.Options(
            {'Verbose': False, 'MaxMainIter': 500, 'RelStopTol': 1e-5,
             'rho': 5e-1, 'AutoRho': {'Enabled': False}})
        bp = cbpdn.ConvBPDN(D, S, lmbda, opt)
        Xp = bp.solve()
        epsilon = cp.linalg.norm(bp.reconstruct(Xp).squeeze() - S)
        opt = cbpdn.ConvMinL1InL2Ball.Options(
            {'Verbose': False, 'MaxMainIter': 500, 'RelStopTol': 1e-5,
             'rho': 2e2, 'RelaxParam': 1.0, 'AutoRho': {'Enabled': False}})
        bc = cbpdn.ConvMinL1InL2Ball(D, S, epsilon=epsilon, opt=opt)
        Xc = bc.solve()
        assert cp.linalg.norm(Xp - Xc) / cp.linalg.norm(Xp) < 1e-3
        assert cp.abs(cp.linalg.norm(Xp.ravel(), 1) -
                      cp.linalg.norm(Xc.ravel(), 1)) < 1e-3


    def test_23(self):
        N = 16
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N)
        lmbda = 1e-1
        try:
            b = cbpdn.ConvBPDNMaskDcpl(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_24(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs)
        lmbda = 1e-1
        try:
            b = cbpdn.ConvBPDNMaskDcpl(D, s, lmbda, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_25(self):
        N = 16
        Nd = 5
        Cs = 3
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, Cs, K)
        lmbda = 1e-1
        try:
            b = cbpdn.ConvBPDNMaskDcpl(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_26(self):
        N = 16
        Nd = 5
        Cd = 3
        M = 4
        D = cp.random.randn(Nd, Nd, Cd, M)
        s = cp.random.randn(N, N, Cd)
        lmbda = 1e-1
        try:
            b = cbpdn.ConvBPDNMaskDcpl(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_27(self):
        N = 16
        Nd = 5
        Cd = 3
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, Cd, M)
        s = cp.random.randn(N, N, Cd, K)
        lmbda = 1e-1
        try:
            b = cbpdn.ConvBPDNMaskDcpl(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_28(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N, K)
        dt = cp.float32
        opt = cbpdn.ConvBPDNMaskDcpl.Options(
            {'Verbose': False, 'LinSolveCheck': True, 'MaxMainIter': 20,
             'AutoRho': {'Enabled': True}, 'DataType': dt})
        lmbda = 1e-1
        b = cbpdn.ConvBPDNMaskDcpl(D, s, lmbda, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt


    def test_29(self):
        N = 16
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N)
        w = cp.ones(s.shape)
        lmbda = 1e-1
        try:
            b = cbpdn.AddMaskSim(cbpdn.ConvBPDN, D, s, w, lmbda)
            b.solve()
            b.reconstruct()
        except Exception as e:
            print(e)
            assert 0


    def test_30(self):
        N = 16
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N)
        w = cp.ones(s.shape)
        dt = cp.float32
        opt = cbpdn.ConvBPDN.Options(
            {'Verbose': False, 'MaxMainIter': 20, 'AutoRho': {'Enabled': True},
             'DataType': dt})
        lmbda = 1e-1
        b = cbpdn.AddMaskSim(cbpdn.ConvBPDN, D, s, w, lmbda, opt=opt)
        b.solve()
        assert b.cbpdn.X.dtype == dt
        assert b.cbpdn.Y.dtype == dt
        assert b.cbpdn.U.dtype == dt


    def test_31(self):
        N = 16
        Nd = 5
        M = 4
        D0 = cp.random.randn(Nd, Nd, M)
        D1 = cp.random.randn(Nd, Nd, M)
        s0 = cp.random.randn(N, N)
        s1 = cp.random.randn(N, N)
        lmbda = 1e-1
        try:
            b = cbpdn.MultiDictConvBPDN(cbpdn.ConvBPDN, (D0, D1), (s0, s1),
                                        lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_32(self):
        N = 16
        Nd = 5
        M = 4
        D = cp.random.randn(Nd, Nd, M)
        s = cp.random.randn(N, N)
        lmbda = 1e-1
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 10})
        b = cbpdn.ConvBPDN(D, s, lmbda, opt)
        bp = pickle.dumps(b)
        c = pickle.loads(bp)
        Xb = b.solve()
        Xc = c.solve()
        assert cp.linalg.norm(Xb - Xc) == 0.0


    def test_33(self):
        opt = cbpdn.GenericConvBPDN.Options({'AuxVarObj': False})
        assert opt['fEvalX'] is True and opt['gEvalY'] is False
        opt['AuxVarObj'] = True
        assert opt['fEvalX'] is False and opt['gEvalY'] is True


    def test_34(self):
        opt = cbpdn.GenericConvBPDN.Options({'AuxVarObj': True})
        assert opt['fEvalX'] is False and opt['gEvalY'] is True
        opt['AuxVarObj'] = False
        assert opt['fEvalX'] is True and opt['gEvalY'] is False
