from __future__ import division
from builtins import object

import pytest
import numpy as np
import pickle

from sporco.admm import cbpdn
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
        b = cbpdn.ConvBPDN(D, s, lmbda, dimK=0)
        assert(b.cri.dimC == 1)
        assert(b.cri.dimK == 0)


    def test_02(self):
        N = 16
        Nd = 5
        Cs = 3
        K = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, Cs, K)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda)
        assert(b.cri.dimC == 1)
        assert(b.cri.dimK == 1)


    def test_03(self):
        N = 16
        Nd = 5
        Cd = 3
        M = 4
        D = np.random.randn(Nd, Nd, Cd, M)
        s = np.random.randn(N, N, Cd)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda)
        assert(b.cri.dimC == 1)
        assert(b.cri.dimK == 0)


    def test_04(self):
        N = 16
        Nd = 5
        Cd = 3
        K = 5
        M = 4
        D = np.random.randn(Nd, Nd, Cd, M)
        s = np.random.randn(N, N, Cd, K)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda)
        assert(b.cri.dimC == 1)
        assert(b.cri.dimK == 1)


    def test_05(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda)
        assert(b.cri.dimC == 0)
        assert(b.cri.dimK == 1)


    def test_06(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        dt = np.float32
        opt = cbpdn.ConvBPDN.Options({'Verbose' : False, 'MaxMainIter' : 20,
                                 'AutoRho' : {'Enabled' : True},
                                 'DataType' : dt})
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt)
        b.solve()
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)
        assert(b.U.dtype == dt)


    def test_07(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        dt = np.float64
        opt = cbpdn.ConvBPDN.Options({'Verbose' : False, 'MaxMainIter' : 20,
                                 'AutoRho' : {'Enabled' : True},
                                 'DataType' : dt})
        lmbda = 1e-1
        b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt)
        b.solve()
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)
        assert(b.U.dtype == dt)


    def test_08(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        lmbda = 1e-1
        try:
            opt = cbpdn.ConvBPDN.Options({'LinSolveCheck' : True})
            b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)
        assert(np.array(b.getitstat().XSlvRelRes).max() < 1e-5)


    def test_09(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        try:
            b = cbpdn.ConvBPDN(D, s)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_10(self):
        N = 64
        M = 4
        Nd = 8
        D = np.random.randn(Nd, Nd, M)
        X0 = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X0[xp] = np.random.randn(X0[xp].size)
        S = np.sum(sl.ifftn(sl.fftn(D, (N, N), (0,1)) *
                   sl.fftn(X0, None, (0,1)), None, (0,1)).real, axis=2)
        lmbda = 1e-4
        rho = 1e-1
        opt = cbpdn.ConvBPDN.Options({'Verbose' : False, 'MaxMainIter' : 500,
                                      'RelStopTol' : 1e-3, 'rho' : rho,
                                      'AutoRho' : {'Enabled' : False}})
        b = cbpdn.ConvBPDN(D, S, lmbda, opt)
        b.solve()
        X1 = b.Y.squeeze()
        assert(sl.rrs(X0,X1) < 5e-5)
        Sr = b.reconstruct().squeeze()
        assert(sl.rrs(S,Sr) < 1e-4)


    def test_11(self):
        N = 63
        M = 4
        Nd = 8
        D = np.random.randn(Nd, Nd, M)
        X0 = np.zeros((N, N, M))
        xr = np.random.randn(N, N, M)
        xp = np.abs(xr) > 3
        X0[xp] = np.random.randn(X0[xp].size)
        S = np.sum(sl.ifftn(sl.fftn(D, (N, N), (0,1)) *
                   sl.fftn(X0, None, (0,1)), None, (0,1)).real, axis=2)
        lmbda = 1e-4
        rho = 1e-1
        opt = cbpdn.ConvBPDN.Options({'Verbose' : False, 'MaxMainIter' : 500,
                                      'RelStopTol' : 1e-3, 'rho' : rho,
                                      'AutoRho' : {'Enabled' : False}})
        b = cbpdn.ConvBPDN(D, S, lmbda, opt)
        b.solve()
        X1 = b.Y.squeeze()
        assert(sl.rrs(X0,X1) < 5e-5)
        Sr = b.reconstruct().squeeze()
        assert(sl.rrs(S,Sr) < 1e-4)


    def test_12(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, Cs)
        lmbda = 1e-1
        try:
            opt = cbpdn.ConvBPDN.Options({'LinSolveCheck' : True})
            b = cbpdn.ConvBPDN(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)
        assert(np.array(b.getitstat().XSlvRelRes).max() < 1e-5)


    def test_13(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, Cs)
        lmbda = 1e-1
        try:
            b = cbpdn.ConvBPDNJoint(D, s, lmbda, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_14(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        dt = np.float32
        opt = cbpdn.ConvBPDNJoint.Options({'Verbose' : False,
                        'MaxMainIter' : 20, 'AutoRho' : {'Enabled' : True},
                        'DataType' : dt})
        lmbda = 1e-1
        mu = 1e-2
        b = cbpdn.ConvBPDNJoint(D, s, lmbda, mu, opt=opt)
        b.solve()
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)
        assert(b.U.dtype == dt)


    def test_15(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        lmbda = 1e-1
        mu = 1e-2
        try:
            b = cbpdn.ConvElasticNet(D, s, lmbda, mu)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_16(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        dt = np.float32
        opt = cbpdn.ConvElasticNet.Options({'Verbose' : False,
                        'MaxMainIter' : 20, 'AutoRho' : {'Enabled' : True},
                        'DataType' : dt})
        lmbda = 1e-1
        mu = 1e-2
        b = cbpdn.ConvElasticNet(D, s, lmbda, mu, opt=opt)
        b.solve()
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)
        assert(b.U.dtype == dt)


    def test_17(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        lmbda = 1e-1
        mu = 1e-2
        try:
            b = cbpdn.ConvBPDNGradReg(D, s, lmbda, mu)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_18(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        dt = np.float32
        opt = cbpdn.ConvBPDNGradReg.Options({'Verbose' : False,
                        'MaxMainIter' : 20, 'AutoRho' : {'Enabled' : True},
                        'DataType' : dt})
        lmbda = 1e-1
        mu = 1e-2
        b = cbpdn.ConvBPDNGradReg(D, s, lmbda, mu, opt=opt)
        b.solve()
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)
        assert(b.U.dtype == dt)


    def test_21(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        lmbda = 1e-1
        try:
            b = cbpdn.ConvBPDNMaskDcpl(D, s, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_22(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        dt = np.float32
        opt = cbpdn.ConvBPDNMaskDcpl.Options({'Verbose' : False,
                    'MaxMainIter' : 20, 'AutoRho' : {'Enabled' : True},
                    'DataType' : dt})
        lmbda = 1e-1
        b = cbpdn.ConvBPDNMaskDcpl(D, s, lmbda, opt=opt)
        b.solve()
        assert(b.X.dtype == dt)
        assert(b.Y.dtype == dt)
        assert(b.U.dtype == dt)


    def test_23(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        w = np.ones(s.shape)
        lmbda = 1e-1
        try:
            b = cbpdn.AddMaskSim(cbpdn.ConvBPDN, D, s, w, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_24(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        w = np.ones(s.shape)
        dt = np.float32
        opt = cbpdn.ConvBPDN.Options({'Verbose' : False, 'MaxMainIter' : 20,
                                 'AutoRho' : {'Enabled' : True},
                                 'DataType' : dt})
        lmbda = 1e-1
        b = cbpdn.AddMaskSim(cbpdn.ConvBPDN, D, s, w, lmbda, opt=opt)
        b.solve()
        assert(b.cbpdn.X.dtype == dt)
        assert(b.cbpdn.Y.dtype == dt)
        assert(b.cbpdn.U.dtype == dt)


    def test_26(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        lmbda = 1e-1
        opt = cbpdn.ConvBPDN.Options({'Verbose' : False, 'MaxMainIter' : 10})
        b = cbpdn.ConvBPDN(D, s, lmbda, opt)
        bp = pickle.dumps(b)
        c = pickle.loads(bp)
        Xb = b.solve()
        Xc = c.solve()
        assert(np.linalg.norm(Xb-Xc)==0.0)


    def test_27(self):
        opt = cbpdn.GenericConvBPDN.Options({'AuxVarObj' : False})
        assert(opt['fEvalX'] is True and opt['gEvalY'] is False)
        opt['AuxVarObj'] = True
        assert(opt['fEvalX'] is False and opt['gEvalY'] is True)


    def test_28(self):
        opt = cbpdn.GenericConvBPDN.Options({'AuxVarObj' : True})
        assert(opt['fEvalX'] is False and opt['gEvalY'] is True)
        opt['AuxVarObj'] = False
        assert(opt['fEvalX'] is True and opt['gEvalY'] is False)
