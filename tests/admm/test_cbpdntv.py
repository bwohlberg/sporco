from __future__ import division
from builtins import object

import numpy as np

from sporco.admm import cbpdntv



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        lmbda = 1e-1
        mu = 1e-2
        try:
            opt = cbpdntv.ConvBPDNScalarTV.Options({'LinSolveCheck': True})
            b = cbpdntv.ConvBPDNScalarTV(D, s, lmbda, mu, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_02(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        lmbda = 1e-1
        mu = 1e-2
        try:
            opt = cbpdntv.ConvBPDNVectorTV.Options({'LinSolveCheck': True})
            b = cbpdntv.ConvBPDNVectorTV(D, s, lmbda, mu, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_03(self):
        N = 16
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N)
        lmbda = 1e-1
        mu = 1e-2
        try:
            opt = cbpdntv.ConvBPDNRecTV.Options({'LinSolveCheck': True})
            b = cbpdntv.ConvBPDNRecTV(D, s, lmbda, mu, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_04(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, Cs)
        lmbda = 1e-1
        mu = 1e-2
        try:
            opt = cbpdntv.ConvBPDNScalarTV.Options({'LinSolveCheck': True})
            b = cbpdntv.ConvBPDNScalarTV(D, s, lmbda, mu, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_05(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, Cs)
        lmbda = 1e-1
        mu = 1e-2
        try:
            opt = cbpdntv.ConvBPDNVectorTV.Options({'LinSolveCheck': True})
            b = cbpdntv.ConvBPDNVectorTV(D, s, lmbda, mu, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_06(self):
        N = 16
        Nd = 5
        Cs = 3
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, Cs)
        lmbda = 1e-1
        mu = 1e-2
        try:
            opt = cbpdntv.ConvBPDNRecTV.Options({'LinSolveCheck': True})
            b = cbpdntv.ConvBPDNRecTV(D, s, lmbda, mu, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_07(self):
        N = 16
        Nd = 5
        Cd = 3
        M = 4
        D = np.random.randn(Nd, Nd, Cd, M)
        s = np.random.randn(N, N, Cd)
        lmbda = 1e-1
        try:
            opt = cbpdntv.ConvBPDNScalarTV.Options({'LinSolveCheck': True})
            b = cbpdntv.ConvBPDNScalarTV(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_08(self):
        N = 16
        Nd = 5
        Cd = 3
        M = 4
        D = np.random.randn(Nd, Nd, Cd, M)
        s = np.random.randn(N, N, Cd)
        lmbda = 1e-1
        try:
            opt = cbpdntv.ConvBPDNVectorTV.Options({'LinSolveCheck': True})
            b = cbpdntv.ConvBPDNVectorTV(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(b.getitstat().XSlvRelRes).max() < 1e-5


    def test_09(self):
        N = 16
        Nd = 5
        Cd = 3
        M = 4
        D = np.random.randn(Nd, Nd, Cd, M)
        s = np.random.randn(N, N, Cd)
        lmbda = 1e-1
        try:
            opt = cbpdntv.ConvBPDNRecTV.Options({'LinSolveCheck': True})
            b = cbpdntv.ConvBPDNRecTV(D, s, lmbda, opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(b.getitstat().XSlvRelRes).max() < 1e-5
