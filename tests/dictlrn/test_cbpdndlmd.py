from __future__ import division
from builtins import object

import numpy as np

from sporco.dictlrn import cbpdndlmd



class TestSet01(object):

    def setup_method(self, method):
        N = 16
        Nd = 5
        M = 4
        K = 3
        np.random.seed(12345)
        self.D0 = np.random.randn(Nd, Nd, M)
        self.S = np.random.randn(N, N, K)


    def test_01(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        opt = cbpdndlmd.ConvBPDNMaskDictLearn.Options({'MaxMainIter': 10})
        try:
            b = cbpdndlmd.ConvBPDNMaskDictLearn(self.D0, self.S, lmbda,
                                                W, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        opt = cbpdndlmd.ConvBPDNMaskDictLearn.Options(
            {'MaxMainIter': 5, 'CCMOD': {'CG': {'MaxIter': 1}}},
            dmethod='cg')
        try:
            b = cbpdndlmd.ConvBPDNMaskDictLearn(self.D0, self.S, lmbda,
                                                W, opt=opt, dmethod='cg')
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        opt = cbpdndlmd.ConvBPDNMaskDictLearn.Options({'MaxMainIter': 10},
                                                      dmethod='cns')
        try:
            b = cbpdndlmd.ConvBPDNMaskDictLearn(self.D0, self.S, lmbda, W,
                                                opt=opt, dmethod='cns')
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_04(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        opt = cbpdndlmd.ConvBPDNMaskDictLearn.Options(
            {'AccurateDFid': True, 'MaxMainIter': 10})
        try:
            b = cbpdndlmd.ConvBPDNMaskDictLearn(self.D0, self.S, lmbda, W,
                                                opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_05(self):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, Nc, M)
        S = np.random.randn(N, N, Nc, K)
        lmbda = 1e-1
        W = np.ones((N, N, 1, K, 1))
        opt = cbpdndlmd.ConvBPDNMaskDictLearn.Options({'MaxMainIter': 10})
        try:
            b = cbpdndlmd.ConvBPDNMaskDictLearn(D0, S, lmbda, W, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_06(self):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, 1, M)
        S = np.random.randn(N, N, Nc, K)
        lmbda = 1e-1
        W = np.ones((N, N, Nc, K, 1))
        opt = cbpdndlmd.ConvBPDNMaskDictLearn.Options({'MaxMainIter': 10})
        try:
            b = cbpdndlmd.ConvBPDNMaskDictLearn(D0, S, lmbda, W, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_07(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        opt = cbpdndlmd.ConvBPDNMaskDictLearn.Options(
            {'AccurateDFid': True, 'MaxMainIter': 10}, dmethod='pgm')
        try:
            b = cbpdndlmd.ConvBPDNMaskDictLearn(self.D0, self.S, lmbda, W,
                                                opt=opt, dmethod='pgm')
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_08(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        opt = cbpdndlmd.ConvBPDNMaskDictLearn.Options(
            {'AccurateDFid': True, 'MaxMainIter': 10}, xmethod='pgm')
        try:
            b = cbpdndlmd.ConvBPDNMaskDictLearn(self.D0, self.S, lmbda, W,
                                                opt=opt, xmethod='pgm')
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_09(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        opt = cbpdndlmd.ConvBPDNMaskDictLearn.Options(
            {'AccurateDFid': True, 'MaxMainIter': 10},
            xmethod='pgm', dmethod='cns')
        try:
            b = cbpdndlmd.ConvBPDNMaskDictLearn(
                self.D0, self.S, lmbda, W, opt=opt, xmethod='pgm',
                dmethod='cns')
            b.solve()
        except Exception as e:
            print(e)
            assert 0
