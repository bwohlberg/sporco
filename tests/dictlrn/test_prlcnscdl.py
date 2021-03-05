from __future__ import division
from builtins import object

import numpy as np

from sporco.dictlrn import prlcnscdl
from sporco.dictlrn import cbpdndl
from sporco.dictlrn import cbpdndlmd


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)
        N = 16
        Nd = 5
        M = 4
        K = 3
        self.D0 = np.random.randn(Nd, Nd, M)
        self.S = np.random.randn(N, N, K)
        self.D0c = np.random.randn(Nd, Nd, M) + 1j * np.random.randn(Nd, Nd, M)
        self.Sc = np.random.randn(N, N, K) + 1j * np.random.randn(N, N, K)


    def test_01(self):
        lmbda = 1e-1
        opt = prlcnscdl.ConvBPDNDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNDictLearn_Consensus(self.D0, self.S[..., 0],
                                            lmbda, opt=opt, nproc=2, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        lmbda = 1e-1
        opt = prlcnscdl.ConvBPDNDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNDictLearn_Consensus(self.D0, self.S, lmbda,
                                                      opt=opt, nproc=2)
            D = b.solve()
            assert np.isrealobj(D)
        except Exception as e:
            print(e)
            assert 0


    def test_02cplx(self):
        lmbda = 1e-1
        opt = prlcnscdl.ConvBPDNDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNDictLearn_Consensus(self.D0c, self.Sc, lmbda,
                                                      opt=opt, nproc=2)
            D = b.solve()
            assert np.iscomplexobj(D)
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        lmbda = 1e-1
        opt = prlcnscdl.ConvBPDNDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNDictLearn_Consensus(self.D0, self.S, lmbda,
                                                      opt, nproc=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_04(self):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, Nc, M)
        S = np.random.randn(N, N, Nc, K)
        lmbda = 1e-1
        opt = prlcnscdl.ConvBPDNDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNDictLearn_Consensus(D0, S, lmbda, opt=opt,
                                                      nproc=2)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_05(self):
        lmbda = 1e-1
        Nit = 10
        opts = cbpdndl.ConvBPDNDictLearn.Options(
            {'MaxMainIter': Nit, 'AccurateDFid': True,
             'CBPDN': {'RelaxParam': 1.0, 'AutoRho': {'Enabled': False}},
             'CCMOD': {'RelaxParam': 1.0, 'AutoRho': {'Enabled': False}}},
             dmethod='cns')
        bs = cbpdndl.ConvBPDNDictLearn(self.D0, self.S, lmbda, opt=opts,
                                       dmethod='cns')
        Ds = bs.solve()
        optp = prlcnscdl.ConvBPDNDictLearn_Consensus.Options(
            {'MaxMainIter': Nit, 'CBPDN': {'RelaxParam': 1.0},
             'CCMOD': {'RelaxParam': 1.0}})
        bp = prlcnscdl.ConvBPDNDictLearn_Consensus(self.D0, self.S, lmbda,
                                                   opt=optp, nproc=2)
        Dp = bp.solve()
        assert np.linalg.norm(Ds - Dp) < 1e-7
        assert np.abs(
            bs.getitstat().ObjFun[-1] - bp.getitstat().ObjFun[-1] < 1e-7)


    def test_06(self):
        lmbda = 1e-1
        Nit = 10
        opts = cbpdndl.ConvBPDNDictLearn.Options(
            {'MaxMainIter': Nit, 'AccurateDFid': True,
             'CBPDN': {'RelaxParam': 1.8, 'AutoRho': {'Enabled': False}},
             'CCMOD': {'RelaxParam': 1.8, 'AutoRho': {'Enabled': False}}},
             dmethod='cns')
        bs = cbpdndl.ConvBPDNDictLearn(self.D0, self.S, lmbda, opt=opts,
                                       dmethod='cns')
        Ds = bs.solve()
        optp = prlcnscdl.ConvBPDNDictLearn_Consensus.Options(
            {'MaxMainIter': Nit, 'CBPDN': {'RelaxParam': 1.8},
             'CCMOD': {'RelaxParam': 1.8}})
        bp = prlcnscdl.ConvBPDNDictLearn_Consensus(self.D0, self.S, lmbda,
                                                   opt=optp, nproc=2)
        Dp = bp.solve()
        assert np.linalg.norm(Ds - Dp) < 1e-7
        assert np.abs(
            bs.getitstat().ObjFun[-1] - bp.getitstat().ObjFun[-1] < 1e-7)


    def test_07(self):
        lmbda = 1e-1
        W = np.array([1.0])
        opt = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus(self.D0,
                        self.S[..., 0], lmbda, W, opt=opt, nproc=2, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_08(self):
        lmbda = 1e-1
        W = np.array([1.0])
        opt = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus(self.D0,
                        self.S, lmbda, W, opt=opt, nproc=2)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_09(self):
        lmbda = 1e-1
        W = np.array([1.0])
        opt = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus(self.D0,
                        self.S, lmbda, W, opt=opt, nproc=0)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_10(self):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, Nc, M)
        S = np.random.randn(N, N, Nc, K)
        lmbda = 1e-1
        W = np.array([1.0])
        opt = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus(D0,
                        S, lmbda, W, opt=opt, nproc=2)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_11(self):
        lmbda = 1e-1
        W = np.array([1.0])
        Nit = 10
        opts = cbpdndlmd.ConvBPDNMaskDictLearn.Options(
            {'MaxMainIter': Nit, 'AccurateDFid': True,
             'CBPDN': {'RelaxParam': 1.0, 'AutoRho': {'Enabled': False}},
             'CCMOD': {'RelaxParam': 1.0, 'AutoRho': {'Enabled': False}}},
             xmethod='admm', dmethod='cns')
        bs = cbpdndlmd.ConvBPDNMaskDictLearn(self.D0, self.S, lmbda, W,
                                             opt=opts)
        Ds = bs.solve()
        optp = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
            {'MaxMainIter': Nit, 'CBPDN': {'RelaxParam': 1.0},
             'CCMOD': {'RelaxParam': 1.0}})
        bp = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus(self.D0,
                    self.S, lmbda, W, opt=optp, nproc=2)
        Dp = bp.solve()
        assert np.linalg.norm(Ds - Dp) < 1e-7
        assert np.abs(
            bs.getitstat().ObjFun[-1] - bp.getitstat().ObjFun[-1] < 1e-7)


    def test_12(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        opt = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus(self.D0,
                        self.S, lmbda, W, opt=opt, nproc=2)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_13(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, 1, 1))
        opt = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus(self.D0,
                        self.S, lmbda, W, opt=opt, nproc=2)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_14(self):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, Nc, M)
        S = np.random.randn(N, N, Nc, K)
        lmbda = 1e-1
        W = np.ones(S.shape[0:2] + (1, 1, 1))
        opt = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus(D0,
                        S, lmbda, W, opt=opt, nproc=2)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_15(self):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 2
        D0 = np.random.randn(Nd, Nd, Nc, M)
        S = np.random.randn(N, N, Nc, K)
        lmbda = 1e-1
        W = np.ones(S.shape + (1,))
        opt = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
            {'MaxMainIter': 10})
        try:
            b = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus(D0,
                        S, lmbda, W, opt=opt, nproc=2)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_16(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, 1, 1))
        Nit = 10
        opts = cbpdndlmd.ConvBPDNMaskDictLearn.Options(
            {'MaxMainIter': Nit, 'AccurateDFid': True,
             'CBPDN': {'RelaxParam': 1.0, 'AutoRho': {'Enabled': False}},
             'CCMOD': {'RelaxParam': 1.0, 'AutoRho': {'Enabled': False}}},
             xmethod='admm', dmethod='cns')
        bs = cbpdndlmd.ConvBPDNMaskDictLearn(self.D0, self.S, lmbda, W,
                    opt=opts)
        Ds = bs.solve()
        optp = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
            {'MaxMainIter': Nit, 'CBPDN': {'RelaxParam': 1.0},
             'CCMOD': {'RelaxParam': 1.0}})
        bp = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus(self.D0,
                    self.S, lmbda, W, opt=optp, nproc=2)
        Dp = bp.solve()
        assert np.linalg.norm(Ds - Dp) < 1e-7
        assert np.abs(
            bs.getitstat().ObjFun[-1] - bp.getitstat().ObjFun[-1] < 1e-7)


    def test_17(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        Nit = 10
        opts = cbpdndlmd.ConvBPDNMaskDictLearn.Options(
            {'MaxMainIter': Nit, 'AccurateDFid': True,
             'CBPDN': {'RelaxParam': 1.0, 'AutoRho': {'Enabled': False}},
             'CCMOD': {'RelaxParam': 1.0, 'AutoRho': {'Enabled': False}}},
             xmethod='admm', dmethod='cns')
        bs = cbpdndlmd.ConvBPDNMaskDictLearn(self.D0, self.S, lmbda, W,
                                             opt=opts)
        Ds = bs.solve()
        optp = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
            {'MaxMainIter': Nit, 'CBPDN': {'RelaxParam': 1.0},
             'CCMOD': {'RelaxParam': 1.0}})
        bp = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus(self.D0,
                    self.S, lmbda, W, opt=optp, nproc=2)
        Dp = bp.solve()
        assert np.linalg.norm(Ds - Dp) < 1e-7
        assert np.abs(
            bs.getitstat().ObjFun[-1] - bp.getitstat().ObjFun[-1] < 1e-7)


    def test_18(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        Nit = 10
        opts = cbpdndlmd.ConvBPDNMaskDictLearn.Options(
            {'MaxMainIter': Nit, 'AccurateDFid': True,
             'CBPDN': {'RelaxParam': 1.8, 'AutoRho': {'Enabled': False}},
             'CCMOD': {'RelaxParam': 1.8, 'AutoRho': {'Enabled': False}}},
             xmethod='admm', dmethod='cns')
        bs = cbpdndlmd.ConvBPDNMaskDictLearn(self.D0, self.S, lmbda, W,
                                             opt=opts)
        Ds = bs.solve()
        optp = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus.Options(
            {'MaxMainIter': Nit, 'CBPDN': {'RelaxParam': 1.8},
             'CCMOD': {'RelaxParam': 1.8}})
        bp = prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus(
            self.D0, self.S, lmbda, W, opt=optp, nproc=2)
        Dp = bp.solve()
        assert np.linalg.norm(Ds - Dp) < 1e-7
        assert np.abs(
            bs.getitstat().ObjFun[-1] - bp.getitstat().ObjFun[-1] < 1e-7)
