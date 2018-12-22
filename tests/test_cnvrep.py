from __future__ import division
from builtins import object

import numpy as np
from sporco import cnvrep


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)



    def test_01(self):
        N = 32
        M = 16
        L = 8
        D = np.random.randn(L, L, M)
        S = np.random.randn(N, N)
        cri = cnvrep.CSC_ConvRepIndexing(D, S, dimK=0)
        assert cri.M == M
        assert cri.K == 1
        assert cri.Nv == (N, N)
        assert str(cri) != ''
        W = np.random.randn(N, N)
        assert cnvrep.l1Wshape(W, cri) == (N, N, 1, 1, 1)
        W = np.random.randn(N, N, M)
        assert cnvrep.l1Wshape(W, cri) == (N, N, 1, 1, M)
        W = np.random.randn(N, N, 1, 1, M)
        assert cnvrep.l1Wshape(W, cri) == (N, N, 1, 1, M)



    def test_02(self):
        N = 32
        M = 16
        L = 8
        K = 4
        D = np.random.randn(L, L, M)
        S = np.random.randn(N, N, K)
        cri = cnvrep.CSC_ConvRepIndexing(D, S, dimK=1)
        assert cri.M == M
        assert cri.K == K
        assert cri.Nv == (N, N)



    def test_03(self):
        N = 32
        M = 16
        L = 8
        C = 3
        D = np.random.randn(L, L, M)
        S = np.random.randn(N, N, C)
        cri = cnvrep.CSC_ConvRepIndexing(D, S, dimK=0)
        assert cri.M == M
        assert cri.K == 1
        assert cri.C == 3
        assert cri.Nv == (N, N)



    def test_04(self):
        dsz = (8, 8, 32)
        ds = cnvrep.DictionarySize(dsz)
        assert ds.nchn == 1
        assert ds.nflt == 32
        assert str(ds) != ''



    def test_05(self):
        dsz = ((8, 8, 16), (12, 12, 32))
        ds = cnvrep.DictionarySize(dsz)
        assert ds.nchn == 1
        assert ds.nflt == 48



    def test_06(self):
        dsz = ((8, 8, 3, 16), (12, 12, 3, 32))
        ds = cnvrep.DictionarySize(dsz)
        assert ds.nchn == 3
        assert ds.nflt == 48



    def test_07(self):
        dsz = (((5, 5, 2, 8), (7, 7, 1, 8)),
               ((9, 9, 2, 16), (10, 10, 1, 16)))
        ds = cnvrep.DictionarySize(dsz)
        assert ds.nchn == 3
        assert ds.nflt == 24



    def test_08(self):
        N = 32
        M = 16
        L = 8
        dsz = (L, L, M)
        S = np.random.randn(N, N)
        cri = cnvrep.CDU_ConvRepIndexing(dsz, S, dimK=0)
        assert cri.M == M
        assert cri.K == 1
        assert cri.Nv == (N, N)
        assert str(cri) != ''
        W = np.random.randn(N, N)
        assert cnvrep.mskWshape(W, cri) == (N, N, 1, 1, 1)



    def test_09(self):
        dsz = (8, 8, 32)
        u = np.zeros((16, 16, 32))
        u[0:8, 0:8, 0:16] = 1.0
        v = cnvrep.zeromean(u, dsz)
        assert np.sum(np.abs(v)) == 0.0



    def test_10(self):
        dsz = ((8, 8, 16), (12, 12, 32))
        u = np.zeros((24, 24, 48))
        u[0:8, 0:8, 0:16] = 1.0
        u[0:12, 0:12, 16:] = 1.0
        v = cnvrep.zeromean(u, dsz)
        assert np.sum(np.abs(v)) == 0.0



    def test_11(self):
        dsz = (((5, 5, 2, 8), (7, 7, 1, 8)),
               ((9, 9, 2, 16), (10, 10, 1, 16)))
        u = np.zeros((16, 16, 3, 24))
        u[0:5, 0:5, 0:2, 0:8] = 1.0
        u[0:7, 0:7, 2, 0:8] = 1.0
        u[0:9, 0:9, 0:2, 8:] = 1.0
        u[0:10, 0:10, 2, 8:] = 1.0
        v = cnvrep.zeromean(u, dsz)
        assert np.sum(np.abs(v)) == 0.0



    def test_12(self):
        dsz = (8, 8, 32)
        u = np.zeros((16, 16, 32))
        u[0:8, 0:8, 0:16] = 1.0
        v = cnvrep.bcrop(u, dsz)
        assert v.shape == dsz



    def test_13(self):
        dsz = ((8, 8, 16), (12, 12, 32))
        u = np.zeros((24, 24, 48))
        u[0:8, 0:8, 0:16] = 1.0
        u[0:12, 0:12, 16:] = 1.0
        v = cnvrep.bcrop(u, dsz)
        assert v.shape == (12, 12, 48)



    def test_14(self):
        dsz = (((5, 5, 2, 8), (7, 7, 1, 8)),
               ((9, 9, 2, 16), (10, 10, 1, 16)))
        u = np.zeros((16, 16, 3, 24))
        u[0:5, 0:5, 0:2, 0:8] = 1.0
        u[0:7, 0:7, 2, 0:8] = 1.0
        u[0:9, 0:9, 0:2, 8:] = 1.0
        u[0:10, 0:10, 2, 8:] = 1.0
        v = cnvrep.bcrop(u, dsz)
        assert v.shape == (10, 10, 3, 24)



    def test_15(self):
        dsz = (3, 3, 1)
        Nv = (6, 6)
        x = np.ones((6, 6, 1))
        fn = cnvrep.getPcn(dsz, Nv)
        y = fn(x)
        assert np.sum(y) == 3.0
        y[0:3, 0:3] = 0
        assert np.sum(y) == 0.0



    def test_16(self):
        dsz = (3, 3, 1)
        Nv = (6, 6)
        x = np.ones((6, 6, 1))
        fn = cnvrep.getPcn(dsz, Nv, crp=True)
        y = fn(x)
        assert np.sum(y) == 3.0
        assert y.shape == (3, 3, 1)



    def test_17(self):
        dsz = (3, 3, 1)
        Nv = (6, 6)
        x = np.ones((6, 6, 1))
        x[0] = 2
        fn = cnvrep.getPcn(dsz, Nv, zm=True)
        y = fn(x)
        assert np.all(y[0:3, 0:3] != 0.0)
        assert np.abs(np.sum(y)) < 1e-14
        y[0:3, 0:3] = 0
        assert np.sum(np.abs(y)) == 0.0



    def test_18(self):
        dsz = (3, 3, 1)
        Nv = (6, 6)
        x = np.ones((6, 6, 1))
        x[0] = 2
        fn = cnvrep.getPcn(dsz, Nv, crp=True, zm=True)
        y = fn(x)
        assert np.all(y != 0.0)
        assert np.abs(np.sum(y)) < 1e-14
        assert y.shape == (3, 3, 1)
