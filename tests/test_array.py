from __future__ import division
from builtins import object

import collections

import numpy as np
from sporco import array


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        nt = collections.namedtuple('NT', ('A', 'B', 'C'))
        t0 = nt(0, 1, 2)
        t0a = array.ntpl2array(t0)
        t1 = array.array2ntpl(t0a)
        assert t0 == t1


    def test_02(self):
        nt = collections.namedtuple('NT', ('A', 'B', 'C'))
        lst = [nt(0, 1, 2), nt(3, 4, 5)]
        lsttp = array.transpose_ntpl_list(lst)
        assert lst[0].A == lsttp.A[0]


    def test_03(self):
        img = np.random.randn(64, 64)
        blk = array.extract_blocks(img, (8, 8))
        assert blk.shape == (8, 8, 3249)


    def test_04(self):
        A = np.random.rand(4, 5, 6, 7, 3)
        blksz = (2, 3, 2)
        stpsz = (2, 1, 2)
        A_blocks = array.extract_blocks(A, blksz, stpsz)
        A_recon = array.combine_blocks(A_blocks, A.shape, stpsz, np.median)
        assert np.allclose(np.where(np.isnan(A_recon), np.nan, A),
                           A_recon, equal_nan=True)


    def test_05(self):
        A = np.random.rand(4, 5, 6, 7, 3)
        blksz = (2, 3, 2)
        stpsz = (2, 1, 2)
        A_blocks = array.extract_blocks(A, blksz, stpsz)
        noise = np.random.rand(*A_blocks.shape)
        A_average_recon = array.average_blocks(A_blocks + noise, A.shape,
                                               stpsz)
        A_combine_recon = array.combine_blocks(A_blocks + noise, A.shape,
                                              stpsz, np.mean)
        assert np.allclose(A_combine_recon, A_average_recon, equal_nan=True)


    def test_06(self):
        x = np.arange(20).reshape((4, 5))
        y = array.rolling_window(x, (3, 3))
        assert y.shape == (3, 3, 2, 3)
        assert y[-1, -1, -1, -1] == 19


    def test_07(self):
        x = np.arange(20).reshape((4, 5))
        y = array.subsample_array(x, (2, 2), pad=True)
        assert y.shape == (2, 2, 2, 3)
        assert y[0, 0, -1, -1] == 14
