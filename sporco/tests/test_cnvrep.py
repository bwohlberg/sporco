from __future__ import division
from builtins import object

import pytest

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
        assert(cri.M == M)
        assert(cri.K == 1)
        assert(cri.Nv == (N, N))



    def test_02(self):
        N = 32
        M = 16
        L = 8
        K = 4
        D = np.random.randn(L, L, M)
        S = np.random.randn(N, N, K)
        cri = cnvrep.CSC_ConvRepIndexing(D, S, dimK=1)
        assert(cri.M == M)
        assert(cri.K == K)
        assert(cri.Nv == (N, N))
