from __future__ import division
from builtins import object

import numpy as np

from sporco.dictlrn import wbpdndl


class TestSet01(object):

    def setup_method(self, method):
        N = 8
        M = 4
        K = 8
        self.D0 = np.random.randn(N, M)
        self.S = np.random.randn(N, K)
        self.W = np.abs(np.random.randn(N, K))


    def test_01(self):
        lmbda = 1e-1
        try:
            b = wbpdndl.WeightedBPDNDictLearn(self.D0, self.S, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        lmbda = 1e-1
        try:
            b = wbpdndl.WeightedBPDNDictLearn(self.D0, self.S, lmbda, W=self.W)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        lmbda = 1e-1
        opt = wbpdndl.WeightedBPDNDictLearn.Options({'AccurateDFid': True,
                                                    'MaxMainIter': 10})
        try:
            b = wbpdndl.WeightedBPDNDictLearn(self.D0, self.S, lmbda,
                                              W=self.W, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
