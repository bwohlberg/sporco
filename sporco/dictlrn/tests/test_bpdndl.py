from __future__ import division
from builtins import object

import numpy as np

from sporco.dictlrn import bpdndl


class TestSet01(object):

    def setup_method(self, method):
        N = 8
        M = 4
        K = 8
        self.D0 = np.random.randn(N, M)
        self.S = np.random.randn(N, K)


    def test_01(self):
        lmbda = 1e-1
        try:
            b = bpdndl.BPDNDictLearn(self.D0, self.S, lmbda)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        try:
            b = bpdndl.BPDNDictLearn(self.D0, self.S)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        lmbda = 1e-1
        opt = bpdndl.BPDNDictLearn.Options({'AccurateDFid': True,
                                            'MaxMainIter': 10})
        try:
            b = bpdndl.BPDNDictLearn(self.D0, self.S, lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
