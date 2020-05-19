from __future__ import division
from builtins import object

import pickle
import numpy as np

from sporco.admm import cbpdn
import sporco.linalg as sl



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)

    def test_37(self):
        Nr = 16
        Nc = 17
        Nd = 5
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(Nr, Nc)
        lmbda = 1e-1
        mu = 1e-2
        try:
            b = cbpdn.ConvBPDNLatInh(D, s, None, None, lmbda, mu)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_38(self):
        N = 16
        Nd = 5
        K = 2
        M = 4
        D = np.random.randn(Nd, Nd, M)
        s = np.random.randn(N, N, K)
        dt = np.float32
        opt = cbpdn.ConvBPDNLatInh.Options({'Verbose': False,
                                            'LinSolveCheck': True, 'MaxMainIter': 20,
                                            'AutoRho': {'Enabled': True}, 'DataType': dt})
        lmbda = 1e-1
        mu = 1e-2
        b = cbpdn.ConvBPDNLatInh(D, s, Wg, Wh, lmbda, mu, opt=opt)
        b.solve()
        assert b.X.dtype == dt
        assert b.Y.dtype == dt
        assert b.U.dtype == dt