from __future__ import division
from builtins import object
from past.utils import old_div

import pytest
import numpy as np
import scipy.linalg

from sporco.admm import cbpdn
import sporco.linalg as sl



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
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
