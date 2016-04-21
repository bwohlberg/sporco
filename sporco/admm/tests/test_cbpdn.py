import pytest

import numpy as np
import scipy.linalg
from sporco.admm import cbpdn
from sporco import util
import sporco.linalg as sl



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        rho = 1e-1
        N = 64
        M = 32
        K = 1
        D = np.random.randn(8, 8, 1, M, 1)
        Df = sl.fftn(D, (N, N), (0,1))
        X = np.random.randn(N, N, 1, M, K)
        Xf = sl.fftn(X, None, (0,1))
        Sf = Df * Xf
        S = sl.ifftn(Sf, None, (0,1))
        DSf = np.conj(Df) * Sf
        Dop = lambda x: np.sum(Df * x, axis=2, keepdims=True)
        DHop = lambda x: np.conj(Df) * x
        Zf = (DHop(Dop(Xf)) + rho*Xf - DSf)/rho
        Xfslv = sl.solvedbi_sm(Df, rho, DSf + rho*Zf, None, 2)
        assert(sl.rrs(DHop(Dop(Xfslv)) + rho*Xfslv, DSf + rho*Zf) < 1e-11)
