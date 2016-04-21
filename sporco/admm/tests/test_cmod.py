import pytest

import numpy as np
from sporco.admm import cmod
import sporco.linalg as sl



class TestSet01(object):

    def setup_method(self, method):
        pass



    def test_01(self):
        rho = 1e-1
        N = 64
        M = 128
        K = 32
        D = np.random.randn(N, M)
        X = np.random.randn(M, K)
        S = D.dot(X)
        Z = (D.dot(X).dot(X.T) + rho*D - S.dot(X.T))/rho
        lu, piv = cmod.factorise(X, rho)
        Dslv = cmod.linsolve(X, rho, lu, piv, S.dot(X.T) + rho*Z)
        assert(sl.rrs(Dslv.dot(X).dot(X.T) + rho*Dslv,
                        S.dot(X.T) + rho*Z) < 1e-11)


    def test_02(self):
        rho = 1e-1
        N = 128
        M = 64
        K = 32
        D = np.random.randn(N, M)
        X = np.random.randn(M, K)
        S = D.dot(X)
        Z = (D.dot(X).dot(X.T) + rho*D - S.dot(X.T))/rho
        lu, piv = cmod.factorise(X, rho)
        Dslv = cmod.linsolve(X, rho, lu, piv, S.dot(X.T) + rho*Z)
        assert(sl.rrs(Dslv.dot(X).dot(X.T) + rho*Dslv,
                        S.dot(X.T) + rho*Z) < 1e-11)
