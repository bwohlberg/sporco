from __future__ import division
from builtins import object

import pytest

import numpy as np
import scipy.optimize as optim
from sporco import prox


def prox_func(x, v, f, alpha, axis=None):
    return 0.5*np.sum((x.reshape(v.shape) - v)**2) + \
        alpha*f(x.reshape(v.shape), axis=axis)


def prox_solve(v, v0, f, alpha, axis=None):
    fnc = lambda x: prox_func(x, v, f, alpha, axis)
    fmn = optim.minimize(fnc, v0.ravel(), method='Nelder-Mead')
    return fmn.x.reshape(v.shape), fmn.fun


def prox_test(v, nrm, prx, alpha):
    px = prx(v, alpha)
    pf = prox_func(px, v, nrm, alpha)
    mx, mf = prox_solve(v, px, nrm, alpha)
    assert(np.abs(pf - mf) <= 1e-15)
    assert(np.linalg.norm(px - mx) <= 1e-14)


def prox_test_axes(v, nrm, prx, alpha):
    for ax in range(v.ndim):
        px = prx(v, alpha, axis=ax)
        pf = prox_func(px, v, nrm, alpha, axis=ax)
        mx, mf = prox_solve(v, px, nrm, alpha, axis=ax)
        assert(np.abs(pf - mf) <= 1e-15)
        assert(np.linalg.norm(px - mx) <= 1e-14)


def proj_solve(v, v0, f, gamma, axis=None):
    fnc = lambda x: 0.5*np.sum((x.reshape(v.shape) - v)**2)
    cns = ({'type': 'ineq', 'fun': lambda x: gamma - f(x.reshape(v.shape),
                                                       axis=axis)})
    fmn = optim.minimize(fnc, v0.ravel(), method='SLSQP', constraints=cns,
                         options={'maxiter': 2000})
    return fmn.x.reshape(v.shape), fmn.fun


def proj_test(v, nrm, prj, gamma):
    pj = prj(v, gamma)
    pf = 0.5*np.sum((pj - v)**2)
    mx, mf = proj_solve(v, pj, nrm, gamma)
    assert(nrm(pj) - gamma <= 1e-14)
    assert(np.abs(pf - mf) <= 1e-6)
    assert(np.linalg.norm(pj - mx) <= 1e-4)



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)
        self.V1 = np.random.randn(6, 5)
        self.V2 = np.random.randn(5, 4, 3)
        self.alpha = 2.0
        self.gamma = 2.0


    def test_01(self):
        nrm = prox.norm_l0
        prx = prox.prox_l0
        prox_test(self.V1, nrm, prx, self.alpha)


    def test_02(self):
        nrm = prox.norm_l1
        prx = prox.prox_l1
        prox_test(self.V1, nrm, prx, self.alpha)


    def test_03(self):
        nrm = prox.norm_l1
        prj = prox.proj_l1
        proj_test(self.V1, nrm, prj, self.gamma)


    def test_04(self):
        nrm = prox.norm_l2
        prx = prox.prox_l2
        prox_test(self.V1, nrm, prx, self.alpha)


    def test_05(self):
        nrm = prox.norm_l2
        prj = prox.proj_l2
        proj_test(self.V1, nrm, prj, self.gamma)


    def test_06(self):
        nrm = prox.norm_l21
        prx = prox.prox_l2
        prox_test_axes(self.V1, nrm, prx, self.alpha)
        prox_test_axes(self.V2, nrm, prx, self.alpha)


    def test_12(self):
        assert(np.sum(np.abs(prox.prox_l1l2(self.V1, 1e-2, 1e-2))) > 0)
        assert(prox.norm_nuclear(self.V1) > 0)

