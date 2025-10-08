from __future__ import division
from builtins import object

import functools

import numpy as np
import scipy.optimize as optim
from sporco import prox
from sporco import metric


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
    assert np.abs(pf - mf) <= 1e-15
    assert np.linalg.norm(px - mx) <= 1e-14


def prox_test_axes(v, nrm, prx, alpha):
    for ax in range(v.ndim):
        px = prx(v, alpha, axis=ax)
        pf = prox_func(px, v, nrm, alpha, axis=ax)
        mx, mf = prox_solve(v, px, nrm, alpha, axis=ax)
        assert np.abs(pf - mf) <= 1e-15
        assert np.linalg.norm(px - mx) <= 1e-14


def proj_solve(v, v0, f, gamma, axis=None):
    fnc = lambda x: 0.5*np.sum((x.reshape(v.shape) - v)**2)
    cns = ({'type': 'ineq', 'fun': lambda x: gamma - f(x.reshape(v.shape),
                                                       axis=axis)})
    fmn = optim.minimize(fnc, v0.ravel(), method='SLSQP', constraints=cns,
                         options={'maxiter': 3000, 'ftol': 1e-12})
    return fmn.x.reshape(v.shape), fmn.fun


def proj_test(v, nrm, prj, gamma):
    pj = prj(v, gamma)
    pf = 0.5*np.sum((pj - v)**2)
    mx, mf = proj_solve(v, pj, nrm, gamma)
    assert nrm(pj) - gamma <= 1e-12
    assert pf - mf <= 2e-6
    assert np.linalg.norm(pj - mx) <= 1e-3



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)
        self.V0 = np.random.randn(16, 1)
        self.V1 = np.random.randn(6, 5)
        self.V2 = np.random.randn(5, 4, 3)
        self.alpha = [1e-2, 1e-1, 1e0, 1e1]
        self.gamma = [1e-2, 1e-1, 1e0, 1e1]


    def test_01(self):
        nrm = prox.norm_l0
        prx = prox.prox_l0
        for alpha in self.alpha:
            prox_test(self.V1, nrm, prx, alpha)


    def test_02(self):
        nrm = prox.norm_l1
        prx = prox.prox_l1
        for alpha in self.alpha:
            prox_test(self.V1, nrm, prx, alpha)


    def test_06(self):
        nrm = prox.norm_l1
        prj = prox.proj_l1
        for gamma in self.gamma:
            proj_test(self.V1, nrm, prj, gamma)


    def test_07(self):
        nrm = prox.norm_l2
        prx = prox.prox_l2
        for alpha in self.alpha:
            prox_test(self.V1, nrm, prx, alpha)


    def test_08(self):
        nrm = prox.norm_l2
        prj = prox.proj_l2
        for gamma in self.gamma:
            proj_test(self.V1, nrm, prj, gamma)


    def test_09(self):
        for beta in [0.0, 0.5, 1.0]:
            nrm = functools.partial(prox.norm_dl1l2, beta=beta)
            prx = functools.partial(prox.prox_dl1l2, beta=beta)
            for alpha in self.alpha:
                prox_test(self.V1, nrm, prx, alpha)


    def test_10(self):
        for beta in [0.0, 0.5, 1.0]:
            for alpha in self.alpha:
                pV1 = prox.prox_dl1l2(self.V1, alpha, beta=beta, axis=1)
                for k in range(pV1.shape[0]):
                    pV1k = prox.prox_dl1l2(self.V1[k], alpha, beta=beta)
                    assert metric.mse(pV1[k], pV1k) < 1e-12


    def test_11(self):
        for beta in [0.0, 0.5, 1.0]:
            for alpha in self.alpha:
                pV1 = prox.prox_dl1l2(self.V1, alpha, beta=beta, axis=0)
                for k in range(pV1.shape[1]):
                    pV1k = prox.prox_dl1l2(self.V1[:, k], alpha, beta=beta)
                    assert metric.mse(pV1[:, k], pV1k) < 1e-12

    def test_13(self):
        nrm = prox.norm_l21
        prx = prox.prox_l2
        for alpha in self.alpha:
            prox_test_axes(self.V1, nrm, prx, alpha)
            prox_test_axes(self.V2, nrm, prx, alpha)


    def test_14(self):
        assert np.sum(np.abs(prox.prox_sl1l2(self.V1, 1e-2, 1e-2))) > 0
        assert prox.norm_nuclear(self.V1) > 0


    def test_19(self):
        V2_a2, rsi = prox.ndto2d(self.V2, axis=2)
        V2_r = prox.ndfrom2d(V2_a2, rsi)
        assert metric.mse(self.V2, V2_r) < 1e-14
        V2_a1, rsi = prox.ndto2d(self.V2, axis=1)
        V2_r = prox.ndfrom2d(V2_a1, rsi)
        assert metric.mse(self.V2, V2_r) < 1e-14
        V2_a0, rsi = prox.ndto2d(self.V2, axis=0)
        V2_r = prox.ndfrom2d(V2_a0, rsi)
        assert metric.mse(self.V2, V2_r) < 1e-14
