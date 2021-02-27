from builtins import object

import numpy as np

from sporco.admm import ppp
import sporco.prox as sp
import sporco.metric as sm


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 64
        lmbda = 0.1
        s = np.random.randn(N, 1)

        def f(x):
            return 0.5 * np.linalg.norm((x - s).ravel())**2

        def proxf(x, rho):
            return s / (1 + rho) + (rho / (1 + rho)) * x

        def proxg(x, rho):
            return sp.prox_l1(x, lmbda / rho)

        opt = ppp.PPP.Options({'Verbose': False, 'RelStopTol': 1e-5,
                               'MaxMainIter': 100, 'rho': 8e-1})
        b = ppp.PPP(s.shape, f, proxf, proxg, opt=opt)
        xppp = b.solve()
        xdrct = sp.prox_l1(s, lmbda)
        assert sm.mse(xdrct, xppp) < 1e-9


    def test_02(self):
        N = 64
        lmbda = 0.1
        s = np.random.randn(N, 1)

        def proxf(x, rho):
            return s / (1 + rho) + (rho / (1 + rho)) * x

        def proxg(x, rho):
            return sp.prox_l1(x, lmbda / rho)

        opt = ppp.PPPConsensus.Options({'Verbose': False, 'RelStopTol': 1e-5,
                                        'MaxMainIter': 100, 'rho': 8e-1})
        b = ppp.PPPConsensus(s.shape, (proxf, proxg), opt=opt)
        xce = b.solve()
        xdrct = sp.prox_l1(s, lmbda)
        assert sm.mse(xdrct, xce) < 1e-9


    def test_03(self):
        N = 64
        lmbda = 0.1
        s = np.random.randn(N, 1)

        def proxf(x, rho):
            return s / (1 + rho) + (rho / (1 + rho)) * x

        def proxg(x, rho):
            return sp.prox_l1(x, lmbda / rho)

        opt = ppp.PPPConsensus.Options({'Verbose': False, 'RelStopTol': 1e-5,
                                        'MaxMainIter': 100, 'rho': 8e-1})
        b = ppp.PPPConsensus(s.shape, (proxf,), proxg=proxg, opt=opt)
        xce = b.solve()
        xdrct = sp.prox_l1(s, lmbda)
        assert sm.mse(xdrct, xce) < 1e-9
