from builtins import object

import numpy as np

from sporco.pgm import ppp
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

        def gradf(x):
            return x - s

        def proxg(x, L):
            return sp.prox_l1(x, lmbda / L)

        opt = ppp.PPP.Options({'Verbose': False, 'RelStopTol': 1e-5,
                               'MaxMainIter': 100, 'L': 9e-1})
        b = ppp.PPP(s.shape, f, gradf, proxg, opt=opt)
        xppp = b.solve()
        xdrct = sp.prox_l1(s, lmbda)
        assert sm.mse(xdrct, xppp) < 1e-9
