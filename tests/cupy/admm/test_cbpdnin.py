from __future__ import division
from builtins import object

import pytest
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).compute_capability
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("GPU device inaccessible", allow_module_level=True)
except ImportError:
    pytest.skip("cupy not installed", allow_module_level=True)

import numpy as np

from sporco.cupy.admm import cbpdnin



class TestSet01(object):

    def setup_method(self, method):
        cp.random.seed(12345)


    def test_01(self):
        D = cp.random.randn(4, 4, 32)
        s = cp.random.randn(8, 8)

        # ConvBPDNInhib class
        opt = cbpdnin.ConvBPDNInhib.Options(
            {'Verbose': False, 'MaxMainIter': 10})

        try:
            b = cbpdnin.ConvBPDNInhib(D, s, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        D = cp.random.randn(4, 4, 32)
        s = cp.random.randn(8, 8)
        Wg = cp.concatenate((cp.eye(16), cp.eye(16)), axis=-1)
        lmbda = 0.1

        # ConvBPDNInhib class
        opt = cbpdnin.ConvBPDNInhib.Options(
            {'Verbose': False, 'MaxMainIter': 10})

        try:
            b = cbpdnin.ConvBPDNInhib(D, s, Wg=Wg, lmbda=lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        D = cp.random.randn(4, 4, 32)
        s = cp.random.randn(8, 8)
        lmbda = 0.1
        gamma = 0.01

        # ConvBPDNInhib class
        opt = cbpdnin.ConvBPDNInhib.Options(
            {'Verbose': False, 'MaxMainIter': 10})

        try:
            b = cbpdnin.ConvBPDNInhib(D, s, lmbda=lmbda, gamma=gamma, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_04(self):
        D = cp.random.randn(4, 4, 32)
        s = cp.random.randn(8, 8)
        lmbda = 0.1
        mu = 0.01

        # ConvBPDNInhib class
        opt = cbpdnin.ConvBPDNInhib.Options(
            {'Verbose': False, 'MaxMainIter': 10})

        try:
            b = cbpdnin.ConvBPDNInhib(D, s, lmbda=lmbda, mu=mu, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_05(self):
        D = cp.random.randn(4, 32)
        s = cp.random.randn(64)
        Wg = np.concatenate((cp.eye(16), cp.eye(16)), axis=-1)
        lmbda = 0.1
        mu = 0.01
        gamma = 0.01

        # ConvBPDNInhib class
        opt = cbpdnin.ConvBPDNInhib.Options(
            {'Verbose': False, 'MaxMainIter': 10})

        try:
            b = cbpdnin.ConvBPDNInhib(
                D, s, Wg=Wg, lmbda=lmbda, mu=mu, gamma=gamma, opt=opt, dimN=1)
            b.solve()
        except Exception as e:
            print(e)
            assert 0
