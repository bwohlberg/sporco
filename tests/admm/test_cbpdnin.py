from __future__ import division
from builtins import object

import numpy as np

from sporco.admm import cbpdnin



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        D = np.random.randn(4, 4, 32)
        s = np.random.randn(8, 8)

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
        D = np.random.randn(4, 4, 32)
        s = np.random.randn(8, 8)
        Wg = np.append(np.eye(16), np.eye(16), axis=-1)
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
        D = np.random.randn(4, 4, 32)
        s = np.random.randn(8, 8, 3)
        Wg = np.append(np.eye(16), np.eye(16), axis=-1)
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


    def test_04(self):
        D = np.random.randn(4, 4, 32)
        s = np.random.randn(8, 8)
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


    def test_05(self):
        D = np.random.randn(4, 4, 32)
        s = np.random.randn(8, 8)
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


    def test_06(self):
        D = np.random.randn(4, 32)
        s = np.random.randn(64)
        Wg = np.append(np.eye(16), np.eye(16), axis=-1)
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
