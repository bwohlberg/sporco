from __future__ import division
from builtins import object

import numpy as np

from sporco import signal


def fn(prm):
    x = prm[0]
    return (x - 0.1)**2


def fnv(prm):
    x = prm[0]
    return ((x - 0.1)**2, (x - 0.5)**2)


class TestSet01(object):

    def test_01(self):
        img = np.random.randn(64, 64)
        imgn = signal.spnoise(img, 0.5)
        assert imgn.shape == img.shape


    def test_02(self):
        msk = signal.rndmask((16, 17), 0.25)
        assert msk.shape == (16, 17)


    def test_03(self):
        msk = signal.rndmask((16, 17), 0.25, dtype=np.float32)
        assert msk.dtype == np.float32


    def test_04(self):
        rgb = np.random.randn(64, 64, 3)
        gry = signal.rgb2gray(rgb)
        assert gry.shape == rgb.shape[0:2]


    def test_05(self):
        img = np.random.randn(64, 64)
        iml, imh = signal.tikhonov_filter(img, 5.0)
        assert iml.shape == img.shape and imh.shape == img.shape


    def test_06(self):
        img = np.random.randn(16, 16, 16)
        iml, imh = signal.tikhonov_filter(img, 2.0, npd=8)
        assert iml.shape == img.shape and imh.shape == img.shape


    def test_08(self):
        shape = (7, 5, 6)
        g = signal.gaussian(shape)
        assert g.shape == shape


    def test_09(self):
        s = np.random.rand(16, 17, 3)
        scn, smn, snrm = signal.local_contrast_normalise(s)
        assert np.linalg.norm(snrm * scn + smn - s) < 1e-7
