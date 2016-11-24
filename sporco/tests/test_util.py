from __future__ import division
from builtins import object

import pytest

import numpy as np

from sporco import util


def fn(prm):
    x = prm[0]
    return (x - 0.1)**2


class TestSet01(object):

    def test_01(self):
        D = np.random.randn(64, 64)
        im = util.tiledict(D, sz=(8, 8))


    def test_02(self):
        D = np.random.randn(8, 8, 64)
        im = util.tiledict(D)


    def test_03(self):
        D = np.random.randn(8, 8, 3, 64)
        im = util.tiledict(D)


    def test_04(self):
        img = np.random.randn(64, 64)
        blk = util.imageblocks(img, (8, 8))
    

    def test_05(self):
        rgb = np.random.randn(64, 64, 3)
        gry = util.rgb2gray(rgb)


    def test_06(self):
        img = np.random.randn(64, 64)
        imgn = util.spnoise(img, 0.5)


    def test_07(self):
        img = np.random.randn(64, 64)
        iml, imh = util.tikhonov_filter(img, 5.0)


    def test_08(self):
        x = np.linspace(-1, 1, 21)
        sprm, sfvl, fvmx, sidx = util.grid_search(fn, (x,))
        assert(np.abs(sprm[0] - 0.1) < 1e-14)
        assert(sidx[0] == 11)


    def test_09(self):
        D = util.convdicts()['G:12x12x72']
        assert(D.shape == (12,12,72))


    def test_10(self):
        ei = util.ExampleImages()
        nm = ei.names()
        assert(len(nm) > 0)
        im = ei.image('barbara')
        assert(im.shape == (576,720,3))
