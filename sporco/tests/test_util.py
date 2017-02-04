from __future__ import division
from builtins import object

import pytest

import numpy as np
import collections

from sporco import util


def fn(prm):
    x = prm[0]
    return (x - 0.1)**2


def fnv(prm):
    x = prm[0]
    return ((x - 0.1)**2, (x - 0.5)**2)


class TestSet01(object):

    def test_01(self):
        nt = collections.namedtuple('NT', ('A', 'B', 'C'))
        t0 = nt(0, 1, 2)
        t0a = util.ntpl2array(t0)
        t1 = util.array2ntpl(t0a)
        

    def test_02(self):
        D = np.random.randn(64, 64)
        im = util.tiledict(D, sz=(8, 8))


    def test_03(self):
        D = np.random.randn(8, 8, 64)
        im = util.tiledict(D)


    def test_04(self):
        D = np.random.randn(8, 8, 64)
        im = util.tiledict(D, sz=((6, 6, 32), (8, 8, 32)))


    def test_05(self):
        D = np.random.randn(8, 8, 3, 64)
        im = util.tiledict(D)


    def test_06(self):
        img = np.random.randn(64, 64)
        blk = util.imageblocks(img, (8, 8))
    

    def test_07(self):
        rgb = np.random.randn(64, 64, 3)
        gry = util.rgb2gray(rgb)


    def test_08(self):
        img = np.random.randn(64, 64)
        imgn = util.spnoise(img, 0.5)


    def test_09(self):
        img = np.random.randn(64, 64)
        iml, imh = util.tikhonov_filter(img, 5.0)


    def test_10(self):
        img = np.random.randn(16, 16, 16)
        iml, imh = util.tikhonov_filter(img, 2.0, npd=8)


    def test_11(self):
        x = np.linspace(-1, 1, 21)
        sprm, sfvl, fvmx, sidx = util.grid_search(fn, (x,))
        assert(np.abs(sprm[0] - 0.1) < 1e-14)
        assert(sidx[0] == 11)


    def test_12(self):
        x = np.linspace(-1, 1, 21)
        sprm, sfvl, fvmx, sidx = util.grid_search(fnv, (x,))
        assert(np.abs(sprm[0][0] - 0.1) < 1e-14)
        assert(np.abs(sprm[0][1] - 0.5) < 1e-14)
        assert(sidx[0][0] == 11)
        assert(sidx[0][1] == 15)


    def test_13(self):
        D = util.convdicts()['G:12x12x72']
        assert(D.shape == (12,12,72))


    def test_14(self):
        ei = util.ExampleImages()
        nm = ei.names()
        assert(len(nm) > 0)
        im = ei.image('barbara')
        assert(im.shape == (576,720,3))
        im = ei.image('barbara', scaled=True, dtype=np.float32, zoom=0.5)
