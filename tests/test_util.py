from __future__ import division
from builtins import object

import pytest

import numpy as np
import os
import platform

from sporco import util


def fn(prm):
    x = prm[0]
    return (x - 0.1)**2


def fnnan(prm):
    x = prm[0]
    if x < 0.0:
        return np.nan
    else:
        return (x - 0.1)**2


def fnv(prm):
    x = prm[0]
    return ((x - 0.1)**2, (x - 0.5)**2)


class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_03(self):
        D = np.random.randn(64, 64)
        im = util.tiledict(D, sz=(8, 8))


    def test_04(self):
        D = np.random.randn(8, 8, 64)
        im = util.tiledict(D)


    def test_05(self):
        D = np.random.randn(8, 8, 64)
        im = util.tiledict(D, sz=((6, 6, 32), (8, 8, 32)))


    def test_06(self):
        D = np.random.randn(8, 8, 3, 64)
        im = util.tiledict(D)


    def test_12(self):
        x = np.linspace(-1, 1, 21)
        sprm, sfvl, fvmx, sidx = util.grid_search(fn, (x,))
        assert np.abs(sprm[0] - 0.1) < 1e-14
        assert sidx[0] == 11


    def test_13(self):
        x = np.linspace(-1, 1, 21)
        sprm, sfvl, fvmx, sidx = util.grid_search(fnnan, (x,))
        assert np.abs(sprm[0] - 0.1) < 1e-14
        assert sidx[0] == 11


    def test_14(self):
        x = np.linspace(-1, 1, 21)
        sprm, sfvl, fvmx, sidx = util.grid_search(fnv, (x,))
        assert np.abs(sprm[0][0] - 0.1) < 1e-14
        assert np.abs(sprm[0][1] - 0.5) < 1e-14
        assert sidx[0][0] == 11
        assert sidx[0][1] == 15


    def test_15(self):
        D = util.convdicts()['G:12x12x72']
        assert D.shape == (12, 12, 72)


    def test_16(self):
        ei = util.ExampleImages()
        im = ei.images()
        assert len(im) > 0
        gp = ei.groups()
        assert len(gp) > 0
        gi = ei.groupimages(gp[0])
        assert len(gi) > 0
        im1 = ei.image('sail.png')
        im2 = ei.image('sail.png', scaled=True, dtype=np.float32,
                       idxexp=np.s_[:, 10:-10], zoom=0.5)
        im3 = ei.image('sail.png', dtype=np.float32,
                       idxexp=np.s_[:, 10:-10], zoom=2.0, gray=True)


    def test_17(self):
        pth = os.path.join(os.path.dirname(util.__file__), 'data')
        ei = util.ExampleImages(pth=pth)
        im = ei.images()
        assert len(im) > 0


    def test_18(self):
        t = util.Timer()
        t.start()
        t0 = t.elapsed()
        t.stop()
        t1 = t.elapsed()
        assert t0 >= 0.0
        assert t1 >= t0
        assert len(t.__str__()) > 0
        assert len(t.labels()) > 0


    def test_19(self):
        t = util.Timer('a')
        t.start(['a', 'b'])
        t0 = t.elapsed('a')
        t.stop('a')
        t.stop('b')
        t.stop(['a', 'b'])
        assert t.elapsed('a') >= 0.0
        assert t.elapsed('b') >= 0.0
        assert t.elapsed('a', total=False) == 0.0


    def test_20(self):
        t = util.Timer('a')
        t.start(['a', 'b'])
        t.reset('a')
        assert t.elapsed('a') == 0.0
        t.reset('all')
        assert t.elapsed('b') == 0.0


    def test_21(self):
        t = util.Timer()
        with util.ContextTimer(t):
            t0 = t.elapsed()
        assert t.elapsed() >= 0.0


    def test_22(self):
        t = util.Timer()
        t.start()
        with util.ContextTimer(t, action='StopStart'):
            t0 = t.elapsed()
        t.stop()
        assert t.elapsed() >= 0.0


    def test_23(self):
        with pytest.raises(ValueError):
            dat = util.netgetdata('http://devnull', maxtry=0)


    def test_24(self):
        with pytest.raises(util.urlerror.URLError):
            dat = util.netgetdata('http://devnull')


    def test_25(self):
        val = util.in_ipython()
        assert val is True or val is False


    def test_26(self):
        val = util.in_notebook()
        assert val is True or val is False


    @pytest.mark.skipif(platform.system() == 'Windows',
                        reason='Feature not supported under Windows')
    def test_27(self):
        assert util.idle_cpu_count() >= 1
