from __future__ import division
from builtins import object

import pytest
try:
    from mpi4py import MPI
except:
    pytest.skip("mpi4py not installed", allow_module_level=True)
import numpy as np

from sporco import mpiutil


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
        self.comm = MPI.COMM_WORLD


    def test_01(self):
        x = np.linspace(-1, 1, 21)
        sprm, sfvl, fvmx, sidx = mpiutil.grid_search(fn, (x,), self.comm)
        assert np.abs(sprm[0] - 0.1) < 1e-14
        assert sidx[0] == 11


    def test_02(self):
        x = np.linspace(-1, 1, 21)
        sprm, sfvl, fvmx, sidx = mpiutil.grid_search(fnnan, (x,), self.comm)
        assert np.abs(sprm[0] - 0.1) < 1e-14
        assert sidx[0] == 11


    def test_03(self):
        x = np.linspace(-1, 1, 21)
        sprm, sfvl, fvmx, sidx = mpiutil.grid_search(fnv, (x,), self.comm)
        assert np.abs(sprm[0][0] - 0.1) < 1e-14
        assert np.abs(sprm[0][1] - 0.5) < 1e-14
        assert sidx[0][0] == 11
        assert sidx[0][1] == 15
