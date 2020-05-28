from __future__ import division
from builtins import object

import pickle
import pytest
try:
    import cupy as cp
    try:
        cp.cuda.Device(0).compute_capability
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("GPU device inaccessible", allow_module_level=True)
except ImportError:
    pytest.skip("cupy not installed", allow_module_level=True)


from sporco.cupy.admm import cbpdn
import sporco.cupy.linalg as sl
from sporco.cupy.util import list2array



class TestSet01(object):

    def setup_method(self, method):
        cp.random.seed(12345)