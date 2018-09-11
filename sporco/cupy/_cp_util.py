# -*- coding: utf-8 -*-
# Copyright (C) 2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Utility functions for cupy module that are active when cupy is
available
"""


import functools
import numpy as np
from sporco.cupy import cp


def array_module(*args):
    """An alias for :func:`cupy.get_array_module`."""

    return cp.get_array_module(*args)


def np2cp(u):
    """Convert a numpy ndarray to a cupy array. This function is an
    alias for :func:`cupy.asarray`.
    """

    return cp.asarray(u)


def cp2np(u):
    """Convert a cupy array to a numpy ndarray. This function is an
    alias for :func:`cupy.asnumpy`.
    """

    return cp.asnumpy(u)


def cupy_wrapper(func):
    """A wrapper function that converts numpy ndarray arguments to cupy
    arrays, and convert any cupy arrays returned by the wrapped
    function into numpy ndarrays.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = list(args)
        for n, a in enumerate(args):
            if isinstance(a, np.ndarray):
                args[n] = cp.asarray(a)
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                kwargs[k] = cp.asarray(v)
        rtn = func(*args, **kwargs)
        if isinstance(rtn, (list, tuple)):
            for n, a in enumerate(rtn):
                if isinstance(a, cp.core.core.ndarray):
                    rtn[n] = cp.asnumpy(a)
        else:
            if isinstance(rtn, cp.core.core.ndarray):
                rtn = cp.asnumpy(rtn)
        return rtn
    return wrapped
