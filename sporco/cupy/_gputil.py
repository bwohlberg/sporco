# -*- coding: utf-8 -*-
# Copyright (C) 2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Additional utility functions for cupy module that are active when
CuPy and GPUtil are available
"""


from collections import namedtuple
from sporco.cupy import cp
import GPUtil


def gpu_info():
    """Return a list of namedtuples representing attributes of each GPU
    device.
    """

    GPUInfo = namedtuple('GPUInfo', ['name', 'driver', 'totalmem', 'freemem'])
    gpus = GPUtil.getGPUs()
    info = []
    for g in gpus:
        info.append(GPUInfo(g.name, g.driver, g.memoryTotal, g.memoryFree))
    return info


def gpu_load(wproc=0.5, wmem=0.5):
    """Return a list of namedtuples representing the current load for
    each GPU device. The processor and memory loads are fractions
    between 0 and 1. The weighted load represents a weighted average
    of processor and memory loads using the parameters `wproc` and
    `wmem` respectively.
    """

    GPULoad = namedtuple('GPULoad', ['processor', 'memory', 'weighted'])
    gpus = GPUtil.getGPUs()
    load = []
    for g in gpus:
        wload = (wproc * g.load + wmem * g.memoryUtil) / (wproc + wmem)
        load.append(GPULoad(g.load, g.memoryUtil, wload))
    return load


def device_by_load(wproc=0.5, wmem=0.5):
    """Get a list of GPU device ids ordered by increasing weighted
    average of processor and memory load.
    """

    gl = gpu_load(wproc=wproc, wmem=wmem)
    # return np.argsort(np.asarray(gl)[:, -1]).tolist()
    return [idx for idx, load in sorted(enumerate(
        [g.weighted for g in gl]), key=(lambda x: x[1]))]


def select_device_by_load(wproc=0.5, wmem=0.5):
    """Set the current device for cupy as the device with the lowest
    weighted average of processor and memory load.
    """

    ids = device_by_load(wproc=wproc, wmem=wmem)
    cp.cuda.Device(ids[0]).use()
    return ids[0]


def available_gpu(*args, **kwargs):
    """This function is an alias for ``GPUtil.getAvailable``. If
    ``GPUtil`` is not installed, it returns [0,] as a default GPU ID."""

    return GPUtil.getAvailable(*args, **kwargs)
