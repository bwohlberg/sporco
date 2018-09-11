# -*- coding: utf-8 -*-
# Copyright (C) 2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Additional utility functions for cupy module that are active when
CuPy or GPUtil are not available
"""


def gpu_info():
    """Return an empty list."""

    return []


def gpu_load(wproc=0.5, wmem=0.5):
    """Return an empty list."""

    return []


def device_by_load(wproc=0.5, wmem=0.5):
    """Return an empty list."""

    return []


def select_device_by_load(wproc=0.5, wmem=0.5):
    """Return 0."""

    return 0


def available_gpu(*args, **kwargs):
    """Returns [0,]."""

    return [0,]
