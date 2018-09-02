# -*- coding: utf-8 -*-
# Copyright (C) 2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Utility functions for cupy module that are active when cupy is not
available
"""


import numpy as np


def array_module(*args):
    """When ``cupy`` is available, this function is an alias for
    :func:`cupy.get_array_module`, otherwise it returns the ``numpy``
    module.
    """

    return np


def np2cp(u):
    """Identity function."""

    return u


def cp2np(u):
    """Identity function."""

    return u


def cupy_wrapper(func):
    """Identity wrapper function."""

    return func


def available_gpu(*args, **kwargs):
    """This function is an alias for ``GPUtil.getAvailable``. If
    ``GPUtil`` is not installed, it returns 0 as a default GPU ID."""

    return 0
