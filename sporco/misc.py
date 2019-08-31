# -*- coding: utf-8 -*-
# Copyright (C) 2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Miscellaneous functions."""

from __future__ import absolute_import

import sys
import warnings
import functools



def renamed_function(depname):
    """Decorator for renamed functions.

    This decorator creates a copy of the decorated function with the
    previous function name and including a deprecation warning.

    Parameters
    ----------
    depname : string
      Previous (now-deprecated) name of function

    Returns
    -------
    func : function
      The decorated function
    """

    def decorator(func):
        dstr = "Function %s is deprecated; please use :func:`%s` " \
               "instead." % (depname, func.__name__)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            wstr = "Function %s is deprecated; please use function %s " \
                   "instead." % (depname, func.__name__)
            warnings.warn(wstr, DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        wrapper.__name__ = depname
        wrapper.__qualname__ = depname
        wrapper.__doc__ = dstr
        setattr(sys.modules[func.__module__], depname, wrapper)
        return func
    return decorator
