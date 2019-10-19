# -*- coding: utf-8 -*-
# Copyright (C) 2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Private utility functions."""

from __future__ import absolute_import

import sys
import warnings
import functools
import importlib


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def renamed_function(depname, depmod=None):
    """Decorator for renamed functions.

    This decorator creates a copy of the decorated function with the
    previous function name and including a deprecation warning.

    Parameters
    ----------
    depname : string
      Previous (now-deprecated) name of function
    depmod : string, optional (default None)
      Module in which now-deprecated function was defined. A value of
      None implies that the now-deprecated function was defined in the
      same module as its replacement.

    Returns
    -------
    func : function
      The decorated function
    """

    def decorator(func):
        thismod = sys.modules[func.__module__]
        if depmod is None:
            mod = thismod
        else:
            mod = importlib.import_module(depmod)
        dstr = "Function %s.%s is deprecated; please use :func:`%s.%s` " \
               "instead." % (mod.__name__, depname, thismod.__name__,
                             func.__name__)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            wstr = "Function %s.%s is deprecated; please use function %s.%s " \
                   "instead." % (mod.__name__, depname, thismod.__name__,
                                 func.__name__)
            warnings.warn(wstr, DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        wrapper.__name__ = depname
        wrapper.__qualname__ = depname
        wrapper.__module__ = mod.__name__
        wrapper.__doc__ = dstr
        setattr(mod, depname, wrapper)
        return func
    return decorator
