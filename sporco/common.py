# -*- coding: utf-8 -*-
# Copyright (C) 2015-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Common functions and classes iterative solver classes"""

from __future__ import division, print_function
from future.utils import with_metaclass
from builtins import object

import sys
import re
import collections
import numpy as np


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def _fix_nested_class_lookup(cls, nstnm):
    """Fix name lookup problem that prevents pickling of classes with
    nested class definitions. The approach is loosely based on that
    implemented at https://git.io/viGqU , simplified and modified to
    work in both Python 2.7 and Python 3.x.

    Parameters
    ----------
    cls : class
      Outer class to which fix is to be applied
    nstnm : string
      Name of nested (inner) class to be renamed
    """

    # Check that nstnm is an attribute of cls
    if nstnm in cls.__dict__:
        # Get the attribute of cls by its name
        nst = cls.__dict__[nstnm]
        # Check that the attribute is a class
        if isinstance(nst, type):
            # Get the module in which the outer class is defined
            mdl = sys.modules[cls.__module__]
            # Construct an extended name by concatenating inner and outer
            # names
            extnm = cls.__name__ + nst.__name__
            # Allow lookup of the nested class within the module via
            # its extended name
            setattr(mdl, extnm, nst)
            # Change the nested class name to the extended name
            nst.__name__ = extnm
    return cls



def _fix_dynamic_class_lookup(cls, pstfx):
    """Fix name lookup problem that prevents pickling of dynamically
    defined classes.

    Parameters
    ----------
    cls : class
      Dynamically generated class to which fix is to be applied
    pstfx : string
      Postfix that can be used to identify dynamically generated classes
      that are equivalent by construction
    """

    # Extended name for the class that will be added to the module namespace
    extnm = '_' + cls.__name__ + '_' + pstfx
    # Get the module in which the dynamic class is defined
    mdl = sys.modules[cls.__module__]
    # Allow lookup of the dynamically generated class within the module via
    # its extended name
    setattr(mdl, extnm, cls)
    # Change the dynamically generated class name to the extended name
    if hasattr(cls, '__qualname__'):
        cls.__qualname__ = extnm
    else:
        cls.__name__ = extnm





class _IterSolver_Meta(type):
    """Metaclass for iterative solver classes that handles
    intialisation of IterationStats namedtuple and applies
    :func:`_fix_nested_class_lookup` to class definitions to fix
    problems with lookup of nested class definitions when using pickle.
    It is also responsible for stopping the object initialisation timer
    at the end of initialisation.
    """

    def __init__(cls, *args):

        # Initialise named tuple type for recording iteration statistics
        cls.IterationStats = collections.namedtuple('IterationStats',
                                                    cls.itstat_fields())
        # Apply _fix_nested_class_lookup function to class after creation
        _fix_nested_class_lookup(cls, nstnm='Options')



    def __call__(cls, *args, **kwargs):

        # Initialise instance
        instance = super(_IterSolver_Meta, cls).__call__(*args, **kwargs)
        # Stop initialisation timer
        instance.timer.stop('init')
        # Return instance
        return instance





class IterativeSolver(with_metaclass(_IterSolver_Meta, object)):
    """Base class for iterative solver classes, providing some common
    infrastructure.
    """

    itstat_fields_objfn = ()
    """Fields in IterationStats associated with the objective function"""
    itstat_fields_alg = ()
    """Fields in IterationStats associated with the specific solver
    algorithm, e.g. ADMM or PGM"""
    itstat_fields_extra = ()
    """Non-standard fields in IterationStats"""



    @classmethod
    def itstat_fields(cls):
        """Construct tuple of field names used to initialise
        IterationStats named tuple.
        """

        return ('Iter',) + cls.itstat_fields_objfn + \
            cls.itstat_fields_alg + cls.itstat_fields_extra + ('Time',)



    def set_dtype(self, opt, dtype):
        """Set the `dtype` attribute. If opt['DataType'] has a value
        other than None, it overrides the `dtype` parameter of this
        method. No changes are made if the `dtype` attribute already
        exists and has a value other than 'None'.

        Note that the `dtype` attribute is expected to have type
        `numpy.dtype` rather than `type`, e.g. for float32 values, it
        should be `np.dtype(np.float32)` rather than `np.float32`.

        Parameters
        ----------
        opt : :class:`cdict.ConstrainedDict` object
          Algorithm options
        dtype : numpy.dtype
          Data type for working variables (overridden by 'DataType' option)
        """

        # Take no action of self.dtype exists and is not None
        if not hasattr(self, 'dtype') or self.dtype is None:
            # DataType option overrides explicitly specified data type
            if opt['DataType'] is None:
                # We expect dtype to already be an instance of np.dtype,
                # but explicitly convert it in case it was incorrectly
                # specified as an instance of type
                self.dtype = np.dtype(dtype)
            else:
                self.dtype = np.dtype(opt['DataType'])



    def set_attr(self, name, val, dval=None, dtype=None, reset=False):
        """Set an object attribute by its name. The attribute value
        can be specified as a primary value `val`, and as default
        value 'dval` that will be used if the primary value is None.
        This arrangement allows an attribute to be set from an entry
        in an options object, passed as `val`, while specifying a
        default value to use, passed as `dval` in the event that the
        options entry is None. Unless `reset` is True, the attribute
        is only set if it doesn't exist, or if it exists with value
        None. This arrangement allows for attributes to be set in
        both base and derived class initialisers, with the derived
        class value taking preference.

        Parameters
        ----------
        name : string
          Attribute name
        val : any
          Primary attribute value
        dval : any
          Default attribute value in case `val` is None
        dtype : data-type, optional (default None)
          If the `dtype` parameter is not None, the attribute `name` is
          set to `val` (which is assumed to be of numeric type) after
          conversion to the specified type.
        reset : bool, optional (default False)
          Flag indicating whether attribute assignment should be
          conditional on the attribute not existing or having value None.
          If False, an attribute value other than None will not be
          overwritten.
        """

        # If `val` is None and `dval` is not None, replace it with dval
        if dval is not None and val is None:
            val = dval

        # If dtype is not None, assume val is numeric and convert it to
        # type dtype
        if dtype is not None and val is not None:
            if isinstance(dtype, type):
                val = dtype(val)
            else:
                val = dtype.type(val)

        # Set attribute value depending on reset flag and whether the
        # attribute exists and is None
        if reset or not hasattr(self, name) or \
           (hasattr(self, name) and getattr(self, name) is None):
            setattr(self, name, val)




def solve_status_str(hdrlbl, fmtmap=None, fwdth0=4, fwdthdlt=6,
                     fprec=2):
    """Construct header and format details for status display of an
    iterative solver.

    Parameters
    ----------
    hdrlbl : tuple of strings
      Tuple of field header strings
    fmtmap : dict or None, optional (default None)
      A dict providing a mapping from field header strings to print
      format strings, providing a mechanism for fields with print
      formats that depart from the standard format
    fwdth0 : int, optional (default 4)
      Number of characters in first field formatted for integers
    fwdthdlt : int, optional (default 6)
      The width of fields formatted for floats is the sum of the value
      of this parameter and the field precision
    fprec : int, optional (default 2)
      Precision of fields formatted for floats

    Returns
    -------
    hdrstr : string
      Complete header string
    fmtstr : string
      Complete print formatting string for numeric values
    nsep : integer
      Number of characters in separator string
    """

    if fmtmap is None:
        fmtmap = {}
    fwdthn = fprec + fwdthdlt

    # Construct a list specifying the format string for each field.
    # Use format string from fmtmap if specified, otherwise use
    # a %d specifier with field width fwdth0 for the first field,
    # or a %e specifier with field width fwdthn and precision
    # fprec
    fldfmt = [fmtmap[lbl] if lbl in fmtmap else
              (('%%%dd' % (fwdth0)) if idx == 0 else
               (('%%%d.%de' % (fwdthn, fprec))))
              for idx, lbl in enumerate(hdrlbl)]
    fmtstr = ('  ').join(fldfmt)

    # Construct a list of field widths for each field by extracting
    # field widths from field format strings
    cre = re.compile(r'%-?(\d+)')
    fldwid = []
    for fmt in fldfmt:
        mtch = cre.match(fmt)
        if mtch is None:
            raise ValueError("Format string '%s' does not contain field "
                             "width" % fmt)
        else:
            fldwid.append(int(mtch.group(1)))

    # Construct list of field header strings formatted to the
    # appropriate field width, and join to construct a combined field
    # header string
    hdrlst = [('%-*s' % (w, t)) for t, w in zip(hdrlbl, fldwid)]
    hdrstr = ('  ').join(hdrlst)

    return hdrstr, fmtstr, len(hdrstr)
