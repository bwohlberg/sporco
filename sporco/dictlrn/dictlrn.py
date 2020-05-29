# -*- coding: utf-8 -*-
# Copyright (C) 2015-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Dictionary learning based on ADMM sparse coding and dictionary updates"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from future.utils import with_metaclass
from builtins import range
from builtins import object

import collections

from sporco import cdict
from sporco import common
from sporco.util import u, Timer
from sporco.common import _fix_nested_class_lookup
from sporco.array import transpose_ntpl_list

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



class IterStatsConfig(object):
    """Configuration object for general dictionary learning algorithm
    iteration statistics.
    """

    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""


    def __init__(self, isfld, isxmap, isdmap, evlmap, hdrtxt, hdrmap,
                 fmtmap=None):
        """
        Parameters
        ----------
        isfld : list
          List of field names for iteration statistics namedtuple
        isxmap : dict
          Dictionary mapping iteration statistics namedtuple field names
          to field names in corresponding X step object iteration
          statistics namedtuple
        isdmap : dict
          Dictionary mapping iteration statistics namedtuple field names
          to field names in corresponding D step object iteration
          statistics namedtuple
        evlmap : dict
          Dictionary mapping iteration statistics namedtuple field names
          to labels in the dict returned by :meth:`DictLearn.evaluate`
        hdrtxt : list
          List of column header titles for verbose iteration statistics
          display
        hdrmap : dict
          Dictionary mapping column header titles to IterationStats entries
        fmtmap : dict, optional (default None)
          A dict providing a mapping from field header strings to print
          format strings, providing a mechanism for fields with print
          formats that depart from the standard format
        """

        self.IterationStats = collections.namedtuple('IterationStats', isfld)
        self.isxmap = isxmap
        self.isdmap = isdmap
        self.evlmap = evlmap
        self.hdrtxt = hdrtxt
        self.hdrmap = hdrmap

        # Call utility function to construct status display formatting
        self.hdrstr, self.fmtstr, self.nsep = common.solve_status_str(
            hdrtxt, fmtmap=fmtmap, fwdth0=type(self).fwiter,
            fprec=type(self).fpothr)



    def iterstats(self, j, t, isx, isd, evl):
        """Construct IterationStats namedtuple from X step and D step
        IterationStats namedtuples.

        Parameters
        ----------
        j : int
          Iteration number
        t : float
          Iteration time
        isx : namedtuple
          IterationStats namedtuple from X step object
        isd : namedtuple
          IterationStats namedtuple from D step object
        evl : dict
          Dict associating result labels with values computed by
          :meth:`DictLearn.evaluate`
        """

        vlst = []
        # Iterate over the fields of the IterationStats namedtuple
        # to be populated with values. If a field name occurs as a
        # key in the isxmap dictionary, use the corresponding key
        # value as a field name in the isx namedtuple for the X
        # step object and append the value of that field as the
        # next value in the IterationStats namedtuple under
        # construction. The isdmap dictionary is handled
        # correspondingly with respect to the isd namedtuple for
        # the D step object. There are also two reserved field
        # names, 'Iter' and 'Time', referring respectively to the
        # iteration number and run time of the dictionary learning
        # algorithm.
        for fnm in self.IterationStats._fields:
            if fnm in self.isxmap:
                vlst.append(getattr(isx, self.isxmap[fnm]))
            elif fnm in self.isdmap:
                vlst.append(getattr(isd, self.isdmap[fnm]))
            elif fnm in self.evlmap:
                vlst.append(evl[fnm])
            elif fnm == 'Iter':
                vlst.append(j)
            elif fnm == 'Time':
                vlst.append(t)
            else:
                vlst.append(None)

        return self.IterationStats._make(vlst)



    def printheader(self):
        """Print status display header and separator strings."""

        print(self.hdrstr)
        self.printseparator()



    def printseparator(self):
        "Print status display separator string."""

        print("-" * self.nsep)



    def printiterstats(self, itst):
        """Print iteration statistics.

        Parameters
        ----------
        itst : namedtuple
          IterationStats namedtuple as returned by :meth:`iterstats`
        """

        itdsp = tuple([getattr(itst, self.hdrmap[col]) for col in self.hdrtxt])
        print(self.fmtstr % itdsp)




class _DictLearn_Meta(type):
    """Metaclass for DictLearn class that handles intialisation of the
    object initialisation timer and stopping this timer at the end of
    initialisation.
    """

    def __init__(cls, *args):

        # Apply _fix_nested_class_lookup function to class after creation
        _fix_nested_class_lookup(cls, nstnm='Options')



    def __call__(cls, *args, **kwargs):

        # Initialise instance
        instance = super(_DictLearn_Meta, cls).__call__(*args, **kwargs)
        # Stop initialisation timer
        instance.timer.stop('init')
        # Return instance
        return instance



class DictLearn(with_metaclass(_DictLearn_Meta, object)):
    """General dictionary learning class that supports alternation
    between user-specified sparse coding and dictionary update steps,
    each of which is based on an ADMM algorithm.
    """


    class Options(cdict.ConstrainedDict):
        """General dictionary learning algorithm options.

        Options:

          ``Verbose`` : Flag determining whether iteration status is
          displayed.

          ``StatusHeader`` : Flag determining whether status header and
          separator are displayed.

          ``IterTimer`` : Label of the timer to use for iteration times.

          ``MaxMainIter`` : Maximum main iterations.

          ``Callback`` : Callback function to be called at the end of
          every iteration.
        """

        defaults = {'Verbose': False, 'StatusHeader': True,
                    'IterTimer': 'solve', 'MaxMainIter': 1000,
                    'Callback': None}


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              DictLearn algorithm options
            """

            if opt is None:
                opt = {}
            cdict.ConstrainedDict.__init__(self, opt)



    def __new__(cls, *args, **kwargs):
        """Create a DictLearn object and start its initialisation timer."""

        instance = super(DictLearn, cls).__new__(cls)
        instance.timer = Timer(['init', 'solve', 'solve_wo_eval'])
        instance.timer.start('init')
        return instance



    def __init__(self, xstep, dstep, opt=None, isc=None):
        """
        Parameters
        ----------
        xstep : bpdn (or similar interface) object
          Object handling X update step
        dstep : cmod (or similar interface) object
          Object handling D update step
        opt : :class:`DictLearn.Options` object
          Algorithm options
        isc : :class:`IterStatsConfig` object
          Iteration statistics and header display configuration
        """

        if opt is None:
            opt = DictLearn.Options()
        self.opt = opt

        if isc is None:
            isc = IterStatsConfig(
                isfld=['Iter', 'ObjFunX', 'XPrRsdl', 'XDlRsdl', 'XRho',
                       'ObjFunD', 'DPrRsdl', 'DDlRsdl', 'DRho', 'Time'],
                isxmap={'ObjFunX': 'ObjFun', 'XPrRsdl': 'PrimalRsdl',
                        'XDlRsdl': 'DualRsdl', 'XRho': 'Rho'},
                isdmap={'ObjFunD': 'DFid', 'DPrRsdl': 'PrimalRsdl',
                        'DDlRsdl': 'DualRsdl', 'DRho': 'Rho'},
                evlmap={},
                hdrtxt=['Itn', 'FncX', 'r_X', 's_X', u('ρ_X'),
                        'FncD', 'r_D', 's_D', u('ρ_D')],
                hdrmap={'Itn': 'Iter', 'FncX': 'ObjFunX',
                        'r_X': 'XPrRsdl', 's_X': 'XDlRsdl',
                        u('ρ_X'): 'XRho', 'FncD': 'ObjFunD',
                        'r_D': 'DPrRsdl', 's_D': 'DDlRsdl',
                        u('ρ_D'): 'DRho'}
            )
        self.isc = isc

        self.xstep = xstep
        self.dstep = dstep

        self.itstat = []
        self.j = 0



    def solve(self):
        """Start (or re-start) optimisation. This method implements the
        framework for the alternation between `X` and `D` updates in a
        dictionary learning algorithm. There is sufficient flexibility
        in specifying the two updates that it calls that it is
        usually not necessary to override this method in derived
        clases.

        If option ``Verbose`` is ``True``, the progress of the
        optimisation is displayed at every iteration. At termination
        of this method, attribute :attr:`itstat` is a list of tuples
        representing statistics of each iteration.

        Attribute :attr:`timer` is an instance of :class:`.util.Timer`
        that provides the following labelled timers:

          ``init``: Time taken for object initialisation by
          :meth:`__init__`

          ``solve``: Total time taken by call(s) to :meth:`solve`

          ``solve_wo_func``: Total time taken by call(s) to
          :meth:`solve`, excluding time taken to compute functional
          value and related iteration statistics

          ``solve_wo_rsdl`` : Total time taken by call(s) to
          :meth:`solve`, excluding time taken to compute functional
          value and related iteration statistics as well as time take
          to compute residuals and implemented ``AutoRho`` mechanism
        """

        # Print header and separator strings
        if self.opt['Verbose'] and self.opt['StatusHeader']:
            self.isc.printheader()

        # Reset timer
        self.timer.start(['solve', 'solve_wo_eval'])

        # Main optimisation iterations
        for self.j in range(self.j, self.j + self.opt['MaxMainIter']):

            # X update
            self.xstep.solve()
            self.post_xstep()

            # D update
            self.dstep.solve()
            self.post_dstep()

            # Evaluate functional
            self.timer.stop('solve_wo_eval')
            evl = self.evaluate()
            self.timer.start('solve_wo_eval')

            # Record elapsed time
            t = self.timer.elapsed(self.opt['IterTimer'])

            # Extract and record iteration stats
            xitstat = self.xstep.itstat[-1] if self.xstep.itstat else \
                      self.xstep.IterationStats(
                          *([0.0,] * len(self.xstep.IterationStats._fields)))
            ditstat = self.dstep.itstat[-1] if self.dstep.itstat else \
                      self.dstep.IterationStats(
                          *([0.0,] * len(self.dstep.IterationStats._fields)))
            itst = self.isc.iterstats(self.j, t, xitstat, ditstat, evl)
            self.itstat.append(itst)

            # Display iteration stats if Verbose option enabled
            if self.opt['Verbose']:
                self.isc.printiterstats(itst)

            # Call callback function if defined
            if self.opt['Callback'] is not None:
                if self.opt['Callback'](self):
                    break


        # Increment iteration count
        self.j += 1

        # Record solve time
        self.timer.stop(['solve', 'solve_wo_eval'])

        # Print final separator string if Verbose option enabled
        if self.opt['Verbose'] and self.opt['StatusHeader']:
            self.isc.printseparator()

        # Return final dictionary
        return self.getdict()



    def post_xstep(self):
        """Handle passing result of xstep to dstep"""

        self.dstep.setcoef(self.xstep.getcoef())



    def post_dstep(self):
        """Handle passing result of dstep to xstep"""

        self.xstep.setdict(self.dstep.getdict())



    def evaluate(self):
        """Evaluate results (e.g. functional value) of previous iteration"""

        return None



    def getdict(self):
        """Get final dictionary"""

        return self.dstep.getdict()



    def getcoef(self):
        """Get final coefficient map array"""

        return self.xstep.getcoef()



    def getitstat(self):
        """Get iteration stats as named tuple of arrays instead of array of
        named tuples.
        """

        return transpose_ntpl_list(self.itstat)
