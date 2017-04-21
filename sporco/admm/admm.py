# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Base classes for ADMM algorithms"""

from __future__ import division
from __future__ import print_function
from future.utils import with_metaclass
from builtins import range
from builtins import object

import copy
import collections
import sys
import warnings
import numpy as np
import scipy
from scipy import linalg

from sporco import cdict
from sporco import util
from sporco.util import u


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def _module_name_nested(cls, nstnm='Options'):
    """Fix name lookup problem that prevents pickling of classes with nested
    class definitions. The approach is loosely based on that implemented at
    https://git.io/viGqU , simplified and modified to work
    in both Python 2.7 and Python 3.x.

    Parameters
    ----------
    cls : class
      Class to which fix is to be applied
    nstnm : str, optional (default 'Options')
      Name of nested class to be renamed
    """

    # Check that nstmm is an attribute of cls
    if nstnm in cls.__dict__:
        # Get the attribute of cls by its name
        nst = cls.__dict__[nstnm]
        # Check that the attribute is a class
        if isinstance(nst, type):
            # Get the module in which the outer class is defined
            mdl = sys.modules[cls.__module__]
            # Construct an extended name by concatenating inner and outer names
            extnm = cls.__name__ + nst.__name__
            # Allow lookup of the nested class within the module via
            # its extended name
            setattr(mdl, extnm, nst)
            # Change the nested class name to the extended name
            nst.__name__ = extnm
    return cls



class _ADMM_Meta(type):
    """Metaclass for ADMM class that handles intialisation of
    IterationStats namedtuple and applies module_name_nested to class
    definitions to fix problems with lookup of nested class
    definitions when using pickle. It is also responsible for stopping
    the object initialisation timer at the end of initialisation.
    """

    def __init__(cls, *args):

        # Initialise named tuple type for recording ADMM iteration statistics
        cls.IterationStats = collections.namedtuple('IterationStats',
                                                    cls.itstat_fields())
        # Ensure that timer attribute has been initialised
        cls.timer = util.Timer(['init', 'solve', 'solve_wo_func',
                                'solve_wo_rsdl'])
        # Apply _module_name_nested function to class after creation
        _module_name_nested(cls)



    def __call__(cls, *args, **kwargs):

        # Initialise instance
        instance = super(_ADMM_Meta, cls).__call__(*args, **kwargs)
        # Stop initialisation timer
        instance.timer.stop('init')
        # Return instance
        return instance




class ADMM(with_metaclass(_ADMM_Meta, object)):
    r"""Base class for Alternating Direction Method of Multipliers (ADMM)
    algorithms :cite:`boyd-2010-distributed`.

    Solve an optimisation problems of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}} \;
       f(\mathbf{x}) + g(\mathbf{y}) \;\mathrm{such\;that}\;
       A\mathbf{x} + B\mathbf{y} = \mathbf{c} \;\;.

    This class is intended to be a base class of other classes that
    specialise to specific optimisation problems.

    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The default fields of the named tuple
    ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``FVal`` :  Value of objective function component :math:`f`

       ``GVal`` : Value of objective function component :math:`g`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual Residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}` (see Sec. 3.3.1 of
       :cite:`boyd-2010-distributed`)

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}` (see Sec. 3.3.1 of
       :cite:`boyd-2010-distributed`)

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """

    class Options(cdict.ConstrainedDict):
        r"""ADMM algorithm options.

        Options:

          ``FastSolve`` : Flag determining whether non-essential
          computation is skipped. When ``FastSolve`` is ``True`` and
          ``Verbose`` is ``False``, the functional value and related
          iteration statistics are not computed. If ``FastSolve`` is
          ``True`` and the ``AutoRho`` mechanism is disabled,
          residuals are also not calculated, in which case the
          residual-based stopping method is also disabled, with the
          number of iterations determined only by ``MaxMainIter``.

          ``Verbose`` : Flag determining whether iteration status is
          displayed.

          ``StatusHeader`` : Flag determining whether status header and
          separator are displayed.

          ``DataType`` : Specify data type for solution variables,
          e.g. ``np.float32``.

          ``Y0`` : Initial value for Y variable.

          ``U0`` : Initial value for U variable.

          ``Callback`` : Callback function to be called at the end of
          every iteration.

          ``IterTimer`` : Label of the timer to use for iteration times.

          ``MaxMainIter`` : Maximum main iterations.

          ``AbsStopTol`` : Absolute convergence tolerance (see Sec. 3.3.1
          of :cite:`boyd-2010-distributed`).

          ``RelStopTol`` : Relative convergence tolerance (see Sec. 3.3.1
          of :cite:`boyd-2010-distributed`).

          ``RelaxParam`` : Relaxation parameter (see Sec. 3.4.3 of
          :cite:`boyd-2010-distributed`). Note: relaxation is disabled
          by setting this value to 1.0.

          ``rho`` : ADMM penalty parameter :math:`\rho`.

          ``AutoRho`` : Options for adaptive rho strategy (see
          :cite:`wohlberg-2015-adaptive` and Sec. 3.4.3 of
          :cite:`boyd-2010-distributed`).

            ``Enabled`` : Flag determining whether adaptive penalty parameter
            strategy is enabled.

            ``Period`` : Iteration period on which rho is updated. If set to
            1, the rho update test is applied at every iteration.

            ``Scaling`` : Multiplier applied to rho when updated
            (:math:`\tau` in :cite:`wohlberg-2015-adaptive`).

            ``RsdlRatio`` : Primal/dual residual ratio in rho update test
            (:math:`\mu` in :cite:`wohlberg-2015-adaptive`).

            ``RsdlTarget`` : Residual ratio targeted by auto rho update
            policy (:math:`\xi` in :cite:`wohlberg-2015-adaptive`).

            ``AutoScaling`` : Flag determining whether RhoScaling value is
            adaptively determined (see Sec. IV.C in
            :cite:`wohlberg-2015-adaptive`). If enabled, ``Scaling``
            specifies a maximum allowed multiplier instead of a fixed
            multiplier.

            ``StdResiduals`` : Flag determining whether standard residual
            definitions are used instead of normalised residuals (see
            Sec. IV.B in :cite:`wohlberg-2015-adaptive`).
        """

        defaults = {'FastSolve' : False, 'Verbose' : False,
                    'StatusHeader' : True, 'DataType' : None,
                    'MaxMainIter' : 1000, 'IterTimer' : 'solve',
                    'AbsStopTol' : 0.0, 'RelStopTol' : 1e-3,
                    'RelaxParam' : 1.0, 'rho' : None,
                    'AutoRho' :
                    {
                        'Enabled' : False, 'Period' : 10,
                        'Scaling' : 2.0, 'RsdlRatio' : 10.0,
                        'RsdlTarget' : None, 'AutoScaling' : False,
                        'StdResiduals' : False
                    },
                    'Y0' : None, 'U0' : None, 'Callback' : None
                   }

        def __init__(self, opt=None):
            """Initialise ADMM algorithm options object."""

            if opt is None:
                opt = {}
            cdict.ConstrainedDict.__init__(self, opt)



    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""

    itstat_fields_objfn = ('ObjFun', 'FVal', 'GVal')
    """Fields in IterationStats associated with the objective function;
    see :meth:`eval_objfun`"""
    itstat_fields_extra = ()
    """Non-standard fields in IterationStats; see :meth:`itstat_extra`"""

    hdrtxt_objfn = ('Fnc', 'f', 'g')
    """Display column headers associated with the objective function;
    see :meth:`eval_objfun`"""
    hdrval_objfun = {'Fnc' : 'ObjFun', 'f' : 'FVal', 'g' : 'GVal'}
    """Dictionary mapping display column headers in :attr:`hdrtxt_objfn`
    to IterationStats entries"""



    def __new__(cls, *args, **kwargs):
        """Create an ADMM object and start its initialisation timer."""

        instance = super(ADMM, cls).__new__(cls)
        instance.timer.start('init')
        return instance



    def __init__(self, Nx, yshape, ushape, dtype, opt=None):
        r"""
        Initialise an ADMM object with problem size and options.

        Parameters
        ----------
        Nx : int
          Size of variable :math:`\mathbf{x}` in objective function
        yshape : tuple of ints
          Shape of working variable Y (the auxiliary variable)
        ushape : tuple of ints
          Shape of working variable U (the scaled dual variable)
        dtype : data-type
          Data type for working variables (overridden by 'DataType' option)
        opt : :class:`ADMM.Options` object
          Algorithm options
        """

        if opt is None:
            opt = ADMM.Options()
        if not isinstance(opt, ADMM.Options):
            raise TypeError("Parameter opt must be an instance of ADMM.Options")

        self.opt = opt
        self.Nx = Nx
        # Working variable U has the same dimensionality as constant c
        # in the constraint Ax + By = c
        self.Nc = np.product(ushape)

        # DataType option overrides data type inferred from __init__
        # parameters of derived class
        self.set_dtype(opt, dtype)

        # Initialise attributes representing penalty parameter and other
        # parameters
        self.set_attr('rho', opt['rho'], dval=1.0, dtype=self.dtype)
        self.set_attr('rho_tau', opt['AutoRho', 'Scaling'], dval=2.0,
                      dtype=self.dtype)
        self.set_attr('rho_mu', opt['AutoRho', 'RsdlRatio'], dval=10.0,
                      dtype=self.dtype)
        self.set_attr('rho_xi', opt['AutoRho', 'RsdlTarget'], dval=1.0,
                      dtype=self.dtype)
        self.set_attr('rlx', opt['RelaxParam'], dval=1.0, dtype=self.dtype)


        # Initialise working variable Y
        if self.opt['Y0'] is None:
            self.Y = self.yinit(yshape)
        else:
            self.Y = self.opt['Y0'].astype(self.dtype, copy=True)
        self.Yprev = self.Y.copy()

        # Initialise working variable U
        if self.opt['U0'] is None:
            self.U = self.uinit(ushape)
        else:
            self.U = self.opt['U0'].astype(self.dtype, copy=True)

        self.itstat = []
        self.k = 0



    def set_dtype(self, opt, dtype):
        """Set the `dtype` attribute. If opt['DataType'] has a value other
        than None, it overrides the `dtype` parameter of this method. No
        changes are made if the `dtype` attribute already exists and has a
        value other than 'None'.

        Parameters
        ----------
        opt : :class:`ADMM.Options` object
          Algorithm options
        dtype : data-type
          Data type for working variables (overridden by 'DataType' option)
        """

        # Take no action of self.dtype exists has is not None
        if not hasattr(self, 'dtype') or self.dtype is None:
            # DataType option overrides explicitly specified data type
            if opt['DataType'] is None:
                self.dtype = dtype
            else:
                self.dtype = np.dtype(opt['DataType'])



    def set_attr(self, name, val, dval=None, dtype=None, reset=False):
        """Set an object attribute.

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
          set to `val` after conversion to the specified type.
          self.dtype
        reset : bool, optional (default False)
          Flag indicating whether attribute assignment should be conditional
          on the attribute not existing or having value None. If False,
          an attribute value other than None will not be not overwritten.
        """

        # If `val` is None and `dval` is not, replace it with dval
        if dval is not None and val is None:
            val = dval

        # If val is flagged as numeric, convert it to type self.dtype
        if dtype is not None and val is not None:
            val = self.dtype.type(val)

        # Set attribute value depending on reset flag and whether the
        # attribute exists and is None
        if reset or not hasattr(self, name) or \
           (hasattr(self, name) and getattr(self, name) is None):
            setattr(self, name, val)



    def yinit(self, yshape):
        """Return initialiser for working variable Y"""

        return np.zeros(yshape, dtype=self.dtype)



    def uinit(self, ushape):
        """Return initialiser for working variable U"""

        return np.zeros(ushape, dtype=self.dtype)



    def solve(self):
        """Start (or re-start) optimisation. This method implements the
        framework for the iterations of an ADMM algorithm. There is sufficient
        flexibility in overriding the component methods that it calls that it
        is usually not necessary to override this method in derived clases.

        If option ``Verbose`` is ``True``, the progress of the
        optimisation is displayed at every iteration. At termination
        of this method, attribute :attr:`itstat` is a list of tuples
        representing statistics of each iteration, unless option
        ``FastSolve`` is ``True`` and option ``Verbose`` is ``False``.

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

        # Open status display
        fmtstr, nsep = self.display_start()

        # Start solve timer
        self.timer.start(['solve', 'solve_wo_func', 'solve_wo_rsdl'])

        # Main optimisation iterations
        for k in range(self.k, self.k + self.opt['MaxMainIter']):

            # Update record of Y from previous iteration
            self.Yprev = self.Y.copy()

            # X update
            self.xstep()

            # Implement relaxation if RelaxParam != 1.0
            self.relax_AX()

            # Y update
            self.ystep()

            # U update
            self.ustep()

            # Compute residuals and stopping thresholds
            self.timer.stop('solve_wo_rsdl')
            if self.opt['AutoRho', 'Enabled'] or not self.opt['FastSolve']:
                r, s, epri, edua = self.compute_residuals()
            self.timer.start('solve_wo_rsdl')

            # Compute and record other iteration statistics and
            # display iteration stats if Verbose option enabled
            self.timer.stop(['solve_wo_func', 'solve_wo_rsdl'])
            if not self.opt['FastSolve']:
                itst = self.iteration_stats(k, r, s, epri, edua)
                self.itstat.append(itst)
                self.display_status(fmtstr, itst)
            self.timer.start(['solve_wo_func', 'solve_wo_rsdl'])

            # Automatic rho adjustment
            self.timer.stop('solve_wo_rsdl')
            if self.opt['AutoRho', 'Enabled'] or not self.opt['FastSolve']:
                self.update_rho(k, r, s)
            self.timer.start('solve_wo_rsdl')

            # Call callback function if defined
            if self.opt['Callback'] is not None:
                self.opt['Callback'](self, k)

            # Stop if residual-based stopping tolerances reached
            if self.opt['AutoRho', 'Enabled'] or not self.opt['FastSolve']:
                if r < epri and s < edua:
                    break


        # Record iteration count
        self.k = k+1

        # Record solve time
        self.timer.stop(['solve', 'solve_wo_func', 'solve_wo_rsdl'])

        # Print final separator string if Verbose option enabled
        self.display_end(nsep)

        return self.getmin()



    @property
    def runtime(self):
        """Transitional property providing access to the new timer
        mechanism. This will be removed in the future.
        """

        warnings.warn("admm.ADMM.runtime attribute has been replaced by "\
            "an upgraded timer class: please see the documentation for "\
            "admm.ADMM.solve method and util.Timer class")
        return self.timer.elapsed('init') + self.timer.elapsed('solve')



    def getmin(self):
        """Get minimizer after optimisation."""

        return self.X



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{x}`.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{y}`.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def ustep(self):
        """Dual variable update."""

        self.U += self.rsdl_r(self.AX, self.Y)



    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        # We need to keep the non-relaxed version of AX since it is
        # required for computation of primal residual r
        self.AXnr = self.cnst_A(self.X)
        if self.rlx == 1.0:
            # If RelaxParam option is 1.0 there is no relaxation
            self.AX = self.AXnr
        else:
            # Avoid calling cnst_c() more than once in case it is expensive
            # (e.g. due to allocation of a large block of memory)
            if not hasattr(self, '_cnst_c'):
                self._cnst_c = self.cnst_c()
            # Compute relaxed version of AX
            alpha = self.rlx
            self.AX = alpha*self.AXnr - (1-alpha)*(self.cnst_B(self.Y) -
                                                   self._cnst_c)



    def compute_residuals(self):
        """Compute residuals and stopping thresholds."""

        if self.opt['AutoRho', 'StdResiduals']:
            r = linalg.norm(self.rsdl_r(self.AXnr, self.Y))
            s = linalg.norm(self.rsdl_s(self.Yprev, self.Y))
            epri = scipy.sqrt(self.Nc)*self.opt['AbsStopTol'] + \
                self.rsdl_rn(self.AXnr, self.Y)*self.opt['RelStopTol']
            edua = scipy.sqrt(self.Nx)*self.opt['AbsStopTol'] + \
                self.rsdl_sn(self.U)*self.opt['RelStopTol']
        else:
            rn = self.rsdl_rn(self.AXnr, self.Y)
            if rn == 0.0:
                rn = 1.0
            sn = self.rsdl_sn(self.U)
            if sn == 0.0:
                sn = 1.0
            r = linalg.norm(self.rsdl_r(self.AXnr, self.Y)) / rn
            s = linalg.norm(self.rsdl_s(self.Yprev, self.Y)) / sn
            epri = scipy.sqrt(self.Nc)*self.opt['AbsStopTol']/rn + \
                self.opt['RelStopTol']
            edua = scipy.sqrt(self.Nx)*self.opt['AbsStopTol']/sn + \
                self.opt['RelStopTol']

        return r, s, epri, edua



    @classmethod
    def itstat_fields(cls):
        """Construct tuple of field names used to initialise IterationStats
        named tuple.
        """

        return ('Iter',) + cls.itstat_fields_objfn + \
          ('PrimalRsdl', 'DualRsdl', 'EpsPrimal', 'EpsDual', 'Rho') + \
          cls.itstat_fields_extra + ('Time',)



    @classmethod
    def hdrtxt(cls):
        """Construct tuple of status display column title"""

        return ('Itn',) + cls.hdrtxt_objfn + ('r', 's', u('ρ'))



    @classmethod
    def hdrval(cls):
        """Construct dictionary mapping display column title to
        IterationStats entries.
        """

        dict = {'Itn' : 'Iter'}
        dict.update(cls.hdrval_objfun)
        dict.update({'r' : 'PrimalRsdl', 's' : 'DualRsdl', u('ρ') : 'Rho'})
        return dict



    def iteration_stats(self, k, r, s, epri, edua):
        """Construct iteration stats record tuple."""

        tk = self.timer.elapsed(self.opt['IterTimer'])
        tpl = (k,) + self.eval_objfn() + (r, s, epri, edua, self.rho) + \
              self.itstat_extra() + (tk,)
        return type(self).IterationStats(*tpl)



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        fval = self.obfn_f(self.X)
        gval = self.obfn_g(self.Y)
        obj = fval + gval
        return (obj, fval, gval)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return ()



    def getitstat(self):
        """Get iteration stats as named tuple of arrays instead of array of
        named tuples.
        """

        if len(self.itstat) == 0:
            return None
        else:
            return type(self).IterationStats(
                *[[self.itstat[k][l] for k in range(len(self.itstat))]
                  for l in range(len(self.itstat[0]))])



    def update_rho(self, k, r, s):
        """Automatic rho adjustment."""

        if self.opt['AutoRho', 'Enabled']:
            tau = self.rho_tau
            mu = self.rho_mu
            xi = self.rho_xi
            if k != 0 and scipy.mod(k+1, self.opt['AutoRho', 'Period']) == 0:
                if self.opt['AutoRho', 'AutoScaling']:
                    if s == 0.0 or r == 0.0:
                        rhomlt = tau
                    else:
                        rhomlt = scipy.sqrt(r/(s*xi) if r > s*xi else (s*xi)/r)
                        if rhomlt > tau:
                            rhomlt = tau
                else:
                    rhomlt = tau
                rsf = 1.0
                if r > xi*mu*s:
                    rsf = rhomlt
                elif s > (mu/xi)*r:
                    rsf = 1.0/rhomlt
                self.rho = self.dtype.type(rsf*self.rho)
                self.U = self.U/rsf
                if rsf != 1.0:
                    self.rhochange()



    def display_start(self):
        """Set up status display if option selected. NB: this method assumes
        that the first entry is the iteration count and the last is
        the rho value.
        """

        if self.opt['Verbose']:
            # If AutoRho option enabled rho is included in iteration status
            if self.opt['AutoRho', 'Enabled']:
                hdrtxt = type(self).hdrtxt()
            else:
                hdrtxt = type(self).hdrtxt()[0:-1]
            # Call utility function to construct status display formatting
            hdrstr, fmtstr, nsep = util.solve_status_str(
                hdrtxt, type(self).fwiter, type(self).fpothr)
            # Print header and separator strings
            if self.opt['StatusHeader']:
                print(hdrstr)
                print("-" * nsep)
        else:
            fmtstr, nsep = '', 0

        return fmtstr, nsep



    def display_status(self, fmtstr, itst):
        """Display current iteration status as selection of fields from
        iteration stats tuple.
        """

        if self.opt['Verbose']:
            hdrtxt = type(self).hdrtxt()
            hdrval = type(self).hdrval()
            itdsp = tuple([getattr(itst, hdrval[col]) for col in hdrtxt])
            if not self.opt['AutoRho', 'Enabled']:
                itdsp = itdsp[0:-1]

            print(fmtstr % itdsp)



    def display_end(self, nsep):
        """Terminate status display if option selected."""

        if self.opt['Verbose'] and self.opt['StatusHeader']:
            print("-" * nsep)



    def var_x(self):
        r"""Get :math:`\mathbf{x}` variable."""

        return self.X



    def var_y(self):
        r"""Get :math:`\mathbf{y}` variable."""

        return self.Y



    def var_u(self):
        r"""Get :math:`\mathbf{u}` variable."""

        return self.U



    def obfn_f(self, X):
        r"""Compute :math:`f(\mathbf{x})` component of ADMM objective function.

        Overriding this method is required if :meth:`eval_objfun`
        is not overridden.
        """

        raise NotImplementedError()



    def obfn_g(self, Y):
        r"""Compute :math:`g(\mathbf{y})` component of ADMM objective function.

        Overriding this method is required if :meth:`eval_objfun`
        is not overridden.
        """

        raise NotImplementedError()



    def cnst_A(self, X):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """

        raise NotImplementedError()



    def cnst_AT(self, X):
        r"""Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """

        raise NotImplementedError()



    def cnst_B(self, Y):
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """

        raise NotImplementedError()



    def cnst_c(self):
        r"""Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """

        raise NotImplementedError()



    def rsdl_r(self, AX, Y):
        """Compute primal residual vector.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        # Avoid calling cnst_c() more than once in case it is expensive
        # (e.g. due to allocation of a large block of memory)
        if not hasattr(self, '_cnst_c'):
            self._cnst_c = self.cnst_c()
        return AX + self.cnst_B(Y) - self._cnst_c



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        return self.rho*self.cnst_AT(self.cnst_B(Y - Yprev))



    def rsdl_rn(self, AX, Y):
        """Compute primal residual normalisation term.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        # Avoid computing the norm of the value returned by cnst_c()
        # more than once
        if not hasattr(self, '_nrm_cnst_c'):
            self._nrm_cnst_c = linalg.norm(self.cnst_c())
        return max((linalg.norm(AX), linalg.norm(self.cnst_B(Y)),
                    self._nrm_cnst_c))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        return self.rho*linalg.norm(self.cnst_AT(U))



    def rhochange(self):
        """Action to be taken, if any, when rho parameter is changed.

        Overriding this method is optional.
        """

        pass





class ADMMEqual(ADMM):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ADMMEqual
       :parts: 2

    |

    Base class for ADMM algorithms with a simple equality constraint.

    Solve optimisation problems of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}} \;
       f(\mathbf{x}) + g(\mathbf{y}) \;\mathrm{such\;that}\;
       \mathbf{x} = \mathbf{y} \;\;.

    This class specialises class ADMM, but remains a base class for
    other classes that specialise to specific optimisation problems.
    """


    class Options(ADMM.Options):
        """ADMMEqual algorithm options.

        Options include all of those defined in :class:`ADMM.Options`,
        together with additional options:

          ``fEvalX`` : Flag indicating whether the :math:`f` component of the
          objective function should be evaluated using variable X
          (``True``) or Y (``False``) as its argument.

          ``gEvalY`` : Flag indicating whether the :math:`g` component of the
          objective function should be evaluated using variable Y
          (``True``) or X (``False``) as its argument.

          ``ReturnX`` : Flag indicating whether the return value of the
          solve method is the X variable (``True``) or the Y variable
          (``False``).
        """

        defaults = copy.deepcopy(ADMM.Options.defaults)
        defaults.update({'fEvalX' : True, 'gEvalY' : True, 'ReturnX' : True})

        def __init__(self, opt=None):
            """Initialise ADMMEqual algorithm options object."""

            if opt is None:
                opt = {}
            ADMM.Options.__init__(self, opt)




    def __init__(self, xshape, dtype, opt=None):
        """
        Initialise an ADMMEqual object with problem size and options.

        Parameters
        ----------
        xshape : tuple of ints
          Shape of working variable X (the primary variable)
        dtype : data-type
          Data type for working variables
        opt : :class:`ADMMEqual.Options` object
          Algorithm options
        """

        if opt is None:
            opt = ADMMEqual.Options()
        Nx = np.product(xshape)
        super(ADMMEqual, self).__init__(Nx, xshape, xshape, dtype, opt)



    def getmin(self):
        """Get minimizer after optimisation."""

        return self.X if self.opt['ReturnX'] else self.Y



    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        self.AXnr = self.X
        if self.rlx == 1.0:
            self.AX = self.X
        else:
            alpha = self.rlx
            self.AX = alpha*self.X + (1-alpha)*self.Y



    def obfn_fvar(self):
        """Variable to be evaluated in computing :meth:`ADMM.obfn_f`, depending
        on the ``fEvalX`` option value.
        """

        return self.X if self.opt['fEvalX'] else self.Y



    def obfn_gvar(self):
        """Variable to be evaluated in computing :meth:`ADMM.obfn_g`, depending
        on the ``gEvalY`` option value.
        """

        return self.Y if self.opt['gEvalY'] else self.X



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        fval = self.obfn_f(self.obfn_fvar())
        gval = self.obfn_g(self.obfn_gvar())
        obj = fval + gval
        return (obj, fval, gval)



    def cnst_A(self, X):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem constraint.
        In this case :math:`A \mathbf{x} = \mathbf{x}` since the constraint
        is :math:`\mathbf{x} = \mathbf{y}`.
        """

        return X


    def cnst_AT(self, Y):
        r"""Compute :math:`A^T \mathbf{y}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{y} = \mathbf{y}` since the constraint
        is :math:`\mathbf{x} = \mathbf{y}`.
        """

        return Y



    def cnst_B(self, Y):
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem constraint.
        In this case :math:`B \mathbf{y} = -\mathbf{y}` since the constraint
        is :math:`\mathbf{x} = \mathbf{y}`.
        """

        return -Y



    def cnst_c(self):
        r"""Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = \mathbf{0}` since
        the constraint is :math:`\mathbf{x} = \mathbf{y}`.
        """

        return 0.0



    def rsdl_r(self, AX, Y):
        """Compute primal residual vector."""

        return AX - Y



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        return self.rho*(Yprev - Y)



    def rsdl_rn(self, AX, Y):
        """Compute primal residual normalisation term."""

        return max((linalg.norm(AX), linalg.norm(Y)))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho*linalg.norm(U)





class ADMMTwoBlockCnstrnt(ADMM):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ADMMTwoBlockCnstrnt
       :parts: 2

    |

    Base class for ADMM algorithms for problems for which
    :math:`g(\mathbf{y}) = g_0(\mathbf{y}_0) + g_1(\mathbf{y}_1)` with
    :math:`\mathbf{y}^T = (\mathbf{y}_0^T \; \mathbf{y}_1^T)`.

    Solve optimisation problems of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x}} \; f(\mathbf{x}) + g_0(A_0 \mathbf{x}) +
       g_1(A_1 \mathbf{x})

    via an ADMM problem of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
       f(\mathbf{x}) + g_0(\mathbf{y}_0) + g_0(\mathbf{y}_1)
       \;\text{such that}\;
       \left( \begin{array}{c} A_0 \\ A_1 \end{array} \right) \mathbf{x}
       - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1 \end{array}
       \right) = \left( \begin{array}{c} \mathbf{c}_0 \\
       \mathbf{c}_1 \end{array} \right) \;\;.

    In this case the ADMM constraint is :math:`A\mathbf{x} + B\mathbf{y} =
    \mathbf{c}` where

    .. math::
       A = \left( \begin{array}{c} A_0 \\ A_1 \end{array} \right)
       \qquad B = -I \qquad \mathbf{y} = \left( \begin{array}{c}
       \mathbf{y}_0 \\ \mathbf{y}_1 \end{array} \right) \qquad
       \mathbf{c} = \left( \begin{array}{c} \mathbf{c}_0 \\
       \mathbf{c}_1 \end{array} \right) \;\;.

    This class specialises class :class:`.ADMM`, but remains a base class for
    other classes that specialise to specific optimisation problems.
    """


    class Options(ADMM.Options):
        r"""ADMMTwoBlockCnstrnt algorithm options.

        Options include all of those defined in :class:`ADMM.Options`,
        together with additional options:

          ``AuxVarObj`` : Flag indicating whether the
          :math:`g(\mathbf{y})` component of the objective function
          should be evaluated using variable X (``False``) or Y
          (``True``) as its argument. Setting this flag to ``True``
          often gives a better estimate of the objective function, but
          at additional computational cost for some problems.

          ``ReturnVar`` : A string (valid values are 'X', 'Y0', or 'Y1')
          indicating which of the objective function variables should be
          returned by the solve method.
        """

        defaults = copy.deepcopy(ADMM.Options.defaults)
        defaults.update({'AuxVarObj' : False, 'ReturnVar' : 'X'})

        def __init__(self, opt=None):
            """Initialise ADMMTwoBlockCnstrnt algorithm options object."""

            if opt is None:
                opt = {}
            ADMM.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'FVal', 'G0Val', 'G1Val')
    """Fields in IterationStats associated with the objective function;
    see :meth:`eval_objfun`"""

    hdrtxt_objfn = ('Fnc', 'f', 'g0', 'g1')
    """Display column headers associated with the objective function;
    see :meth:`eval_objfun`"""
    hdrval_objfun = {'Fnc' : 'ObjFun', 'f' : 'FVal',
                     'g0' : 'G0Val', 'g1' : 'G1Val'}
    """Dictionary mapping display column headers in :attr:`hdrtxt_objfn`
    to IterationStats entries"""



    def __init__(self, Nx, yshape, blkaxis, blkidx, dtype, opt=None):
        r"""
        Initialise an ADMMTwoBlockCnstrnt object with problem size and options.

        Parameters
        ----------
        Nx : int
          Size of variable :math:`\mathbf{x}` in objective function
        yshape : tuple of ints
          Shape of working variable Y (the auxiliary variable)
        blkaxis : int
          Axis on which :math:`\mathbf{y}_0` and :math:`\mathbf{y}_1` are
          concatenated to form :math:`\mathbf{y}`
        blkidx : int
          Index of boundary between :math:`\mathbf{y}_0` and
          :math:`\mathbf{y}_1` on axis on which they are concatenated to
          form :math:`\mathbf{y}`
        dtype : data-type
          Data type for working variables
        opt : :class:`ADMMTwoBlockCnstrnt.Options` object
          Algorithm options
        """

        if opt is None:
            opt = ADMM.Options()
        super(ADMMTwoBlockCnstrnt, self).__init__(Nx, yshape, yshape,
                                                  dtype, opt)
        self.blkaxis = blkaxis
        self.blkidx = blkidx



    def getmin(self):
        """Get minimizer after optimisation."""

        if self.opt['ReturnVar'] == 'X':
            return self.var_x()
        elif self.opt['ReturnVar'] == 'Y0':
            return self.var_y0()
        elif self.opt['ReturnVar'] == 'Y1':
            return self.var_y1()
        else:
            raise ValueError(self.opt['ReturnVar'] + ' is not a valid value'
                             'for option ReturnVar')



    def block_sep0(self, Y):
        r"""Separate variable into component corresponding to
        :math:`\mathbf{y}_0` in :math:`\mathbf{y}\;\;`.
        """

        return Y[(slice(None),)*self.blkaxis + (slice(0, self.blkidx),)]



    def block_sep1(self, Y):
        r"""Separate variable into component corresponding to
        :math:`\mathbf{y}_1` in :math:`\mathbf{y}\;\;`.
        """

        return Y[(slice(None),)*self.blkaxis + (slice(self.blkidx, None),)]



    def block_sep(self, Y):
        r"""Separate variable into components corresponding to blocks
        :math:`\mathbf{y}_0` and :math:`\mathbf{y}_1` in
        :math:`\mathbf{y}\;\;`.
        """

        return (self.block_sep0(Y), self.block_sep1(Y))



    def block_cat(self, Y0, Y1):
        r"""Concatenate components corresponding to :math:`\mathbf{y}_0` and
        :math:`\mathbf{y}_1` to form :math:`\mathbf{y}\;\;`.
        """

        return np.concatenate((Y0, Y1), axis=self.blkaxis)



    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        self.AXnr = self.cnst_A(self.X)
        if self.rlx == 1.0:
            self.AX = self.AXnr
        else:
            if not hasattr(self, '_cnst_c0'):
                self._cnst_c0 = self.cnst_c0()
            if not hasattr(self, '_cnst_c1'):
                self._cnst_c1 = self.cnst_c1()
            alpha = self.rlx
            self.AX = alpha*self.AXnr + \
                      (1-alpha)*self.block_cat(self.var_y0() + self._cnst_c0,
                                               self.var_y1() + self._cnst_c1)



    def var_y0(self):
        r"""Get :math:`\mathbf{y}_0` variable."""

        return self.block_sep0(self.Y)



    def var_y1(self):
        r"""Get :math:`\mathbf{y}_1` variable."""

        return self.block_sep1(self.Y)



    def obfn_fvar(self):
        """Variable to be evaluated in computing :meth:`ADMM.obfn_f`."""

        return self.X



    def obfn_g0var(self):
        """Variable to be evaluated in computing
        :meth:`ADMMTwoBlockCnstrnt.obfn_g0`, depending on the ``AuxVarObj``
        option value.
        """

        return self.var_y0() if self.opt['AuxVarObj'] else \
            self.cnst_A0(self.X) - self.cnst_c0()



    def obfn_g1var(self):
        """Variable to be evaluated in computing
        :meth:`ADMMTwoBlockCnstrnt.obfn_g1`, depending on the ``AuxVarObj``
        option value.
        """

        return self.var_y1() if self.opt['AuxVarObj'] else \
            self.cnst_A1(self.X) - self.cnst_c1()



    def obfn_f(self, X):
        r"""Compute :math:`f(\mathbf{x})` component of ADMM objective
        function. Unless overridden, :math:`f(\mathbf{x}) = 0`.
        """

        return 0.0



    def obfn_g(self, Y):
        r"""Compute :math:`g(\mathbf{y}) = g_0(\mathbf{y}_0) +
        g_1(\mathbf{y}_1)` component of ADMM objective function.
        """

        return self.obfn_g0(self.obfn_g0var()) + self.obfn_g1(self.obfn_g1var())



    def obfn_g0(self, Y0):
        r"""Compute :math:`g_0(\mathbf{y}_0)` component of ADMM objective
        function.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def obfn_g1(self, Y1):
        r"""Compute :math:`g_1(\mathbf{y_1})` component of ADMM objective
        function.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        fval = self.obfn_f(self.obfn_fvar())
        g0val = self.obfn_g0(self.obfn_g0var())
        g1val = self.obfn_g1(self.obfn_g1var())
        obj = fval + g0val + g1val
        return (obj, fval, g0val, g1val)



    def cnst_A(self, X):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint.
        """

        return self.block_cat(self.cnst_A0(X), self.cnst_A1(X))



    def cnst_AT(self, Y):
        r"""Compute :math:`A^T \mathbf{y}` where

        .. math::
           A^T \mathbf{y} = \left( \begin{array}{cc} A_0^T & A_1^T
           \end{array} \right) \left( \begin{array}{c} \mathbf{y}_0
           \\ \mathbf{y}_1 \end{array} \right) = A_0^T \mathbf{y}_0 +
           A_1^T \mathbf{y}_1 \;\;.
        """

        return self.cnst_A0T(self.block_sep0(Y)) + \
            self.cnst_A1T(self.block_sep1(Y))



    def cnst_B(self, Y):
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem constraint.
        In this case :math:`B \mathbf{y} = -\mathbf{y}` since the constraint
        is :math:`A \mathbf{x} - \mathbf{y} = \mathbf{c}`.
        """

        return -Y



    def cnst_c(self):
        r"""Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. This method should not be used or overridden: all
        calculations should make use of components :meth:`cnst_c0` and
        :meth:`cnst_c1` so that these methods can return scalar zeros
        instead of zero arrays if appropriate.
        """

        raise NotImplementedError()



    def cnst_c0(self):
        r"""Compute constant component :math:`\mathbf{c}_0` of
        :math:`\mathbf{c}` in the ADMM problem constraint. Unless overridden,
        :math:`\mathbf{c}_0 = 0`.
        """

        return 0.0



    def cnst_c1(self):
        r"""Compute constant component :math:`\mathbf{c}_1` of
        :math:`\mathbf{c}` in the ADMM problem constraint. Unless overridden,
        :math:`\mathbf{c}_1 = 0`.
        """

        return 0.0



    def cnst_A0(self, X):
        r"""Compute :math:`A_0 \mathbf{x}` component of :math:`A \mathbf{x}` in
        ADMM problem constraint (see :meth:`cnst_A`). Unless overridden,
        :math:`A_0 \mathbf{x} = \mathbf{x}`, i.e. :math:`A_0 = I`.
        """

        return X



    def cnst_A0T(self, Y0):
        r"""Compute :math:`A_0^T \mathbf{y}_0` component of
        :math:`A^T \mathbf{y}` (see :meth:`cnst_AT`). Unless overridden,
        :math:`A_0^T \mathbf{y}_0 = \mathbf{y}_0`, i.e. :math:`A_0 = I`.
        """

        return Y0



    def cnst_A1(self, X):
        r"""Compute :math:`A_1 \mathbf{x}` component of :math:`A \mathbf{x}`
        in ADMM problem constraint (see :meth:`cnst_A`). Unless overridden,
        :math:`A_1 \mathbf{x} = \mathbf{x}`, i.e. :math:`A_1 = I`.
        """

        return X



    def cnst_A1T(self, Y1):
        r"""Compute :math:`A_1^T \mathbf{y}_1` component of
        :math:`A^T \mathbf{y}` (see :meth:`cnst_AT`). Unless overridden,
        :math:`A_1^T \mathbf{y}_1 = \mathbf{y}_1`, i.e. :math:`A_1 = I`.
        """

        return Y1



    def rsdl_r(self, AX, Y):
        """Compute primal residual vector.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_c0` and :meth:`cnst_c1` are not
        overridden.
        """

        if not hasattr(self, '_cnst_c0'):
            self._cnst_c0 = self.cnst_c0()
        if not hasattr(self, '_cnst_c1'):
            self._cnst_c1 = self.cnst_c1()
        return AX - self.block_cat(self.var_y0() + self._cnst_c0,
                                   self.var_y1() + self._cnst_c1)



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        return self.rho*self.cnst_AT(Yprev - Y)



    def rsdl_rn(self, AX, Y):
        """Compute primal residual normalisation term.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        if not hasattr(self, '_cnst_nrm_c'):
            self._cnst_nrm_c = np.sqrt(linalg.norm(self.cnst_c0())**2 +
                                       linalg.norm(self.cnst_c1())**2)
        return max((linalg.norm(AX), linalg.norm(Y), self._cnst_nrm_c))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        return self.rho*linalg.norm(self.cnst_AT(U))
