# -*- coding: utf-8 -*-
# Copyright (C) 2015-2025 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Base classes for ADMM algorithms"""

from __future__ import division, print_function

import copy
import warnings
import numpy as np

from sporco import cdict
from sporco import util
from sporco.util import u
from sporco.array import transpose_ntpl_list
from sporco.fft import real_dtype
from sporco import common


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



class ADMM(common.IterativeSolver):
    r"""Base class for Alternating Direction Method of Multipliers (ADMM)
    algorithms :cite:`boyd-2010-distributed`.

    Solve an optimisation problem of the form

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

        defaults = {'FastSolve': False, 'Verbose': False,
                    'StatusHeader': True, 'DataType': None,
                    'MaxMainIter': 1000, 'IterTimer': 'solve',
                    'AbsStopTol': 0.0, 'RelStopTol': 1e-3,
                    'RelaxParam': 1.0, 'rho': None,
                    'AutoRho':
                    {
                        'Enabled': False, 'Period': 10,
                        'Scaling': 2.0, 'RsdlRatio': 10.0,
                        'RsdlTarget': None, 'AutoScaling': False,
                        'StdResiduals': False
                    },
                    'Y0': None, 'U0': None, 'Callback': None
                   }

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ADMM algorithm options
            """

            if opt is None:
                opt = {}
            cdict.ConstrainedDict.__init__(self, opt)



    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""

    itstat_fields_objfn = ('ObjFun', 'FVal', 'GVal')
    """Fields in IterationStats associated with the objective function;
    see :meth:`eval_objfn`"""
    itstat_fields_alg = ('PrimalRsdl', 'DualRsdl', 'EpsPrimal', 'EpsDual',
                         'Rho')
    """Fields in IterationStats associated with the specific solver
    algorithm"""
    itstat_fields_extra = ()
    """Non-standard fields in IterationStats; see :meth:`itstat_extra`"""

    hdrtxt_objfn = ('Fnc', 'f', 'g')
    """Display column headers associated with the objective function;
    see :meth:`eval_objfn`"""
    hdrval_objfun = {'Fnc': 'ObjFun', 'f': 'FVal', 'g': 'GVal'}
    """Dictionary mapping display column headers in :attr:`hdrtxt_objfn`
    to IterationStats entries"""



    def __new__(cls, *args, **kwargs):
        """Create an ADMM object and start its initialisation timer."""

        instance = super(ADMM, cls).__new__(cls)
        instance.timer = util.Timer(['init', 'solve', 'solve_wo_func',
                                     'solve_wo_rsdl'])
        instance.timer.start('init')
        return instance



    def __init__(self, Nx, yshape, ushape, dtype, opt=None):
        r"""
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
            raise TypeError('Parameter opt must be an instance of '
                            'ADMM.Options')

        self.opt = opt
        self.Nx = Nx
        # Working variable U has the same dimensionality as constant c
        # in the constraint Ax + By = c
        self.Nc = np.prod(np.array(ushape))

        # DataType option overrides data type inferred from __init__
        # parameters of derived class
        self.set_dtype(opt, dtype)

        # Initialise attributes representing penalty parameter and other
        # parameters
        rdt = real_dtype(self.dtype)
        self.set_attr('rho', opt['rho'], dval=1.0, dtype=rdt)
        self.set_attr('rho_tau', opt['AutoRho', 'Scaling'], dval=2.0,
                      dtype=rdt)
        self.set_attr('rho_mu', opt['AutoRho', 'RsdlRatio'], dval=10.0,
                      dtype=rdt)
        self.set_attr('rho_xi', opt['AutoRho', 'RsdlTarget'], dval=1.0,
                      dtype=rdt)
        self.set_attr('rlx', opt['RelaxParam'], dval=1.0, dtype=rdt)


        # Initialise working variable X
        if not hasattr(self, 'X'):
            self.X = None

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



    def yinit(self, yshape):
        """Return initialiser for working variable Y"""

        return np.zeros(yshape, dtype=self.dtype)



    def uinit(self, ushape):
        """Return initialiser for working variable U"""

        return np.zeros(ushape, dtype=self.dtype)



    def solve(self):
        """Start (or re-start) optimisation. This method implements the
        framework for the iterations of an ADMM algorithm. There is
        sufficient flexibility in overriding the component methods that
        it calls that it is usually not necessary to override this method
        in derived clases.

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
        for self.k in range(self.k, self.k + self.opt['MaxMainIter']):

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
                itst = self.iteration_stats(self.k, r, s, epri, edua)
                self.itstat.append(itst)
                self.display_status(fmtstr, itst)
            self.timer.start(['solve_wo_func', 'solve_wo_rsdl'])

            # Automatic rho adjustment
            self.timer.stop('solve_wo_rsdl')
            if self.opt['AutoRho', 'Enabled'] or not self.opt['FastSolve']:
                self.update_rho(self.k, r, s)
            self.timer.start('solve_wo_rsdl')

            # Call callback function if defined
            if self.opt['Callback'] is not None:
                if self.opt['Callback'](self):
                    break

            # Stop if residual-based stopping tolerances reached
            if self.opt['AutoRho', 'Enabled'] or not self.opt['FastSolve']:
                if r < epri and s < edua:
                    break


        # Increment iteration count
        self.k += 1

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

        warnings.warn("admm.ADMM.runtime attribute has been replaced by "
                      "an upgraded timer class: please see the documentation "
                      "for admm.ADMM.solve method and util.Timer class",
                      PendingDeprecationWarning)
        return self.timer.elapsed('init') + self.timer.elapsed('solve')



    def getmin(self):
        """Get minimiser after optimisation."""

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
            self.AX = alpha*self.AXnr - (1 - alpha)*(self.cnst_B(self.Y) -
                                                     self._cnst_c)



    def compute_residuals(self):
        """Compute residuals and stopping thresholds."""

        if self.opt['AutoRho', 'StdResiduals']:
            r = np.linalg.norm(self.rsdl_r(self.AXnr, self.Y))
            s = np.linalg.norm(self.rsdl_s(self.Yprev, self.Y))
            epri = np.sqrt(self.Nc) * self.opt['AbsStopTol'] + \
                self.rsdl_rn(self.AXnr, self.Y) * self.opt['RelStopTol']
            edua = np.sqrt(self.Nx) * self.opt['AbsStopTol'] + \
                self.rsdl_sn(self.U) * self.opt['RelStopTol']
        else:
            rn = self.rsdl_rn(self.AXnr, self.Y)
            if rn == 0.0:
                rn = 1.0
            sn = self.rsdl_sn(self.U)
            if sn == 0.0:
                sn = 1.0
            r = np.linalg.norm(self.rsdl_r(self.AXnr, self.Y)) / rn
            s = np.linalg.norm(self.rsdl_s(self.Yprev, self.Y)) / sn
            epri = np.sqrt(self.Nc) * self.opt['AbsStopTol'] / rn + \
                self.opt['RelStopTol']
            edua = np.sqrt(self.Nx) * self.opt['AbsStopTol'] / sn + \
                self.opt['RelStopTol']

        return r, s, epri, edua



    @classmethod
    def hdrtxt(cls):
        """Construct tuple of status display column titles."""

        return ('Itn',) + cls.hdrtxt_objfn + ('r', 's', u('ρ'))



    @classmethod
    def hdrval(cls):
        """Construct dictionary mapping display column title to
        IterationStats entries.
        """

        hdrmap = {'Itn': 'Iter'}
        hdrmap.update(cls.hdrval_objfun)
        hdrmap.update({'r': 'PrimalRsdl', 's': 'DualRsdl', u('ρ'): 'Rho'})
        return hdrmap



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

        return transpose_ntpl_list(self.itstat)



    def update_rho(self, k, r, s):
        """Automatic rho adjustment."""

        if self.opt['AutoRho', 'Enabled']:
            tau = self.rho_tau
            mu = self.rho_mu
            xi = self.rho_xi
            if k != 0 and np.mod(k + 1, self.opt['AutoRho', 'Period']) == 0:
                if self.opt['AutoRho', 'AutoScaling']:
                    if s == 0.0 or r == 0.0:
                        rhomlt = tau
                    else:
                        rhomlt = np.sqrt(r / (s * xi) if r > s * xi else
                                         (s * xi) / r)
                        if rhomlt > tau:
                            rhomlt = tau
                else:
                    rhomlt = tau
                rsf = 1.0
                if r > xi * mu * s:
                    rsf = rhomlt
                elif s > (mu / xi) * r:
                    rsf = 1.0 / rhomlt
                self.rho *= real_dtype(self.dtype).type(rsf)
                self.U /= rsf
                if rsf != 1.0:
                    self.rhochange()



    def display_start(self):
        """Set up status display if option selected. NB: this method
        assumes that the first entry is the iteration count and the last
        is the rho value.
        """

        if self.opt['Verbose']:
            # If AutoRho option enabled rho is included in iteration status
            if self.opt['AutoRho', 'Enabled']:
                hdrtxt = type(self).hdrtxt()
            else:
                hdrtxt = type(self).hdrtxt()[0:-1]
            # Call utility function to construct status display formatting
            hdrstr, fmtstr, nsep = common.solve_status_str(
                hdrtxt, fwdth0=type(self).fwiter, fprec=type(self).fpothr)
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

        Overriding this method is required if :meth:`eval_objfn`
        is not overridden.
        """

        raise NotImplementedError()



    def obfn_g(self, Y):
        r"""Compute :math:`g(\mathbf{y})` component of ADMM objective function.

        Overriding this method is required if :meth:`eval_objfn`
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

        return self.rho * self.cnst_AT(self.cnst_B(Y - Yprev))



    def rsdl_rn(self, AX, Y):
        """Compute primal residual normalisation term.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        # Avoid computing the norm of the value returned by cnst_c()
        # more than once
        if not hasattr(self, '_nrm_cnst_c'):
            self._nrm_cnst_c = np.linalg.norm(self.cnst_c())
        return max((np.linalg.norm(AX), np.linalg.norm(self.cnst_B(Y)),
                    self._nrm_cnst_c))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        return self.rho * np.linalg.norm(self.cnst_AT(U))



    def rhochange(self):
        """Action to be taken, if any, when rho parameter is changed.

        Overriding this method is optional.
        """

        pass





class ADMMEqual(ADMM):
    r"""
    Base class for ADMM algorithms with a simple equality constraint.

    |

    .. inheritance-diagram:: ADMMEqual
       :parts: 2

    |

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

          ``fEvalX`` : Flag indicating whether the :math:`f` component of
          the objective function should be evaluated using variable X
          (``True``) or Y (``False``) as its argument.

          ``gEvalY`` : Flag indicating whether the :math:`g` component of
          the objective function should be evaluated using variable Y
          (``True``) or X (``False``) as its argument.

          ``ReturnX`` : Flag indicating whether the return value of the
          solve method is the X variable (``True``) or the Y variable
          (``False``).
        """

        defaults = copy.deepcopy(ADMM.Options.defaults)
        defaults.update({'fEvalX': True, 'gEvalY': True, 'ReturnX': True})

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ADMMEqual algorithm options
            """

            if opt is None:
                opt = {}
            ADMM.Options.__init__(self, opt)




    def __init__(self, xshape, dtype, opt=None):
        """
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
        Nx = np.prod(np.array(xshape))
        super(ADMMEqual, self).__init__(Nx, xshape, xshape, dtype, opt)



    def getmin(self):
        """Get minimiser after optimisation."""

        return self.X if self.opt['ReturnX'] else self.Y



    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        self.AXnr = self.X
        if self.rlx == 1.0:
            self.AX = self.X
        else:
            alpha = self.rlx
            self.AX = alpha*self.X + (1 - alpha)*self.Y



    def obfn_fvar(self):
        """Variable to be evaluated in computing :meth:`ADMM.obfn_f`,
        depending on the ``fEvalX`` option value.
        """

        return self.X if self.opt['fEvalX'] else self.Y



    def obfn_gvar(self):
        """Variable to be evaluated in computing :meth:`ADMM.obfn_g`,
        depending on the ``gEvalY`` option value.
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
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint. In this case :math:`A \mathbf{x} = \mathbf{x}` since
        the constraint is :math:`\mathbf{x} = \mathbf{y}`.
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
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem
        constraint. In this case :math:`B \mathbf{y} = -\mathbf{y}` since
        the constraint is :math:`\mathbf{x} = \mathbf{y}`.
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

        return self.rho * (Yprev - Y)



    def rsdl_rn(self, AX, Y):
        """Compute primal residual normalisation term."""

        return max((np.linalg.norm(AX), np.linalg.norm(Y)))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho * np.linalg.norm(U)





class ADMMTwoBlockCnstrnt(ADMM):
    r"""
    Base class for ADMM algorithms for problems for which
    :math:`g(\mathbf{y}) = g_0(\mathbf{y}_0) + g_1(\mathbf{y}_1)` with
    :math:`\mathbf{y}^T = (\mathbf{y}_0^T \; \mathbf{y}_1^T)`.

    |

    .. inheritance-diagram:: ADMMTwoBlockCnstrnt
       :parts: 2

    |

    Solve optimisation problems of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x}} \; f(\mathbf{x}) + g_0(A_0 \mathbf{x})
       + g_1(A_1 \mathbf{x})

    via an ADMM problem of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
       f(\mathbf{x}) + g_0(\mathbf{y}_0) + g_0(\mathbf{y}_1)
       \;\text{such that}\;
       \left( \begin{array}{c} A_0 \\ A_1 \end{array} \right) \mathbf{x}
       - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1 \end{array}
       \right) = \left( \begin{array}{c} \mathbf{c}_0 \\
       \mathbf{c}_1 \end{array} \right) \;\;.

    In this case the ADMM constraint is :math:`A\mathbf{x} + B\mathbf{y}
    = \mathbf{c}` where

    .. math::
       A = \left( \begin{array}{c} A_0 \\ A_1 \end{array} \right)
       \qquad B = -I \qquad \mathbf{y} = \left( \begin{array}{c}
       \mathbf{y}_0 \\ \mathbf{y}_1 \end{array} \right) \qquad
       \mathbf{c} = \left( \begin{array}{c} \mathbf{c}_0 \\
       \mathbf{c}_1 \end{array} \right) \;\;.

    This class specialises class :class:`.ADMM`, but remains a base class
    for other classes that specialise to specific optimisation problems.
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
        defaults.update({'AuxVarObj': False, 'ReturnVar': 'X'})

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ADMMTwoBlockCnstrnt algorithm options
            """

            if opt is None:
                opt = {}
            ADMM.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'FVal', 'G0Val', 'G1Val')
    """Fields in IterationStats associated with the objective function;
    see :meth:`eval_objfn`"""

    hdrtxt_objfn = ('Fnc', 'f', 'g0', 'g1')
    """Display column headers associated with the objective function;
    see :meth:`eval_objfn`"""
    hdrval_objfun = {'Fnc': 'ObjFun', 'f': 'FVal',
                     'g0': 'G0Val', 'g1': 'G1Val'}
    """Dictionary mapping display column headers in :attr:`hdrtxt_objfn`
    to IterationStats entries"""



    def __init__(self, Nx, yshape, blkaxis, blkidx, dtype, opt=None):
        r"""
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
        self.blkaxis = blkaxis
        self.blkidx = blkidx
        super(ADMMTwoBlockCnstrnt, self).__init__(Nx, yshape, yshape,
                                                  dtype, opt)



    def getmin(self):
        """Get minimiser after optimisation."""

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
        r"""Concatenate components corresponding to :math:`\mathbf{y}_0`
        and :math:`\mathbf{y}_1` to form :math:`\mathbf{y}\;\;`.
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
            self.AX = alpha*self.AXnr + (1 - alpha)*self.block_cat(
                self.var_y0() + self._cnst_c0,
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

        return self.obfn_g0(self.obfn_g0var()) + \
            self.obfn_g1(self.obfn_g1var())



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
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem
        constraint. In this case :math:`B \mathbf{y} = -\mathbf{y}` since
        the constraint is :math:`A \mathbf{x} - \mathbf{y} = \mathbf{c}`.
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
        :math:`\mathbf{c}` in the ADMM problem constraint. Unless
        overridden, :math:`\mathbf{c}_0 = 0`.
        """

        return 0.0



    def cnst_c1(self):
        r"""Compute constant component :math:`\mathbf{c}_1` of
        :math:`\mathbf{c}` in the ADMM problem constraint. Unless
        overridden, :math:`\mathbf{c}_1 = 0`.
        """

        return 0.0



    def cnst_A0(self, X):
        r"""Compute :math:`A_0 \mathbf{x}` component of :math:`A \mathbf{x}`
        in ADMM problem constraint (see :meth:`cnst_A`). Unless overridden,
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
        return AX - self.block_cat(self.block_sep0(Y) + self._cnst_c0,
                                   self.block_sep1(Y) + self._cnst_c1)



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        return self.rho * self.cnst_AT(Yprev - Y)



    def rsdl_rn(self, AX, Y):
        """Compute primal residual normalisation term.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        if not hasattr(self, '_cnst_nrm_c'):
            self._cnst_nrm_c = np.sqrt(np.linalg.norm(self.cnst_c0())**2 +
                                       np.linalg.norm(self.cnst_c1())**2)
        return max((np.linalg.norm(AX), np.linalg.norm(Y), self._cnst_nrm_c))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        return self.rho * np.linalg.norm(self.cnst_AT(U))





class ADMMConsensus(ADMM):
    r"""
    Base class for ADMM algorithms with a global variable consensus
    structure (see Ch. 7 of :cite:`boyd-2010-distributed`).

    |

     .. inheritance-diagram:: ADMMConsensus
        :parts: 2

    |

    Solve optimisation problems of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x}} \; \sum_i f_i(\mathbf{x}) + g(\mathbf{x})

    via an ADMM problem of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x}_i,\mathbf{y}} \;
       \sum_i f(\mathbf{x}_i) + g(\mathbf{y}) \;\mathrm{such\;that}\;
       \left( \begin{array}{c} \mathbf{x}_0 \\ \mathbf{x}_1 \\
       \vdots \end{array} \right) = \left( \begin{array}{c}
       I \\ I \\ \vdots \end{array} \right) \mathbf{y} \;\;.

    This class specialises class ADMM, but remains a base class for
    other classes that specialise to specific optimisation problems.
    """

    class Options(ADMM.Options):
        """ADMMConsensus algorithm options.

        Options include all of those defined in :class:`ADMM.Options`,
        together with additional options:

          ``fEvalX`` : Flag indicating whether the :math:`f` component
          of the objective function should be evaluated using variable
          X (``True``) or Y (``False``) as its argument.

          ``gEvalY`` : Flag indicating whether the :math:`g` component of
          the objective function should be evaluated using variable Y
          (``True``) or X (``False``) as its argument.

          ``AuxVarObj`` : Flag selecting choices of ``fEvalX`` and
          ``gEvalY`` that give meaningful functional values. If ``True``,
          ``fEvalX`` and ``gEvalY`` are set to ``False`` and ``True``
          respectively, and vice versa if ``False``. Setting this flag
          to ``True`` often gives a better estimate of the objective
          function, at some additional computational cost.
        """

        defaults = copy.deepcopy(ADMM.Options.defaults)
        defaults.update({'fEvalX': False, 'gEvalY': True,
                         'AuxVarObj': True})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ADMMConsensus algorithm options
            """

            if opt is None:
                opt = {}
            ADMM.Options.__init__(self, opt)



        def __setitem__(self, key, value):
            """Set options 'fEvalX' and 'gEvalY' appropriately when option
            'AuxVarObj' is set.
            """

            ADMM.Options.__setitem__(self, key, value)

            if key == 'AuxVarObj':
                if value is True:
                    self['fEvalX'] = False
                    self['gEvalY'] = True
                else:
                    self['fEvalX'] = True
                    self['gEvalY'] = False




    def __init__(self, Nb, yshape, dtype, opt=None):
        r"""
        Parameters
        ----------
        yshape : tuple
          Shape of variable :math:`\mathbf{y}` in objective function
        Nb : int
          Number of blocks / consensus components
        opt : :class:`ADMMConsensus.Options` object
          Algorithm options
        """

        if opt is None:
            opt = ADMMConsensus.Options()
        self.Nb = Nb
        self.xshape = yshape + (Nb,)
        self.yshape = yshape
        Nx = Nb * np.prod(yshape)
        super(ADMMConsensus, self).__init__(Nx, yshape, self.xshape,
                                            dtype, opt)



    def getmin(self):
        """Get minimiser after optimisation."""

        return self.Y



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to block vector
        :math:`\mathbf{x} = \left( \begin{array}{ccc} \mathbf{x}_0^T &
        \mathbf{x}_1^T & \ldots \end{array} \right)^T\;`.
        """

        for i in range(self.Nb):
            self.xistep(i)



    def xistep(self, i):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{x}`
        component :math:`\mathbf{x}_i`.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{y}`.
        """

        rho = self.Nb * self.rho
        mAXU = np.mean(self.AX + self.U, axis=-1)
        self.Y[:] = self.prox_g(mAXU, rho)



    def prox_g(self, X, rho):
        r"""Proximal operator of :math:`\rho^{-1} g(\cdot)`.

        Overriding this method is required. Note that this method
        should compute the proximal operator of
        :math:`\rho^{-1} g(\cdot)`, *not* the proximal operator
        of :math:`\rho g(\cdot)`.
        """

        raise NotImplementedError()



    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        self.AXnr = self.X
        if self.rlx == 1.0:
            self.AX = self.X
        else:
            alpha = self.rlx
            self.AX = alpha*self.X + (1 - alpha)*self.Y[..., np.newaxis]



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        fval = self.obfn_f()
        gval = self.obfn_g(self.obfn_gvar())
        obj = fval + gval
        return (obj, fval, gval)



    def obfn_fvar(self, i):
        r"""Variable to be evaluated in computing :math:`f_i(\cdot)`,
        depending on the ``fEvalX`` option value.
        """

        return self.X[..., i] if self.opt['fEvalX'] else self.Y



    def obfn_gvar(self):
        r"""Variable to be evaluated in computing :math:`g(\cdot)`,
        depending on the ``gEvalY`` option value.
        """

        return self.Y if self.opt['gEvalY'] else np.mean(self.X, axis=-1)



    def obfn_f(self):
        r"""Compute :math:`f(\mathbf{x}) = \sum_i f(\mathbf{x}_i)`
        component of ADMM objective function.
        """

        obf = 0.0
        for i in range(self.Nb):
            obf += self.obfn_fi(self.obfn_fvar(i), i)
        return obf



    def obfn_fi(self, X, i):
        r"""Compute :math:`f(\mathbf{x}_i)` component of ADMM objective
        function.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def rsdl_r(self, AX, Y):
        """Compute primal residual vector."""

        return AX - Y[..., np.newaxis]



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        # Since s = rho A^T B (y^(k+1) - y^(k)) and B = -(I I I ...)^T,
        # the correct calculation here would involve replicating (Yprev - Y)
        # on the axis on which the blocks of X are stacked. Since this would
        # require allocating additional memory, and since it is only the norm
        # of s that is required, instead of replicating the vector it is
        # scaled to have the same l2 norm as the replicated vector
        return np.sqrt(self.Nb) * self.rho * (Yprev - Y)



    def rsdl_rn(self, AX, Y):
        """Compute primal residual normalisation term."""

        # The primal residual normalisation term is
        # max( ||A x^(k)||_2, ||B y^(k)||_2 ) and B = -(I I I ...)^T.
        # The scaling by sqrt(Nb) of the l2 norm of Y accounts for the
        # block replication introduced by multiplication by B
        return max((np.linalg.norm(AX), np.sqrt(self.Nb) * np.linalg.norm(Y)))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho * np.linalg.norm(U)





class WeightedADMMConsensus(ADMMConsensus):
    r"""
    Base class for ADMM algorithms with a global variable consensus
    structure (see Ch. 7 of :cite:`boyd-2010-distributed`), including
    scalar weighting of each component as in Equ. (2) in
    :cite:`buzzard-2018-plug`

    |

     .. inheritance-diagram:: WeightedADMMConsensus
        :parts: 2

    |

    Solve optimisation problems of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x}} \; \sum_i \alpha_i f_i(\mathbf{x}) +
       g(\mathbf{x})

    via an ADMM problem of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x}_i,\mathbf{y}} \;
       \sum_i \alpha_i f(\mathbf{x}_i) + g(\mathbf{y})
       \;\mathrm{such\;that}\; \left( \begin{array}{c} \sqrt{\alpha_0}
       \mathbf{x}_0 \\ \sqrt{\alpha_1} \mathbf{x}_1 \\
       \vdots \end{array} \right) = \left( \begin{array}{c}
       \sqrt{\alpha_0} I \\ \sqrt{\alpha_1} I \\ \vdots \end{array}
       \right) \mathbf{y} \;\;.

    This class specialises class ADMMConsensus, but remains a base class
    for other classes that specialise to specific optimisation problems.
    """

    class Options(ADMMConsensus.Options):
        """WeightedADMMConsensus algorithm options.

        Options are the same as those defined in
        :class:`ADMMConsensus.Options`.
        """

        defaults = copy.deepcopy(ADMMConsensus.Options.defaults)


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              WeightedADMMConsensus algorithm options
            """

            if opt is None:
                opt = {}
            ADMMConsensus.Options.__init__(self, opt)



    def __init__(self, Nb, alpha, yshape, dtype, opt=None):
        r"""
        Parameters
        ----------
        yshape : tuple
          Shape of variable :math:`\mathbf{y}` in objective function
        Nb : int
          Number of blocks / consensus components
        alpha : array_like
          Array of component weights
        opt : :class:`WeightedADMMConsensus.Options` object
          Algorithm options
        """

        if opt is None:
            opt = WeightedADMMConsensus.Options()
        self.alpha = alpha
        super(WeightedADMMConsensus, self).__init__(Nb, yshape, dtype, opt)
        self.alphaxd = alpha.reshape((1,) * (self.U.ndim-1) + (alpha.size,))



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{y}`.
        """

        asum = np.sum(self.alpha)
        rho = asum * self.rho
        wmAXU = np.average(self.AX + self.U, axis=-1, weights=self.alpha)
        self.Y[:] = self.prox_g(wmAXU, rho)



    def ustep(self):
        """Dual variable update."""

        self.U += self.AX - self.Y[..., np.newaxis]



    def obfn_gvar(self):
        r"""Variable to be evaluated in computing :math:`g(\cdot)`,
        depending on the ``gEvalY`` option value.
        """

        return self.Y if self.opt['gEvalY'] else \
            np.average(self.X, axis=-1, weights=self.alpha)



    def obfn_f(self):
        r"""Compute :math:`f(\mathbf{x}) = \sum_i \alpha_i
        f(\mathbf{x}_i)` component of ADMM objective function.
        """

        obf = 0.0
        for i in range(self.Nb):
            obf += self.alpha[i] * self.obfn_fi(self.obfn_fvar(i), i)
        return obf



    def rsdl_r(self, AX, Y):
        """Compute primal residual vector."""

        return np.sqrt(self.alphaxd) * (AX - Y[..., np.newaxis])



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        # Since s = rho A^T B (y^(k+1) - y^(k)) and (without scaling
        # factors alpha) B = -(I I I ...)^T, the correct calculation
        # here would involve replicating (Yprev - Y) on the axis on
        # which the blocks of X are stacked. Since this would require
        # allocating additional memory, and since it is only the norm
        # of s that is required, instead of replicating the vector it is
        # scaled to have the same l2 norm as the replicated vector
        return self.rho * np.sqrt(np.sum(self.alpha**2)) * (Yprev - Y)



    def rsdl_rn(self, AX, Y):
        """Compute primal residual normalisation term."""

        # The primal residual normalisation term is
        # max( ||A x^(k)||_2, ||B y^(k)||_2 ) and (without scaling
        # factors alpha) B = -(I I I ...)^T.
        # The scaling by factor for the l2 norm of Y accounts for the
        # block replication introduced by multiplication by the B
        # including the alpha weighting factors
        return max((np.linalg.norm(np.sqrt(self.alphaxd) * AX),
                    np.sqrt(np.sum(self.alpha**2)) * np.linalg.norm(Y)))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho * np.linalg.norm(U)
