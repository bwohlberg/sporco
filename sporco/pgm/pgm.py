# -*- coding: utf-8 -*-
# Copyright (C) 2016-2020 by Cristina Garcia-Cardona <cgarciac@lanl.gov>
#                            Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Base classes for PGM algorithms."""

from __future__ import division, print_function
from builtins import range

import copy
import numpy as np

from sporco.cdict import ConstrainedDict
from sporco.common import IterativeSolver, solve_status_str
from sporco.fft import rfftn, irfftn
from sporco.array import transpose_ntpl_list
from sporco.util import Timer

from .backtrack import BacktrackRobust
from .momentum import MomentumNesterov
from .stepsize import StepSizePolicyBB

__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""


__all__ = ['PGM', 'PGMDFT']



class PGM(IterativeSolver):
    r"""Base class for Proximal Gradient Method (PGM) algorithms
    (see for example Ch. 10 of :cite:`beck-2017-first` and
    Sec. 4.2 and 4.3 of :cite:`parikh-2014-proximal`). Algorithms
    such as FISTA :cite:`beck-2009-fast` and a robust variant of
    FISTA :cite:`florea-2017-robust` are also supported.

    Solve optimisation problems of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x}} \; f(\mathbf{x}) + g(\mathbf{x}) \;\;,

    where :math:`f, g` are convex functions and :math:`f` is smooth.

    This class is intended to be a base class of other classes that
    specialise to specific optimisation problems.

    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The default fields of the named tuple
    ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``FVal`` :  Value of smooth objective function component :math:`f`

       ``GVal`` : Value of objective function component :math:`g`

       ``F_Btrack`` : Value of objective function :math:`f + g`
       (see Sec. 2.2 of :cite:`beck-2009-fast`) when backtracking

       ``Q_Btrack`` : Value of Quadratic approximation :math:`Q_L`
       (see Sec. 2.3 of :cite:`beck-2009-fast`) when backtracking

       ``IterBtrack`` : Number of iterations in backtracking

       ``Rsdl`` : Residual

       ``L`` : Inverse of gradient step parameter

       ``Time`` : Cumulative run time
    """

    class Options(ConstrainedDict):
        r"""PGM algorithm options.

        Options:

          ``FastSolve`` : Flag determining whether non-essential
          computation is skipped. When ``FastSolve`` is ``True`` and
          ``Verbose`` is ``False``, the functional value and related
          iteration statistics are not computed. If ``FastSolve`` is
          ``True`` residuals are also not calculated, in which case the
          residual-based stopping method is also disabled, with the
          number of iterations determined only by ``MaxMainIter``.

          ``Verbose`` : Flag determining whether iteration status is
          displayed.

          ``StatusHeader`` : Flag determining whether status header and
          separator are displayed.

          ``DataType`` : Specify data type for solution variables,
          e.g. ``np.float32``.

          ``X0`` : Initial value for X variable.

          ``Callback`` : Callback function to be called at the end of
          every iteration.

          ``MaxMainIter`` : Maximum main iterations.

          ``IterTimer`` : Label of the timer to use for iteration times.

          ``RelStopTol`` : Relative convergence tolerance for fixed point
          residual (see Sec. 4.3 of :cite:`liu-2018-first`).

          ``L`` : Inverse of gradient step parameter :math:`L`.

          ``AutoStop`` : Options for adaptive stopping strategy (fixed
          point residual, see Sec. 4.3 of :cite:`liu-2018-first`).

            ``Enabled`` : Flag determining whether the adaptive stopping
            relative parameter strategy is enabled.

            ``Tau0`` : numerator in adaptive criterion
            (:math:`\tau_0` in :cite:`liu-2018-first`).

          ``Monotone`` : Flag determining whether a monotone PGM version
          from :cite:`beck-2009-tv` is used. Default is False.

          ``Momentum`` : Momentum coefficient adaptation object. Standard
          options are Nesterov :cite:`beck-2009-fast`
          (:class:`.MomentumNesterov`), Linear
          :cite:`chambolle-2015-convergence`
          (:class:`.MomentumLinear`), and GenLinear
          :cite:`rodriguez-2019-convergence`
          (:class:`.MomentumGenLinear`), but a custom class derived
          from :class:`.MomentumBase` may also be specified. Default
          is :class:`.MomentumNesterov`.

          ``StepSizePolicy`` : non-iterative L adaptation object.
          Standard options are Cauchy :cite:`yuan-2008-stepsize`
          Sec. 3 (:class:`.StepSizePolicyCauchy`), and Barzilai-Borwein
          :cite:`barzilai-1988-stepsize`
          (:class:`.StepSizePolicyBB`), but a custom class derived
          from :class:`.StepSizePolicyBase` may also be specified.
          Default is None, no non-iterative L adaptation. Note that in
          case that both step size and Backtrack strategies are enabled
          only Backtrack will be used.

          ``Backtrack`` : PGM backtracking options. Options are Standard
          :cite:`beck-2009-fast` (:class:`.BacktrackStandard`) and
          Robust :cite:`florea-2017-robust` (:class:`.BacktrackRobust`),
          but a custom class derived from :class:`.BacktrackBase` may
          also be specified. Default is None, no backtracking. Note that
          in case that both step size and Backtrack strategies
          are enabled only Backtrack will be used.

        """

        defaults = {'FastSolve': False, 'Verbose': False,
                    'StatusHeader': True, 'DataType': None,
                    'X0': None, 'Callback': None,
                    'MaxMainIter': 1000, 'IterTimer': 'solve',
                    'RelStopTol': 1e-3, 'L': None,
                    'AutoStop': {'Enabled': False, 'Tau0': 1e-2},
                    'Monotone': False,
                    'Momentum': MomentumNesterov(),
                    'StepSizePolicy': None,
                    'Backtrack': None}

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              PGM algorithm options
            """

            if opt is None:
                opt = {}
            ConstrainedDict.__init__(self, opt)



    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""

    itstat_fields_objfn = ('ObjFun', 'FVal', 'GVal')
    """Fields in IterationStats associated with the objective function;
    see :meth:`eval_objfun`"""
    itstat_fields_alg = ('Rsdl', 'F_Btrack', 'Q_Btrack', 'IterBTrack', 'L')
    """Fields in IterationStats associated with the specific solver
    algorithm"""
    itstat_fields_extra = ()
    """Non-standard fields in IterationStats; see :meth:`itstat_extra`"""

    hdrtxt_objfn = ('Fnc', 'f', 'g')
    """Display column headers associated with the objective function;
    see :meth:`eval_objfun`"""
    hdrval_objfun = {'Fnc': 'ObjFun', 'f': 'FVal', 'g': 'GVal'}
    """Dictionary mapping display column headers in :attr:`hdrtxt_objfn`
    to IterationStats entries"""



    def __new__(cls, *args, **kwargs):
        """Create a PGM object and start its initialisation timer."""

        instance = super(PGM, cls).__new__(cls)
        instance.timer = Timer(['init', 'solve', 'solve_wo_func',
                                'solve_wo_rsdl', 'solve_wo_btrack'])
        instance.timer.start('init')
        return instance



    def __init__(self, xshape, dtype, opt=None):
        r"""
        Parameters
        ----------
        xshape : tuple of ints
          Shape of working variable X
        dtype : data-type
          Data type for working variables (overridden by 'DataType' option)
        opt : :class:`PGM.Options` object
          Algorithm options
        """

        if opt is None:
            opt = PGM.Options()
        if not isinstance(opt, PGM.Options):
            raise TypeError("Parameter opt must be an instance of "
                            "PGM.Options")

        self.opt = opt

        # DataType option overrides data type inferred from __init__
        # parameters of derived class
        self.set_dtype(opt, dtype)

        # Initialise attributes representing step parameter and other
        # parameters
        self.set_attr('L', opt['L'], dval=1.0, dtype=self.dtype)

        # Configure policy for step size
        # Step size policy is turned off if Backtrack is enabled
        self.stepsizepolicy = self.opt['StepSizePolicy']
        if self.opt['Backtrack'] is not None:
            self.stepsizepolicy = None

        # Configure Momentum coefficients
        self.momentum = self.opt['Momentum']

        # If using adaptative stop criterion, set tau0 parameter
        if self.opt['AutoStop', 'Enabled']:
            self.tau0 = self.opt['AutoStop', 'Tau0']

        # Initialise working variable X
        if self.opt['X0'] is None:
            self.X = self.xinit(xshape)
        else:
            self.X = self.opt['X0'].astype(self.dtype, copy=True)

        # Default values for variables created only if Backtrack is enabled
        self.F = None
        self.Q = None
        self.iterBTrack = None
        self.backtrack = self.opt['Backtrack']

        self.Y = None

        self.itstat = []
        self.k = 0
        self.t = 1



    def xinit(self, xshape):
        """Return initialiser for working variable X."""

        return np.zeros(xshape, dtype=self.dtype)



    def solve(self):
        """Start (or re-start) optimisation. This method implements the
        framework for the iterations of a PGM algorithm. There is
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
          to compute residuals

          ``solve_wo_btrack`` : Total time taken by call(s) to
          :meth:`solve`, excluding time taken to compute functional
          value and related iteration statistics as well as time take
          to compute residuals and implemented ``Backtrack`` mechanism
        """

        # Open status display
        fmtstr, nsep = self.display_start()

        # Start solve timer
        self.timer.start(['solve', 'solve_wo_func', 'solve_wo_rsdl',
                          'solve_wo_btrack'])

        # Main optimisation iterations
        for self.k in range(self.k, self.k + self.opt['MaxMainIter']):

            # Update record of X and Y from previous iteration
            self.on_iteration_start()

            # Compute backtracking
            if self.opt['Backtrack'] is not None and self.k >= 0:
                self.timer.stop('solve_wo_btrack')
                # Compute backtracking
                self.backtrack.update(self)
                self.timer.start('solve_wo_btrack')
            else:
                # Compute just proximal step
                self.xstep()
                # Update by combining previous iterates
                self.ystep()

            # Compute residuals and stopping thresholds
            self.timer.stop(['solve_wo_rsdl', 'solve_wo_btrack'])
            if not self.opt['FastSolve']:
                frcxd, adapt_tol = self.compute_residuals()
            self.timer.start('solve_wo_rsdl')

            # Compute and record other iteration statistics and
            # display iteration stats if Verbose option enabled
            self.timer.stop(['solve_wo_func', 'solve_wo_rsdl',
                             'solve_wo_btrack'])
            if not self.opt['FastSolve']:
                itst = self.iteration_stats(self.k, frcxd)
                self.itstat.append(itst)
                self.display_status(fmtstr, itst)
            self.timer.start(['solve_wo_func', 'solve_wo_rsdl',
                              'solve_wo_btrack'])

            # Call callback function if defined
            if self.opt['Callback'] is not None:
                if self.opt['Callback'](self):
                    break

            # Stop if residual-based stopping tolerances reached
            if not self.opt['FastSolve']:
                if frcxd < adapt_tol:
                    break

        # Increment iteration count
        self.k += 1

        # Record solve time
        self.timer.stop(['solve', 'solve_wo_func', 'solve_wo_rsdl',
                         'solve_wo_btrack'])

        # Print final separator string if Verbose option enabled
        self.display_end(nsep)

        return self.getmin()



    def getmin(self):
        """Get minimiser after optimisation."""

        return self.X



    def xstep(self, grad=None):
        """Compute proximal update (gradient descent + regularization).
           Optionally, a monotone PGM version from :cite:`beck-2009-tv`
           is available.
        """

        if grad is None:
            grad = self.grad_f()

        if self.stepsizepolicy is not None:
            if self.k > 1:
                self.L = self.stepsizepolicy.update(self, grad)
            if isinstance(self.stepsizepolicy, StepSizePolicyBB):
                # BB variants are two-point methods
                self.stepsizepolicy.store_prev_state(self.X, grad)

        V = self.Y - (1. / self.L) * grad

        self.X = self.prox_g(V)

        if self.opt['Monotone'] and self.k > 0:
            self.ZZ = self.X.copy()
            self.objfn = self.eval_objfn()
            if self.objfn_prev[0] < self.objfn[0]:
                # If increment on objective function
                # revert to previous iterate
                self.X = self.Xprv.copy()
                self.objfn = self.objfn_prev

        return grad



    def ystep(self):
        """Build next update by a smart combination of previous updates
        (standard PGM :cite:`beck-2009-fast`). Optionally, a monotone
        PGM version from :cite:`beck-2009-tv` is available.
        """

        # Update t step
        tprv = self.t
        self.t = self.momentum.update(self.var_momentum())

        # Update Y
        if self.opt['Monotone'] and self.k > 0:
            self.Y = self.X + (tprv / self.t) * (self.ZZ - self.X) \
                + ((tprv - 1.) / self.t) * (self.X - self.Xprv)
        else:
            self.Y = self.X + ((tprv - 1.) / self.t) * (self.X - self.Xprv)



    def eval_linear_approx(self, Dxy, gradY):
        r"""Compute term :math:`\langle \nabla f(\mathbf{y}), \mathbf{x}
        - \mathbf{y} \rangle` that is part of the quadratic function
        :math:`Q_L` used for backtracking.
        """

        return np.sum(Dxy * gradY)



    def grad_f(self, V):
        """Compute gradient of :math:`f` at V.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def prox_g(self, V):
        """Compute proximal operator of :math:`g`.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def hessian_f(self, V):
        """Compute Hessian of :math:`f` and apply to V.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def on_iteration_start(self):
        """Store previous X and Y states."""

        self.Xprv = self.X.copy()
        if (not self.opt['FastSolve'] or isinstance(self.backtrack,
                                                    BacktrackRobust)):
            self.Yprv = self.Y.copy()

        if self.opt['Monotone']:
            if self.k == 0:
                self.objfn = self.eval_objfn()
            self.objfn_prev = self.objfn


    def eval_Dxy(self):
        """Evaluate difference of state and auxiliary state updates."""

        return self.X - self.Y



    def compute_residuals(self):
        """Compute residuals and stopping thresholds."""

        r = self.rsdl()
        adapt_tol = self.opt['RelStopTol']

        if self.opt['AutoStop', 'Enabled']:
            adapt_tol = self.tau0 / (1. + self.k)

        return r, adapt_tol



    @classmethod
    def hdrtxt(cls):
        """Construct tuple of status display column title."""

        return ('Itn',) + cls.hdrtxt_objfn + ('Rsdl', 'F', 'Q', 'It_Bt', 'L')



    @classmethod
    def hdrval(cls):
        """Construct dictionary mapping display column title to
        IterationStats entries.
        """

        hdr = {'Itn': 'Iter'}
        hdr.update(cls.hdrval_objfun)
        hdr.update({'Rsdl': 'Rsdl', 'F': 'F_Btrack', 'Q': 'Q_Btrack',
                     'It_Bt': 'IterBTrack', 'L': 'L'})

        return hdr



    def iteration_stats(self, k, frcxd):
        """Construct iteration stats record tuple."""

        tk = self.timer.elapsed(self.opt['IterTimer'])
        if self.opt['Monotone']:
            tpl = (k,) + self.objfn \
                + (frcxd, self.F, self.Q, self.iterBTrack, self.L) \
                + self.itstat_extra() + (tk,)
        else:
            tpl = (k,) + self.eval_objfn() \
                + (frcxd, self.F, self.Q, self.iterBTrack, self.L) \
                + self.itstat_extra() + (tk,)
        return type(self).IterationStats(*tpl)



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        fval = self.obfn_f(self.X)
        gval = self.obfn_g(self.X)
        obj = fval + gval
        return (obj, fval, gval)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return ()



    def getitstat(self):
        """Get iteration stats as named tuple of arrays instead of
        array of named tuples.
        """

        return transpose_ntpl_list(self.itstat)



    def display_start(self):
        """Set up status display if option selected. NB: this method
        assumes that the first entry is the iteration count and the
        last is the L value.
        """

        if self.opt['Verbose']:
            # If backtracking option enabled F, Q, itBT, L are
            # included in iteration status
            if self.opt['Backtrack'] is not None:
                hdrtxt = type(self).hdrtxt()
            else:
                hdrtxt = type(self).hdrtxt()[0:-4]
            # Call utility function to construct status display formatting
            hdrstr, fmtstr, nsep = solve_status_str(
                hdrtxt, fmtmap={'It_Bt': '%5d'}, fwdth0=type(self).fwiter,
                fprec=type(self).fpothr)
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
            if self.opt['Backtrack'] is None:
                itdsp = itdsp[0:-4]

            print(fmtstr % itdsp)



    def display_end(self, nsep):
        """Terminate status display if option selected."""

        if self.opt['Verbose'] and self.opt['StatusHeader']:
            print("-" * nsep)



    def var_x(self):
        r"""Get :math:`\mathbf{x}` variable."""

        return self.X



    def var_y(self, y=None):
        r"""Get, or update and get, :math:`\mathbf{y}` variable."""

        if y is not None:
            self.Y = y

        return self.Y



    def var_xprv(self):
        r"""Get :math:`\mathbf{x}` variable of previous iteration."""

        return self.Xprv



    def var_momentum(self):
        """Most momentum coefficient methods require iteration but Nesterov
           requires current t."""

        if isinstance(self.momentum, MomentumNesterov):
            return self.t
        return self.k



    def obfn_f(self, X):
        r"""Compute :math:`f(\mathbf{x})` component of PGM objective
        function.

        Overriding this method is required (even if :meth:`eval_objfun`
        is overriden, since this method is required for backtracking).
        """

        raise NotImplementedError()



    def obfn_g(self, X):
        r"""Compute :math:`g(\mathbf{x})` component of PGM objective
        function.

        Overriding this method is required if :meth:`eval_objfun`
        is not overridden.
        """

        raise NotImplementedError()



    def rsdl(self):
        """Compute fixed point residual (see Sec. 4.3 of
        :cite:`liu-2018-first`)."""

        if self.opt['Monotone'] and self.k > 0:
            return np.linalg.norm((self.X - self.Y).ravel())
        return np.linalg.norm((self.X - self.Yprv).ravel())





class PGMDFT(PGM):
    r"""
    Base class for PGM algorithms with gradients and updates computed
    in the frequency domain.

    |

    .. inheritance-diagram:: PGMDFT
       :parts: 2

    |

    Solve optimisation problems of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x}} \; f(\mathbf{x}) + g(\mathbf{x})
       \;\;,

    where :math:`f, g` are convex functions and :math:`f` is smooth.

    This class specialises class PGM, but remains a base class for
    other classes that specialise to specific optimisation problems.
    """


    class Options(PGM.Options):
        """PGMDFT algorithm options.

        Options include all of those defined in :class:`PGM.Options`.
        """

        defaults = copy.deepcopy(PGM.Options.defaults)

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              PGMDFT algorithm options
            """

            if opt is None:
                opt = {}
            PGM.Options.__init__(self, opt)



    def __init__(self, xshape, Nv, axisN, dtype, opt=None):
        """
        Parameters
        ----------
        xshape : tuple of ints
          Shape of working variable X (the primary variable)
        Nv : tuple of ints
          Shape of spatial indices of variable X (needed for DFT)
        axisN : tuple of ints
          Axis indices of spatial components of X (needed for DFT)
        dtype : data-type
          Data type for working variables
        opt : :class:`PGMDFT.Options` object
          Algorithm options
        """

        if opt is None:
            opt = PGMDFT.Options()
        super(PGMDFT, self).__init__(xshape, dtype, opt)
        self.Nv = Nv
        self.axisN = axisN



    def xstep(self, gradf=None):
        """Compute proximal update (gradient descent + constraint).
        Variables are mapped back and forth between input and
        frequency domains. Optionally, a monotone PGM version from
        :cite:`beck-2009-tv` is available.
        """

        if gradf is None:
            gradf = self.grad_f()

        if self.stepsizepolicy is not None:
            if self.k > 1:
                self.L = self.stepsizepolicy.update(self, gradf)
            if isinstance(self.stepsizepolicy, StepSizePolicyBB):
                # BB variants are two-point methods
                self.stepsizepolicy.store_prev_state(self.Xf, gradf)

        self.Vf[:] = self.Yf - (1. / self.L) * gradf
        V = irfftn(self.Vf, self.Nv, self.axisN)

        self.X[:] = self.prox_g(V)
        self.Xf = rfftn(self.X, None, self.axisN)

        if self.opt['Monotone'] and self.k > 0:
            self.ZZf = self.Xf.copy()
            self.objfn = self.eval_objfn()
            if self.objfn_prev[0] < self.objfn[0]:
                # If increment on objective function
                # revert to previous iterate
                self.Xf = self.Xfprv.copy()
                self.objfn = self.objfn_prev

        return gradf



    def ystep(self):
        """Update auxiliary state by a smart combination of previous
        updates in the frequency domain (standard PGM
        :cite:`beck-2009-fast`). Optionally, a monotone PGM version
        from :cite:`beck-2009-tv` is available.
        """

        # Update t step
        tprv = self.t
        self.t = self.momentum.update(self.var_momentum())

        # Update Y
        if self.opt['Monotone'] and self.k > 0:
            self.Yf = self.Xf + (tprv / self.t) * (self.ZZf - self.Xf) \
                + ((tprv - 1.) / self.t) * (self.Xf - self.Xfprv)
        else:
            self.Yf = self.Xf + ((tprv - 1.) / self.t) * (self.Xf - self.Xfprv)



    def on_iteration_start(self):
        """Store previous X and Y in frequency domain."""

        self.Xfprv = self.Xf.copy()
        if (not self.opt['FastSolve'] or isinstance(self.backtrack,
                                                    BacktrackRobust)):
            self.Yfprv = self.Yf.copy()

        if self.opt['Monotone']:
            if self.k == 0:
                self.objfn = self.eval_objfn()
            self.objfn_prev = self.objfn



    def eval_Dxy(self):
        """Evaluate difference of state and auxiliary state in
        frequency domain.
        """

        return self.Xf - self.Yf



    def var_x(self):
        r"""Get :math:`\mathbf{x}` variable in frequency domain."""

        return self.Xf



    def var_y(self, y=None):
        r"""Get, or update and get, :math:`\mathbf{y}` variable in
        frequency domain."""

        if y is not None:
            self.Yf = y

        return self.Yf



    def var_xprv(self):
        r"""Get :math:`\mathbf{x}` variable of previous iteration in
        frequency domain.
        """

        return self.Xfprv



    def eval_linear_approx(self, Dxy, gradY):
        r"""Compute term :math:`\langle \nabla f(\mathbf{y}),
        \mathbf{x} - \mathbf{y} \rangle` (in frequency domain) that is
        part of the quadratic function :math:`Q_L` used for
        backtracking. Since this class computes the backtracking in
        the DFT, it is important to preserve the DFT scaling.
        """

        return np.sum(np.real(np.conj(Dxy) * gradY))
