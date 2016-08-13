#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Base classes for ADMM algorithms"""

from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object

import numpy as np
from scipy import linalg
import scipy
import copy
import collections

from sporco import cdict
from sporco import util

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class ADMM(object):
    """Base class for Alternating Direction Method of Multipliers (ADMM)
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

       ``EpsPrimal`` : Primal residual stopping tolerance \
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance \
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """


    class Options(cdict.ConstrainedDict):
        """ADMM algorithm options.

        Options:

          ``Verbose`` : Flag determining whether iteration status is displayed.

          ``StatusHeader`` : Flag determining whether status header and \
              separator are dislayed

          ``DataType`` : Specify data type for solution variables, \
              e.g. ``np.float32``

          ``Y0`` : Initial value for Y variable

          ``U0`` : Initial value for U variable

          ``Callback`` : Callback function to be called at the end of \
               every iteration

          ``MaxMainIter`` : Maximum main iterations

          ``AbsStopTol`` : Absolute convergence tolerance (see \
              Sec. 3.3.1 of :cite:`boyd-2010-distributed`)

          ``RelStopTol`` : Relative convergence tolerance (see \
              Sec. 3.3.1 of :cite:`boyd-2010-distributed`)

          ``RelaxParam`` : Relaxation parameter (see \
              Sec. 3.4.3 of :cite:`boyd-2010-distributed`)

          ``rho`` : ADMM penalty parameter

          ``AutoRho`` : Options for adaptive rho strategy (see \
              :cite:`wohlberg-2015-adaptive` and Sec. 3.4.3 of \
              :cite:`boyd-2010-distributed`)

            ``Enabled`` : Flag determining whether adaptive rho \
              strategy is enabled

            ``Period`` : Iteration period on which rho is updated

            ``Scaling`` : Multiplier applied to rho when updated

            ``RsdlRatio`` : Primal/dual residual ratio in rho update test

            ``RsdlTarget`` : Residual ratio targeted by auto rho update policy

            ``AutoScaling`` : Flag determining whether RhoScaling value is \
                adaptively determined. If  enabled, Scaling specifies a \
                maximum allowed multiplier instead of a fixed multiplier.

            ``StdResiduals`` : Flag determining whether standard residual \
                definitions are used instead of normalised residuals
        """

        defaults = {'Verbose' : False, 'StatusHeader' : True,
                    'DataType' : None, 'MaxMainIter' : 1000,
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



        def set_dtype(self, dtype, override=False):
            """Set DataType value"""

            if override or self['DataType'] is None:
                self['DataType'] = dtype


    IterationStats = collections.namedtuple('IterationStats',
                ['Iter', 'ObjFun', 'FVal', 'GVal', 'PrimalRsdl', 'DualRsdl',
                 'EpsPrimal', 'EpsDual', 'Rho', 'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""
    hdrtxt = ['Itn', 'Fnc', 'f', 'g', 'r', 's', 'rho']
    """Display column header text. NB: The display_start function assumes
    that the first entry is the iteration count and the last is the
    rho value"""
    hdrval = {'Itn' : 'Iter', 'Fnc' : 'ObjFun', 'f' : 'FVal', 'g' : 'GVal',
              'r' : 'PrimalRsdl', 's' : 'DualRsdl', 'rho' : 'Rho'}
    """Dictionary mapping display column headers to IterationStats entries"""



    def __init__(self, Nx, Nc, opt=None):
        """
        Initialise an ADMM object with problem size and options.

        Parameters
        ----------
        Nx : int
          Size of variable :math:`\mathbf{x}` in objective function
        Nc : int
          Size of constant :math:`\mathbf{c}` in constraint
        opt : :class:`ADMM.Options` object
          Algorithm options
        """

        if opt is None:
            opt = ADMM.Options()
        if not isinstance(opt, ADMM.Options):
            raise TypeError("Parameter opt must be an instance of ADMM.Options")
        self.runtime = 0.0
        self.timer = util.Timer()
        self.Nx = Nx
        self.Nc = Nc
        self.itstat = []
        self.k = 0

        self.Y = None
        self.U = None

        if opt['DataType'] is not None:
            if opt['Y0'] is not None:
                opt['Y0'] = opt['Y0'].astype(opt['DataType'])
            if opt['U0'] is not None:
                opt['U0'] = opt['U0'].astype(opt['DataType'])
        self.opt = opt

        self.runtime += self.timer.elapsed()



    def solve(self):
        """Start (or re-start) optimisation. This method implements the
        framework for the iterations of an ADMM algorithm. There is sufficient
        flexibility in overriding the component methods that it calls that it
        is usually not necessary to override this method in derived clases.

        If option ``Verbose`` is ``True``, the progress of the
        optimisation is displayed at every iteration. At termination
        of this method, attribute :attr:`itstat` is a list of tuples
        representing statistics of each iteration.
        """

        # Open status display
        fmtstr, nsep = self.display_start()

        # Reset timer
        self.timer.start()

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
            r, s, epri, edua = self.compute_residuals()

            # Compute functional value
            tk = self.timer.elapsed()
            itst = self.iteration_stats(k, r, s, epri, edua, tk)
            self.itstat.append(itst)

            # Display iteration stats if Verbose option enabled
            self.display_status(fmtstr, k, itst)

            # Automatic rho adjustment
            self.update_rho(k, r, s)

            # Call callback function if defined
            if self.opt['Callback'] is not None:
                self.opt['Callback'](self, k)

            # Stop if residual-based stopping tolerances reached
            if r < epri and s < edua:
                break


        # Record run time
        self.runtime += self.timer.elapsed()

        # Record iteration count
        self.k = k+1

        # Print final separator string if Verbose option enabled
        self.display_end(nsep)

        return self.X



    def xstep(self):
        """Minimise Augmented Lagrangian with respect to x.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def ystep(self):
        """Minimise Augmented Lagrangian with respect to y.

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
        if self.opt['RelaxParam'] == 1.0:
            self.AX = self.AXnr
        else:
            if not hasattr(self, 'c'):
                self.c = self.cnst_c()
            alpha = self.opt['RelaxParam']
            self.AX = alpha*self.AXnr - (1-alpha)*(self.cnst_B(self.Y) - self.c)



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



    def iteration_stats(self, k, r, s, epri, edua, tk):
        """Construct iteration stats record tuple."""

        fval = self.obfn_f(self.X)
        gval = self.obfn_g(self.Y)
        obj = gval + gval
        itst = type(self).IterationStats(k, obj, fval, gval, r, s,
                                         epri, edua, self.rho, tk)
        return itst



    def getitstat(self):
        """Get iteration stats as named tuple of arrays instead of array of
        named tuples.
        """

        if len(self.itstat) == 0:
            return None
        else:
            return type(self).IterationStats(
                *[[self.itstat[k][l] for k in range(len(self.itstat))]
                  for l in range(len(self.itstat[0]))]
            )



    def update_rho(self, k, r, s):

        """Automatic rho adjustment."""

        if self.opt['AutoRho','Enabled']:
            tau = self.opt['AutoRho','Scaling']
            mu = self.opt['AutoRho','RsdlRatio']
            xi = self.opt['AutoRho','RsdlTarget']
            if k != 0 and scipy.mod(k+1,self.opt['AutoRho','Period']) == 0:
                if self.opt['AutoRho','AutoScaling']:
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
                self.rho = rsf*self.rho
                self.U = self.U/rsf
                if rsf != 1.0:
                    self.rhochange()



    def display_start(self):
        """Set up status display if option selected."""

        if self.opt['Verbose']:
            # If AutoRho option enabled rho is included in iteration status
            if self.opt['AutoRho','Enabled']:
                hdrtxt = type(self).hdrtxt
            else:
                hdrtxt = type(self).hdrtxt[0:-1]
            # Call utility function to construct status display formatting
            hdrstr, fmtstr, nsep = util.solve_status_str(hdrtxt,
                                type(self).fwiter, type(self).fpothr)
            # Print header and separator strings
            if self.opt['StatusHeader']:
                print(hdrstr)
                print("-" * nsep)
        else:
            fmtstr, nsep = '', 0

        return fmtstr, nsep



    def display_status(self, fmtstr, k, itst):
        """Display current iteration status as selection of fields from
        iteration stats tuple.
        """

        if self.opt['Verbose']:
            itdsp = tuple([getattr(itst, type(self).hdrval[col]) for col in
                           type(self).hdrtxt])
            if not self.opt['AutoRho','Enabled']:
                itdsp = itdsp[0:-1]

            print(fmtstr % itdsp)



    def display_end(self, nsep):
        """Terminate status display if option selected."""

        if self.opt['Verbose'] and self.opt['StatusHeader']:
            print("-" * nsep)



    def var_x(self):
        """Get :math:`\mathbf{x}` variable."""

        return self.X



    def var_y(self):
        """Get :math:`\mathbf{y}` variable."""

        return self.Y



    def obfn_f(self, X):
        """Compute :math:`f(\mathbf{x})` component of ADMM objective function.

        Overriding this method is required if :meth:`iteration_stats`
        is not overridden.
        """

        raise NotImplementedError()



    def obfn_g(self, Y):
        """Compute :math:`g(\mathbf{y})` component of ADMM objective function.

        Overriding this method is required if :meth:`iteration_stats`
        is not overridden.
        """

        raise NotImplementedError()



    def cnst_A(self, X):
        """Compute :math:`A \mathbf{x}` component of ADMM problem constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """

        raise NotImplementedError()



    def cnst_AT(self, X):
        """Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """

        raise NotImplementedError()



    def cnst_B(self, Y):
        """Compute :math:`B \mathbf{y}` component of ADMM problem constraint.

        Overriding this method is required if methods :meth:`rsdl_r`,
        :meth:`rsdl_s`, :meth:`rsdl_rn`, and :meth:`rsdl_sn` are not
        overridden.
        """

        raise NotImplementedError()



    def cnst_c(self):
        """Compute constant component :math:`\mathbf{c}` of ADMM problem
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

        if not hasattr(self, 'c'):
            self.c = self.cnst_c()
        return AX + self.cnst_B(Y) - self.c



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

        if not hasattr(self, 'nc'):
            self.nc = linalg.norm(self.cnst_c())
        return max((linalg.norm(AX), linalg.norm(self.cnst_B(Y)), self.nc))



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
    """Base class for ADMM algorithms with a simple equality constraint.

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

          ``fEvalX`` : Flag indicating whether the :math:`f` component of the \
              objective function should be evaluated using variable X \
              (``True``) or Y (``False``) as its argument

          ``gEvalY`` : Flag indicating whether the :math:`g` component of the \
              objective function should be evaluated using variable Y \
              (``True``) or X (``False``) as its argument

          ``ReturnX`` : Flag indicating whether the return value of the \
              solve method is the X variable (``True``) or the Y variable \
              (``False``)
        """

        defaults = copy.deepcopy(ADMM.Options.defaults)
        defaults.update({'fEvalX' : True, 'gEvalY' : True, 'ReturnX' : True})

        def __init__(self, opt=None):
            """Initialise ADMM algorithm options object."""

            if opt is None:
                opt = {}
            ADMM.Options.__init__(self, opt)




    def __init__(self, Nx, opt=None):
        """
        Initialise an ADMMEqual object with problem size and options.

        Parameters
        ----------
        Nx : int
          Size of variable :math:`\mathbf{x}` in objective function
        opt : :class:`ADMMEqual.Options` object
          Algorithm options
        """

        if opt is None:
            opt = ADMMEqual.Options()
        super(ADMMEqual, self).__init__(Nx, Nx, opt)



    def solve(self):
        """Run optimisation."""

        super(ADMMEqual, self).solve()

        return self.X if self.opt['ReturnX'] else self.Y



    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        self.AXnr = self.X
        if self.opt['RelaxParam'] == 1.0:
            self.AX = self.X
        else:
            alpha = self.opt['RelaxParam']
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



    def iteration_stats(self, k, r, s, epri, edua, tk):
        """Construct iteration stats record tuple."""

        fval = self.obfn_f(self.obfn_fvar())
        gval = self.obfn_g(self.obfn_gvar())
        obj = fval + gval
        itst = type(self).IterationStats(k, obj, fval, gval, r, s,
                                         epri, edua, self.rho, tk)
        return itst



    def cnst_A(self, X):
        """Compute :math:`A \mathbf{x}` component of ADMM problem constraint.
        In this case :math:`A \mathbf{x} = \mathbf{x}` since the constraint
        is :math:`\mathbf{x} = \mathbf{y}`.
        """

        return X


    def cnst_AT(self, X):
        """Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{x} = \mathbf{x}` since the constraint
        is :math:`\mathbf{x} = \mathbf{y}`.
        """

        return X



    def cnst_B(self, Y):
        """Compute :math:`B \mathbf{y}` component of ADMM problem constraint.
        In this case :math:`B \mathbf{y} = -\mathbf{y}` since the constraint
        is :math:`\mathbf{x} = \mathbf{y}`.
        """

        return -Y



    def cnst_c(self):
        """Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = \mathbf{0}` since
        the constraint is :math:`\mathbf{x} = \mathbf{y}`.
        """

        return 0.0



    def rsdl_r(self, AX, Y):
        """Compute primal residual vector."""

        return AX - Y



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        return self.rho*(Y - Yprev)



    def rsdl_rn(self, AX, Y):
        """Compute primal residual normalisation term."""

        return max((linalg.norm(AX), linalg.norm(Y)))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho*linalg.norm(U)




class ADMMTwoBlockCnstrnt(ADMM):
    """Base class for ADMM algorithms for problems for which
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
       \;\\text{such that}\;
       \\left( \\begin{array}{c} A_0 \\\\ A_1 \\end{array} \\right) \mathbf{x}
       - \\left( \\begin{array}{c} \mathbf{y}_0 \\\\ \mathbf{y}_1 \\end{array}
       \\right) = \\left( \\begin{array}{c} \mathbf{c}_0 \\\\
       \mathbf{c}_1 \\end{array} \\right) \;\;.

    This class specialises class :class:`.ADMM`, but remains a base class for
    other classes that specialise to specific optimisation problems.
    """


    class Options(ADMM.Options):
        """ADMMTwoBlockCnstrnt algorithm options.

        Options include all of those defined in :class:`ADMM.Options`,
        together with additional options:

        ``AuxVarObj`` : Flag indicating whether the :math:`g(\mathbf{y})`
        component of the objective function should be evaluated using
        variable X  (``False``) or Y (``True``) as its argument.

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



    IterationStats = collections.namedtuple('IterationStats',
                ['Iter', 'ObjFun', 'FVal', 'G0Val', 'G1Val',
                 'PrimalRsdl', 'DualRsdl', 'EpsPrimal', 'EpsDual',
                 'Rho', 'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    hdrtxt = ['Itn', 'Fnc', 'f', 'g0', 'g1', 'r', 's', 'rho']
    """Display column header text. NB: The display_start function assumes
    that the first entry is the iteration count and the last is the
    rho value"""
    hdrval = {'Itn' : 'Iter', 'Fnc' : 'ObjFun', 'f' : 'FVal',
              'g0' : 'G0Val', 'g1' : 'G1Val', 'r' : 'PrimalRsdl',
              's' : 'DualRsdl', 'rho' : 'Rho'}
    """Dictionary mapping display column headers to IterationStats entries"""



    def __init__(self, Nx, Nc, blkaxis, blkidx, opt=None):
        """
        Initialise an ADMMTwoBlockCnstrnt object with problem size and options.

        Parameters
        ----------
        Nx : int
          Size of variable :math:`\mathbf{x}` in objective function
        Nc : int
          Size of constant :math:`\mathbf{c}` in constraint
        blkaxis : int
          Axis on which :math:`\mathbf{y}_0` and :math:`\mathbf{y}_1` are
          concatenated to form :math:`\mathbf{y}`
        blkidx : int
          Index of boundary between :math:`\mathbf{y}_0` and
          :math:`\mathbf{y}_1` on axis on which they are concatenated to
          form :math:`\mathbf{y}`
        opt : :class:`ADMMTwoBlockCnstrnt.Options` object
          Algorithm options
        """

        if opt is None:
            opt = ADMM.Options()
        super(ADMMTwoBlockCnstrnt, self).__init__(Nx, Nc, opt)
        self.blkaxis = blkaxis
        self.blkidx = blkidx



    def solve(self):
        """Run optimisation."""

        super(ADMMTwoBlockCnstrnt, self).solve()

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
        """Separate variable into component corresponding to Y0 in Y."""

        return Y[(slice(None),)*self.blkaxis + (slice(0,self.blkidx),)]



    def block_sep1(self, Y):
        """Separate variable into component corresponding to Y1 in Y."""

        return Y[(slice(None),)*self.blkaxis + (slice(self.blkidx,None),)]



    def block_sep(self, Y):
        """Separate variable into components corresponding to blocks in Y."""

        return (self.block_sep0(Y), self.block_sep1(Y))



    def block_cat(self, Y0, Y1):
        """Concatenate components corresponding to Y0 and Y1 blocks into Y."""

        return np.concatenate((Y0, Y1), axis=self.blkaxis)



    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        self.AXnr = self.cnst_A(self.X)
        if self.opt['RelaxParam'] == 1.0:
            self.AX = self.AXnr
        else:
            if not hasattr(self, 'c0'):
                self.c0 = self.cnst_c0()
            if not hasattr(self, 'c1'):
                self.c1 = self.cnst_c1()
            alpha = self.opt['RelaxParam']
            self.AX = alpha*self.cnst_A(self.X) + \
                      (1-alpha)*self.block_cat(self.var_y0() + self.c0,
                                               self.var_y1() + self.c1)



    def var_y0(self):
        """Get :math:`\mathbf{y}_0` variable."""

        return self.block_sep0(self.Y)



    def var_y1(self):
        """Get :math:`\mathbf{y}_1` variable."""

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
        """Compute :math:`f(\mathbf{x})` component of ADMM objective
        function. Unless overridden, :math:`f(\mathbf{x}) = 0`.
        """

        return 0.0



    def obfn_g(self, Y):
        """Compute :math:`g(\mathbf{y})` component of ADMM objective
        function.
        """

        return self.obfn_g0(self.obfn_g0var()) + self.obfn_g1(self.obfn_g1var())



    def obfn_g0(self, Y0):
        """Compute :math:`g_0(\mathbf{y}_0)` component of ADMM objective
        function.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def obfn_g1(self, Y1):
        """Compute :math:`g_1(\mathbf{y_1})` component of ADMM objective
        function.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def iteration_stats(self, k, r, s, epri, edua, tk):
        """Construct iteration stats record tuple."""

        fval = self.obfn_f(self.obfn_fvar())
        g0val = self.obfn_g0(self.obfn_g0var())
        g1val = self.obfn_g1(self.obfn_g1var())
        obj = fval + g0val + g1val
        itst = type(self).IterationStats(k, obj, fval, g0val, g1val, r, s,
                                         epri, edua, self.rho, tk)
        return itst



    def cnst_A(self, X):
        """Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint.
        """

        return self.block_cat(self.cnst_A0(X), self.cnst_A1(X))



    def cnst_AT(self, Y):
        """Compute :math:`A^T \mathbf{y}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint.
        """

        return self.cnst_A0T(self.block_sep0(Y)) + \
            self.cnst_A1T(self.block_sep1(Y))



    def cnst_B(self, Y):
        """Compute :math:`B \mathbf{y}` component of ADMM problem constraint.
        In this case :math:`B \mathbf{y} = -\mathbf{y}` since the constraint
        is :math:`A \mathbf{x} - \mathbf{y} = \mathbf{c}`.
        """

        return -Y



    def cnst_c(self):
        """Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. This method should not be used or overridden: all
        calculations should make use of components :meth:`cnst_c0` and
        :meth:`cnst_c1` so that these methods can return scalar zeros
        instead of zero arrays if appropriate.
        """

        raise NotImplementedError()



    def cnst_c0(self):
        """Compute constant component :math:`\mathbf{c}_0` of ADMM problem
        constraint. Unless overridden, :math:`\mathbf{c}_0 = 0`.
        """

        return 0.0



    def cnst_c1(self):
        """Compute constant component :math:`\mathbf{c}_1` of ADMM problem
        constraint. Unless overridden, :math:`\mathbf{c}_1 = 0`.
        """

        return 0.0



    def cnst_A0(self, X):
        """Compute :math:`A_0 \mathbf{x}` component of ADMM problem
        constraint. Unless overridden, :math:`A_0 \mathbf{x} = \mathbf{x}`,
        i.e. :math:`A_0 = I`.
        """

        return X



    def cnst_A0T(self, X):
        """Compute :math:`A_0^T \mathbf{x}` where :math:`A_0 \mathbf{x}` is a
        component of ADMM problem constraint. Unless overridden,
        :math:`A_0 \mathbf{x} = \mathbf{x}`, i.e. :math:`A_0 = I`.
        """

        return X



    def cnst_A1(self, X):
        """Compute :math:`A_1 \mathbf{x}` component of ADMM problem
        constraint. Unless overridden, :math:`A_1 \mathbf{x} = \mathbf{x}`,
        i.e. :math:`A_1 = I`.
        """

        return X



    def cnst_A1T(self, X):
        """Compute :math:`A_1^T \mathbf{x}` where :math:`A_1 \mathbf{x}` is a
        component of ADMM problem constraint. Unless overridden,
        :math:`A_1 \mathbf{x} = \mathbf{x}`, i.e. :math:`A_1 = I`.
        """

        return X



    def rsdl_r(self, AX, Y):
        """Compute primal residual vector.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_c0` and :meth:`cnst_c1` are not
        overridden.
        """

        if not hasattr(self, 'c0'):
            self.c0 = self.cnst_c0()
        if not hasattr(self, 'c1'):
            self.c1 = self.cnst_c1()
        return AX - self.block_cat(self.var_y0() + self.c0,
                                   self.var_y1() + self.c1)



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

        if not hasattr(self, 'nc'):
            self.nc = np.sqrt(linalg.norm(self.cnst_c0())**2 +
                              linalg.norm(self.cnst_c1())**2)
        return max((linalg.norm(AX), linalg.norm(Y), self.nc))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        return self.rho*linalg.norm(self.cnst_AT(U))
