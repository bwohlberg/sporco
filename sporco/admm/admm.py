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

            # Update record of Y from previous iteration
            self.Yprev = self.Y.copy()

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

        # Print final seperator string if Verbose option enabled
        self.display_end(nsep)

        return self.X




    def xstep(self):
        """Minimise Augmented Lagrangian with respect to x.

        Overriding this method is required."""

        raise NotImplementedError()



    def ystep(self):
        """Minimise Augmented Lagrangian with respect to y.

        Overriding this method is required."""

        raise NotImplementedError()



    def ustep(self):
        """Dual variable update."""

        self.U += self.rsdl_r(self.AX, self.Y)




    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        if self.opt['RelaxParam'] == 1.0:
            self.AX = self.cnst_A(self.X)
        else:
            if not hasattr(self, 'c'):
                self.c = self.cnst_c()
            alpha = self.opt['RelaxParam']
            self.AX = alpha*self.cnst_A(self.X) - \
                (1-alpha)*(self.cnst_B(self.Y) - self.c)




    def compute_residuals(self):
        """Compute residuals and stopping thresholds."""

        if self.opt['AutoRho', 'StdResiduals']:
            r = linalg.norm(self.rsdl_r(self.AX, self.Y))
            s = linalg.norm(self.rsdl_s(self.Yprev, self.Y))
            epri = scipy.sqrt(self.Nc)*self.opt['AbsStopTol'] + \
                self.rsdl_rn(self.AX, self.Y)*self.opt['RelStopTol']
            edua = scipy.sqrt(self.Nx)*self.opt['AbsStopTol'] + \
                self.rsdl_sn(self.U)*self.opt['RelStopTol']
        else:
            rn = self.rsdl_rn(self.AX, self.Y)
            sn = self.rsdl_sn(self.U)
            r = linalg.norm(self.rsdl_r(self.AX, self.Y)) / rn
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
            # Print header and seperator strings
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

        Overriding this method is required."""

        raise NotImplementedError()



    def obfn_g(self, Y):
        """Compute :math:`g(\mathbf{y})` component of ADMM objective function.

        Overriding this method is required."""

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

        Overriding this method is optional."""

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

        if self.opt['RelaxParam'] == 1.0:
            self.AX = self.X
        else:
            alpha = self.opt['RelaxParam']
            self.AX = alpha*self.X + (1-alpha)*self.Y



    def obfn_fvar(self):
        """Variable to be evaluated in computing :meth:`ADMM.obfn_f`, depending
        on the ``fEvalX`` option value."""

        return self.X if self.opt['fEvalX'] else self.Y



    def obfn_gvar(self):
        """Variable to be evaluated in computing :meth:`ADMM.obfn_g`, depending
        on the ``gEvalY`` option value."""

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
        is :math:`\mathbf{x} = \mathbf{y}`"""

        return X


    def cnst_AT(self, X):
        """Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{x} = \mathbf{x}` since the constraint
        is :math:`\mathbf{x} = \mathbf{y}`"""

        return X



    def cnst_B(self, Y):
        """Compute :math:`B \mathbf{y}` component of ADMM problem constraint.
        In this case :math:`B \mathbf{y} = -\mathbf{y}` since the constraint
        is :math:`\mathbf{x} = \mathbf{y}`"""

        return -Y



    def cnst_c(self):
        """Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = \mathbf{0}` since
        the constraint is :math:`\mathbf{x} = \mathbf{y}`"""

        return np.zeros(self.X.shape, self.X.dtype)



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
