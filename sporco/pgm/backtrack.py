# -*- coding: utf-8 -*-
# Copyright (C) 2020 by Cristina Garcia-Cardona <cgarciac@lanl.gov>
#                       Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Backtracking methods for PGM algorithms"""

from __future__ import division, print_function

import numpy as np


__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""



class BacktrackBase(object):
    """Base class for computing step size for
    proximal gradient method via backtracking.

    This class is intended to be a base class of other classes
    that specialise to specific backtracking options.

    After termination of the :meth:`update` method the new
    state in the proximal gradient method is computed.
    This also updates all the supporting variables.
    """

    def __init__(self):

        super(BacktrackBase, self).__init__()



    def update(self, solverobj):
        """Update step size via backtracking.

        Overriding this method is required.
        """

        raise NotImplementedError()





class BacktrackStandard(BacktrackBase):
    """Class to estimate step size L by computing a linesearch
    that guarantees that F <= Q according to the standard PGM
    backtracking strategy in :cite:`beck-2009-fast`.

    After termination of the :meth:`update` method the new
    state in the proximal gradient method is computed.
    This also updates all the supporting variables.
    """

    def __init__(self, gamma_u=1.2, maxiter=50):
        r"""
        Parameters
        ----------
        gamma_u : float
          Multiplier applied to increase L when backtracking
          in standard PGM (corresponding to :math:`\eta` in
          :cite:`beck-2009-fast`).

        maxiter : int
          Maximum iterations of updating L when backtracking.
        """

        super(BacktrackStandard, self).__init__()
        # Initialise attributes controlling the backtracking
        self.gamma_u = gamma_u
        self.maxiter = maxiter



    def update(self, solverobj):
        """
        Parameters
        ----------
        solverobj : PGM object
            object containing state and functions
            required to adjust the step size
        """

        gradY = solverobj.grad_f()  # Given Y(f), this update computes gradY(f)

        maxiter = self.maxiter

        iterBTrack = 0
        linesearch = 1
        while linesearch and iterBTrack < maxiter:

            solverobj.xstep(gradY)  # Given gradY(f), L, this updates X(f)

            f = solverobj.obfn_f(solverobj.var_x())
            Dxy = solverobj.eval_Dxy()
            Q = solverobj.obfn_f(solverobj.var_y()) + \
                solverobj.eval_linear_approx(Dxy, gradY) + \
                (solverobj.L / 2.) * np.linalg.norm(Dxy.flatten(), 2)**2

            if f <= Q:
                linesearch = 0
            else:
                solverobj.L *= self.gamma_u

            iterBTrack += 1

        solverobj.F = f
        solverobj.Q = Q
        solverobj.iterBTrack = iterBTrack
        # Update auxiliary sequence
        solverobj.ystep()





class BacktrackRobust(BacktrackBase):
    """Class to estimate step size L by computing a linesearch
    that guarantees that F <= Q according to the robust PGM
    backtracking strategy in :cite:`florea-2017-robust`.

    After termination of the :meth:`update` method the new
    state in the proximal gradient method is computed.
    This also updates all the supporting variables.
    """

    def __init__(self, gamma_d=0.9, gamma_u=2.0, maxiter=50):
        r"""
        Parameters
        ----------
        gamma_d : float
          Multiplier applied to decrease L when
          backtracking in robust PGM (:math:`\gamma_d` in
          :cite:`florea-2017-robust`).

        gamma_u : float
          Multiplier applied to increase L when
          backtracking in robust PGM (corresponding
          Total :math:`\gamma_u` in :cite:`florea-2017-robust`).

        maxiter : int
          Maximum iterations of updating L when backtracking.
        """

        super(BacktrackRobust, self).__init__()

        # Initialise attributes controlling the backtracking
        self.gamma_d = gamma_d
        self.gamma_u = gamma_u
        self.maxiter = maxiter

        self.Tk = 0.
        self.Zrb = None



    def update(self, solverobj):
        """
        Parameters
        ----------
        solverobj : PGM object
          object containing state and functions
          required to adjust the step size
        """

        if self.Zrb is None:
            self.Zrb = solverobj.var_x().copy()

        solverobj.L *= self.gamma_d
        maxiter = self.maxiter

        iterBTrack = 0
        linesearch = 1

        while linesearch and iterBTrack < maxiter:

            t = float(1. + np.sqrt(1. + 4. * solverobj.L * self.Tk)) / \
                (2. * solverobj.L)
            T = self.Tk + t
            y = (self.Tk * solverobj.var_xprv() + t * self.Zrb) / T
            solverobj.var_y(y)

            gradY = solverobj.xstep()  # Given Y(f), L, this updates X(f)

            f = solverobj.obfn_f(solverobj.var_x())
            Dxy = solverobj.eval_Dxy()
            Q = solverobj.obfn_f(solverobj.var_y()) + \
                solverobj.eval_linear_approx(Dxy, gradY) + \
                (solverobj.L / 2.) * np.linalg.norm(Dxy.flatten(), 2)**2

            if f <= Q:
                linesearch = 0
            else:
                solverobj.L *= self.gamma_u

            iterBTrack += 1

        self.Tk = T
        self.Zrb += (t * solverobj.L * (solverobj.var_x() - solverobj.var_y()))

        solverobj.F = f
        solverobj.Q = Q
        solverobj.iterBTrack = iterBTrack
