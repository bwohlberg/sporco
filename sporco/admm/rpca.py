# -*- coding: utf-8 -*-
# Copyright (C) 2015-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithms for Robust PCA optimisation"""

from __future__ import division, absolute_import

import copy
import numpy as np

from sporco.admm import admm
import sporco.prox as sp
from sporco.util import u


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class RobustPCA(admm.ADMM):
    r"""ADMM algorithm for Robust PCA problem :cite:`candes-2011-robust`
    :cite:`cai-2010-singular`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{X, Y} \;
        \| X \|_* + \lambda \| Y \|_1 \quad \text{such that}
        \quad X + Y = S \;\;.

    This problem is unusual in that it is already in ADMM form without
    the need for any variable splitting.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``NrmNuc`` :  Value of nuclear norm term :math:`\| X \|_*`

       ``NrmL1`` : Value of :math:`\ell_1` norm term :math:`\| Y \|_1`

       ``Cnstr`` : Constraint violation :math:`\| X + Y - S\|_2`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """



    class Options(admm.ADMM.Options):
        """RobustPCA algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMM.Options`, together with
        an additional option:

          ``fEvalX`` : Flag indicating whether the :math:`f` component
          of the objective function should be evaluated using variable
          X (``True``) or Y (``False``) as its argument.

          ``gEvalY`` : Flag indicating whether the :math:`g` component
          of the objective function should be evaluated using variable
          Y (``True``) or X (``False``) as its argument.
        """

        defaults = copy.deepcopy(admm.ADMM.Options.defaults)
        defaults.update({'gEvalY': True, 'fEvalX': True, 'RelaxParam': 1.8})
        defaults['AutoRho'].update({'Enabled': True, 'Period': 1,
                                    'AutoScaling': True, 'Scaling': 1000.0,
                                    'RsdlRatio': 1.2})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              RobustPCA algorithm options
            """

            if opt is None:
                opt = {}
            admm.ADMM.Options.__init__(self, opt)

            if self['AutoRho', 'RsdlTarget'] is None:
                self['AutoRho', 'RsdlTarget'] = 1.0



    itstat_fields_objfn = ('ObjFun', 'NrmNuc', 'NrmL1', 'Cnstr')
    hdrtxt_objfn = ('Fnc', 'NrmNuc', u('Nrmℓ1'), 'Cnstr')
    hdrval_objfun = {'Fnc': 'ObjFun', 'NrmNuc': 'NrmNuc',
                     u('Nrmℓ1'): 'NrmL1', 'Cnstr': 'Cnstr'}



    def __init__(self, S, lmbda=None, opt=None):
        """
        Parameters
        ----------
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        opt : RobustPCA.Options object
          Algorithm options
        """

        if opt is None:
            opt = RobustPCA.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        # Set default lambda value if not specified
        if lmbda is None:
            lmbda = S.shape[0] ** -0.5
        self.lmbda = self.dtype.type(lmbda)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(2.0*self.lmbda + 0.1),
                      dtype=self.dtype)

        Nx = S.size
        super(RobustPCA, self).__init__(Nx, S.shape, S.shape, S.dtype, opt)

        self.S = np.asarray(S, dtype=self.dtype)



    def uinit(self, ushape):
        """Return initialiser for working variable U"""

        if  self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            return (self.lmbda/self.rho)*np.sign(self.Y)



    def solve(self):
        """Start (or re-start) optimisation."""

        super(RobustPCA, self).solve()
        return self.X, self.Y



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.X, self.ss = sp.prox_nuclear(self.S - self.Y - self.U,
                                          1/self.rho)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y = np.asarray(sp.prox_l1(self.S - self.AX - self.U,
                                       self.lmbda/self.rho), dtype=self.dtype)



    def obfn_fvar(self):
        """Variable to be evaluated in computing regularisation term,
        depending on 'fEvalX' option value.
        """

        if self.opt['fEvalX']:
            return self.X
        else:
            return self.cnst_c() - self.cnst_B(self.Y)



    def obfn_gvar(self):
        """Variable to be evaluated in computing regularisation term,
        depending on 'gEvalY' option value.
        """

        if self.opt['gEvalY']:
            return self.Y
        else:
            return self.cnst_c() - self.cnst_A(self.X)



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        if self.opt['fEvalX']:
            rnn = np.sum(self.ss)
        else:
            rnn = sp.norm_nuclear(self.obfn_fvar())
        rl1 = np.sum(np.abs(self.obfn_gvar()))
        cns = np.linalg.norm(self.X + self.Y - self.S)
        obj = rnn + self.lmbda*rl1
        return (obj, rnn, rl1, cns)



    def cnst_A(self, X):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint.  In this case :math:`A \mathbf{x} = \mathbf{x}`.
        """

        return X



    def cnst_AT(self, X):
        r"""Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{x} = \mathbf{x}`.
        """

        return X



    def cnst_B(self, Y):
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem
        constraint.  In this case :math:`B \mathbf{y} = -\mathbf{y}`.
        """

        return Y


    def cnst_c(self):
        r"""Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = \mathbf{s}`.
        """

        return self.S
