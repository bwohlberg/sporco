#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Dictionary learning based on BPDN sparse coding"""

from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import object

import numpy as np
from scipy import linalg
import collections
import copy

from sporco import cdict
from sporco import util
from sporco.admm import bpdn
from sporco.admm import cmod

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class BPDNDictLearn(object):
    """Dictionary learning based on BPDN and CMOD

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{D, X} \;
       (1/2) \| D X - S \|_F^2 + \lambda \| X \|_1 \quad \\text{such that}
       \quad \|\mathbf{d}_m\|_2 = 1

    via interleaved alternation between the ADMM steps of the
    :class:`.BPDN` and :class:`.CnstrMOD` problems.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term \
       :math:`(1/2) \| D X - S \|_F^2`

       ``Reg`` : Value of regularisation term \
       :math:`\| X \|_1`

       ``XPrRsdl`` : Norm of X primal residual

       ``XDlRsdl`` : Norm of X dual residual

       ``DPrRsdl`` : Norm of D primal residual

       ``DDlRsdl`` : Norm of D dual Residual

       ``Rho`` : X penalty parameter

       ``Sigma`` : D penalty parameter

       ``Time`` : Cumulative run time
    """


    class Options(cdict.ConstrainedDict):
        """BPDN dictionary learning algorithm options.

        Options:

          ``Verbose`` : Flag determining whether iteration status is displayed.

          ``StatusHeader`` : Flag determining whether status header and \
              separator are dislayed

          ``MaxMainIter`` : Maximum main iterations

          ``BPDN`` : Options :class:`sporco.admm.bpdn.BPDN.Options`

          ``CMOD`` : Options :class:`sporco.admm.cmod.CnstrMOD.Options`
        """

        defaults = {'Verbose' : False, 'StatusHeader' : True,
                    'MaxMainIter' : 1000,
                    'BPDN' : copy.deepcopy(bpdn.BPDN.Options.defaults),
                    'CMOD' : copy.deepcopy(cmod.CnstrMOD.Options.defaults)}


        def __init__(self, opt=None):
            """Initialise BPDN dictionary learning algorithm options."""

            cdict.ConstrainedDict.__init__(self, {
                'BPDN' : bpdn.BPDN.Options({'MaxMainIter' : 1,
                    'AutoRho' : {'Period' : 10, 'AutoScaling' : False,
                    'RsdlRatio' : 10.0, 'Scaling': 2.0, 'RsdlTarget' : 1.0}}),
                'CMOD' : cmod.CnstrMOD.Options({'MaxMainIter' : 1,
                                    'AutoRho' : {'Period' : 10}})
                })

            if opt is None:
                opt = {}
            self.update(opt)


    IterationStats = collections.namedtuple('IterationStats',
            ['Iter', 'ObjFun', 'DFid', 'Reg', 'XPrRsdl', 'XDlRsdl',
             'DPrRsdl', 'DDlRsdl', 'Rho', 'Sigma', 'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""
    hdrtxt = ['Itn', 'Fnc', 'DFid', 'l1', 'Cnstr', 'r_X', 's_X',
              'r_D', 's_D', 'rho_X', 'rho_D']
    """Display column header text"""


    def __init__(self, D0, S, lmbda=None, opt=None):
        """
        Initialise a BPDNDictLearn object with problem size and options.

        Parameters
        ----------
        D0 : array_like, shape (N, M)
          Initial dictionary matrix
        S : array_like, shape (N, K)
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        opt : :class:`BPDNDictLearn.Options` object
          Algorithm options
        """

        self.runtime = 0.0
        self.timer = util.Timer()

        if opt is None:
            opt = BPDNDictLearn.Options()
        self.opt = opt
        Nc = D0.shape[1]
        Nm = S.shape[1]
        D0 = cmod.getPcn(opt['CMOD'])(D0)
        self.bpdn = bpdn.BPDN(D0, S, lmbda, opt['BPDN'])
        opt['CMOD'].update({'Y0' : D0, 'U0' : np.zeros((S.shape[0], Nc))})
        self.cmod = cmod.CnstrMOD(self.bpdn.Y, S, (Nc, Nm), opt['CMOD'])

        self.itstat = []
        self.j = 0

        self.runtime += self.timer.elapsed()



    def solve(self):
        """Run optimisation"""

        if self.opt['Verbose']:
            hdrtxt = type(self).hdrtxt
            # Call utility function to construct status display formatting
            hdrstr, fmtstr, nsep = util.solve_status_str(hdrtxt,
                                        type(self).fwiter, type(self).fpothr)
            # Print header and seperator strings
            if self.opt['StatusHeader']:
                print(hdrstr)
                print("-" * nsep)

        # Reset timer
        self.timer.start()

        for j in range(0, self.opt['MaxMainIter']):

            # X update
            self.bpdn.solve()
            self.cmod.setcoef(self.bpdn.Y)

            # D update
            self.cmod.solve()
            self.bpdn.setdict(self.cmod.Y)

            # Compute functional value etc.
            Ef = self.bpdn.D.dot(self.cmod.A) - self.bpdn.S
            dfd = 0.5*(linalg.norm(Ef)**2)
            l1n = linalg.norm(self.bpdn.Y.ravel(), 1)
            obj = dfd + self.bpdn.lmbda*l1n
            cns = linalg.norm((self.cmod.Pcn(self.cmod.X) - self.cmod.X))

            # Get X and D primal and dual residuals
            rX = self.bpdn.itstat[-1].PrimalRsdl
            sX = self.bpdn.itstat[-1].DualRsdl
            rD = self.cmod.itstat[-1].PrimalRsdl
            sD = self.cmod.itstat[-1].DualRsdl

            # Construct iteration stats for current iteration and append to
            # record of iteration stats
            tk = self.timer.elapsed()
            itstatk = type(self).IterationStats(j, obj, dfd, l1n,
                                    rX, sX, rD, sD, self.bpdn.rho,
                                    self.cmod.rho, tk)
            self.itstat.append(itstatk)

            # Display iteration stats if Verbose option enabled
            if self.opt['Verbose']:
                itdsp = (j, obj, dfd, l1n, cns, rX, sX, rD, sD,
                         self.bpdn.rho, self.cmod.rho)
                print(fmtstr % itdsp)



        # Record run time
        self.runtime += self.timer.elapsed()

        # Record iteration count
        self.j = j+1

        # Print final seperator string if Verbose option enabled
        if self.opt['Verbose'] and self.opt['StatusHeader']:
            print("-" * nsep)
