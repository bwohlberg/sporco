#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Dictionary learning based on CBPDN sparse coding"""

from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import object

import numpy as np
from scipy import linalg
import collections
import copy

import sporco.linalg as sl
from sporco import util
from sporco import cdict
from sporco.admm import cbpdn
from sporco.admm import ccmod

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class ConvBPDNDictLearn(object):
    """Dictionary learning based on CBPDN and CCMOD
    :cite:`wohlberg-2016-efficient`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
       (1/2) \sum_k \\left \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \\right \|_2^2 + \lambda \sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1 \quad \\text{such that}
       \quad \|\mathbf{d}_m\|_2 = 1

    via interleaved alternation between the ADMM steps of the
    :class:`.ConvBPDN` and :class:`.ConvCnstrMOD` problems.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term \
       :math:`(1/2) \sum_k \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \|_2^2`

       ``Reg`` : Value of regularisation term \
       :math:`\sum_k \sum_m \| \mathbf{x}_{k,m} \|_1`

       ``XPrRsdl`` : Norm of X primal residual

       ``XDlRsdl`` : Norm of X dual residual

       ``DPrRsdl`` : Norm of D primal residual

       ``DDlRsdl`` : Norm of D dual Residual

       ``Rho`` : X penalty parameter

       ``Sigma`` : D penalty parameter

       ``Time`` : Cumulative run time
    """


    class Options(cdict.ConstrainedDict):
        """CBPDN dictionary learning algorithm options.

        Options:

          ``Verbose`` : Flag determining whether iteration status is displayed.

          ``StatusHeader`` : Flag determining whether status header and \
              separator are dislayed

          ``MaxMainIter`` : Maximum main iterations

          ``DictSize`` : Dictionary size vector

          ``CBPDN`` : Options :class:`sporco.admm.cbpdn.ConvBPDN.Options`

          ``CCMOD`` : Options :class:`sporco.admm.ccmod.ConvCnstrMOD.Options`
        """

        defaults = {'Verbose' : False, 'StatusHeader' : True,
                    'MaxMainIter' : 1000, 'DictSize' : None,
                    'CBPDN' : copy.deepcopy(cbpdn.ConvBPDN.Options.defaults),
                    'CCMOD' : \
                    copy.deepcopy(ccmod.ConvCnstrMOD.Options.defaults)}


        def __init__(self, opt=None):
            """Initialise ConvBPDN dictionary learning algorithm options."""

            cdict.ConstrainedDict.__init__(self, {
                'CBPDN' : cbpdn.ConvBPDN.Options({'MaxMainIter' : 1,
                    'AutoRho' : {'Period' : 10, 'AutoScaling' : False,
                    'RsdlRatio' : 10.0, 'Scaling': 2.0, 'RsdlTarget' : 1.0}}),
                'CCMOD' : ccmod.ConvCnstrMOD.Options({'MaxMainIter' : 1,
                    'AutoRho' : {'Period' : 10, 'AutoScaling' : False,
                    'RsdlRatio' : 10.0, 'Scaling': 2.0, 'RsdlTarget' : 1.0}})
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


    def __init__(self, D0, S, lmbda=None, opt=None, dimN=2, dimK=1):
        """
        Initialise a ConvBPDNDictLearn object with problem size and options.

        Parameters
        ----------
        D0 : array_like
          Initial dictionary array
        S : array_like
          Signal array
        lmbda : float
          Regularisation parameter
        opt : :class:`ConvBPDNDictLearn.Options` object
          Algorithm options
        dimN : integer
          Number of spatial/time dimensions
        dimK : integer
          Number of signal dimensions

        """

        self.runtime = 0.0
        self.timer = util.Timer()

        if opt is None:
            opt = ConvBPDNDictLearn.Options()
        self.opt = opt

        # Number of channels in S
        dimC = D0.ndim-dimN-1
        if dimC == 1:
            C = S.shape[dimN]
        else:
            C = 1

        # Number of filters
        M = D0.shape[dimN+dimC]

        # Get tuple indicating spatial size and number of dictionary filters
        if self.opt['DictSize'] is None:
            dsz = D0.shape[0:dimN] + (M,)
        else:
            dsz = self.opt['DictSize']

        # Spatial axis indices and number of samples in each spatial dimension
        axisN = tuple(range(0, dimN))
        Nv = S.shape[0:dimN]

        # Normalise dictionary
        D0 = ccmod.normalise(ccmod.bcrop(D0, dsz, dimN), axisN)

        # Initialise ConvBPDN and ConvCnstrMOD objects
        self.cbpdn = cbpdn.ConvBPDN(D0, S, lmbda, opt['CBPDN'], dimN)
        opt['CCMOD'].update({'Y0' : ccmod.zpad(
            ccmod.stdformD(D0, C, M, dimN), Nv),
                             'U0' : np.zeros(Nv + (C,) + (1,) + (M,))})
        self.ccmod = ccmod.ConvCnstrMOD(None, S, dsz, opt['CCMOD'], dimN, dimK)

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

        # Start timer
        self.timer.start()

        for j in range(self.j, self.j + self.opt['MaxMainIter']):

            # X update
            self.cbpdn.solve()
            self.ccmod.setcoef(self.cbpdn.Y)

            # D update
            self.ccmod.solve()
            self.cbpdn.setdict(self.ccmod.Y)

            # Compute functional value etc.
            Ef = np.sum(self.cbpdn.Df * self.ccmod.Af, axis=self.cbpdn.axisM,
                        keepdims=True) - self.cbpdn.Sf
            dfd = sl.rfl2norm2(Ef, self.cbpdn.S.shape,
                               axis=tuple(range(self.cbpdn.dimN)))/2.0
            l1n = linalg.norm(self.cbpdn.Y.ravel(), 1)
            obj = dfd + self.cbpdn.lmbda*l1n
            cns = linalg.norm((self.ccmod.Pcn(self.ccmod.X) - self.ccmod.X))

            # Get X and D primal and dual residuals
            rX = self.cbpdn.itstat[-1].PrimalRsdl
            sX = self.cbpdn.itstat[-1].DualRsdl
            rD = self.ccmod.itstat[-1].PrimalRsdl
            sD = self.ccmod.itstat[-1].DualRsdl

            # Construct iteration stats for current iteration and append to
            # record of iteration stats
            tk = self.timer.elapsed()
            itstatk = type(self).IterationStats(j, obj, dfd, l1n, rX, sX, 
                                    rD, sD, self.cbpdn.rho, self.ccmod.rho, tk)
            self.itstat.append(itstatk)

            # Display iteration stats if Verbose option enabled
            if self.opt['Verbose']:
                itdsp = (j, obj, dfd, l1n, cns, rX, sX, rD, sD,
                         self.cbpdn.rho, self.ccmod.rho)
                print(fmtstr % itdsp)



        # Record run time
        self.runtime += self.timer.elapsed()

        # Record iteration count
        self.j = j+1

        # Print final seperator string if Verbose option enabled
        if self.opt['Verbose'] and self.opt['StatusHeader']:
            print("-" * nsep)
