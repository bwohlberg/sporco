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
import copy

from sporco.util import u
from sporco.admm import cbpdn
from sporco.admm import ccmod
from sporco.admm import dictlrn

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class ConvBPDNDictLearn(dictlrn.DictLearn):
    """Dictionary learning based on ConvBPDN and ConvCnstrMOD
    :cite:`wohlberg-2016-efficient`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
       (1/2) \sum_k \\left \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \\right \|_2^2 + \lambda \sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1 \quad \\text{such that}
       \quad \mathbf{d}_m \in C \;\;,

    where :math:`C` is the feasible set consisting of filters with
    unit norm and constrained support, via interleaved alternation
    between the ADMM steps of the :class:`.ConvBPDN` and
    :class:`.ConvCnstrMOD` problems. The multi-channel variants
    supported by :class:`.ConvCnstrMOD` are also supported.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term \
       :math:`(1/2) \sum_k \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \|_2^2`

       ``RegL1`` : Value of regularisation term \
       :math:`\sum_k \sum_m \| \mathbf{x}_{k,m} \|_1`

       ``Cnstr`` : Constraint violation measure

       ``XPrRsdl`` : Norm of X primal residual

       ``XDlRsdl`` : Norm of X dual residual

       ``XRho`` : X penalty parameter

       ``DPrRsdl`` : Norm of D primal residual

       ``DDlRsdl`` : Norm of D dual residual

       ``DRho`` : D penalty parameter

       ``Time`` : Cumulative run time
    """


    class Options(dictlrn.DictLearn.Options):
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
                'CCMOD' : copy.deepcopy(ccmod.ConvCnstrMOD.Options.defaults)}


        def __init__(self, opt=None):
            """Initialise ConvBPDN dictionary learning algorithm options."""

            dictlrn.DictLearn.Options.__init__(self, {
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



    def __init__(self, D0, S, lmbda=None, opt=None, dimK=1, dimN=2):
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
        dimK : int, optional (default 1)
          Number of signal dimensions
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        if opt is None:
            opt = ConvBPDNDictLearn.Options()
        self.opt = opt

        # Get dictionary size
        if self.opt['DictSize'] is None:
            dsz = D0.shape
        else:
            dsz = self.opt['DictSize']

        # Construct object representing problem dimensions
        cri = ccmod.ConvRepIndexing(dsz, S, dimK, dimN)

        # Normalise dictionary
        D0 = ccmod.getPcn0(opt['CCMOD', 'ZeroMean'], dsz, dimN,
                           dimC=cri.dimCd)(D0)

        # Modify D update options to include initial values for Y and U
        opt['CCMOD'].update({'Y0' : ccmod.zpad(
            ccmod.stdformD(D0, cri.C, cri.M, dimN), cri.Nv),
                             'U0' : np.zeros(cri.shpD)})

        # Create X update object
        xstep = cbpdn.ConvBPDN(D0, S, lmbda, opt['CBPDN'], dimK=dimK,
                               dimN=dimN)

        # Create D update object
        dstep = ccmod.ConvCnstrMOD(None, S, dsz, opt['CCMOD'], dimK=dimK,
                                    dimN=dimN)

        # Configure iteration statistics reporting
        isc = dictlrn.IterStatsConfig(
            isfld = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr', 'XPrRsdl',
                     'XDlRsdl', 'XRho', 'DPrRsdl', 'DDlRsdl', 'DRho', 'Time'],
            isxmap = {'ObjFun' : 'ObjFun', 'DFid' : 'DFid', 'RegL1' : 'RegL1',
                      'XPrRsdl' : 'PrimalRsdl', 'XDlRsdl' : 'DualRsdl',
                      'XRho' : 'Rho'},
            isdmap = {'Cnstr' :  'Cnstr', 'DPrRsdl' : 'PrimalRsdl',
                      'DDlRsdl' : 'DualRsdl', 'DRho' : 'Rho'},
            evlmap = {},
            hdrtxt = ['Itn', 'Fnc', 'DFid', 'l1', 'Cnstr', 'r_X', 's_X',
                      u('ρ_X'), 'r_D', 's_D', u('ρ_D')],
            hdrmap = {'Itn' : 'Iter', 'Fnc' : 'ObjFun', 'DFid' : 'DFid',
                      'l1' : 'RegL1', 'Cnstr' : 'Cnstr', 'r_X' : 'XPrRsdl',
                      's_X' : 'XDlRsdl', u('ρ_X') : 'XRho', 'r_D' : 'DPrRsdl',
                      's_D' : 'DDlRsdl', u('ρ_D') : 'DRho'}
            )

        # Call parent constructor
        super(ConvBPDNDictLearn, self).__init__(xstep, dstep, opt, isc)
