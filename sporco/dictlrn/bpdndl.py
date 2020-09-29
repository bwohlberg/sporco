# -*- coding: utf-8 -*-
# Copyright (C) 2015-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Dictionary learning based on BPDN sparse coding"""

from __future__ import print_function, absolute_import

import copy
import numpy as np

from sporco.util import u
from sporco.admm import bpdn
from sporco.admm import cmod
from sporco.dictlrn import dictlrn

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class BPDNDictLearn(dictlrn.DictLearn):
    r"""
    Dictionary learning based on BPDN and CnstrMOD

    |

    .. inheritance-diagram:: BPDNDictLearn
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{D, X} \; (1/2) \| D X - S \|_F^2 + \lambda \|
       X \|_1 \quad \text{such that} \quad \|\mathbf{d}_m\|_2 = 1

    via interleaved alternation between the ADMM steps of the
    :class:`.admm.bpdn.BPDN` and :class:`.admm.cmod.CnstrMOD` problems.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| D X - S \|_F^2`

       ``RegL1`` : Value of regularisation term :math:`\| X \|_1`

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
        """BPDN dictionary learning algorithm options.

        Options include all of those defined in
        :class:`sporco.dictlrn.dictlrn.DictLearn.Options`, together with
        additional options:

          ``AccurateDFid`` : Flag determining whether data fidelity term is
          estimated from the value computed in the X update (``False``) or
          is computed after every outer iteration over an X update and a D
          update (``True``), which is slower but more accurate.

          ``BPDN`` : Options :class:`sporco.admm.bpdn.BPDN.Options`

          ``CMOD`` : Options :class:`sporco.admm.cmod.CnstrMOD.Options`
        """

        defaults = copy.deepcopy(dictlrn.DictLearn.Options.defaults)
        defaults.update(
            {'AccurateDFid': False,
             'BPDN': copy.deepcopy(bpdn.BPDN.Options.defaults),
             'CMOD': copy.deepcopy(cmod.CnstrMOD.Options.defaults)})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              BPDNDictLearn algorithm options
            """

            dictlrn.DictLearn.Options.__init__(
                self, {'BPDN': bpdn.BPDN.Options(
                    {'MaxMainIter': 1, 'AutoRho':
                     {'Period': 10, 'AutoScaling': False, 'RsdlRatio': 10.0,
                      'Scaling': 2.0, 'RsdlTarget': 1.0}}),
                       'CMOD': cmod.CnstrMOD.Options(
                           {'MaxMainIter': 1, 'AutoRho': {'Period': 10},
                            'AuxVarObj': False})
                      })

            if opt is None:
                opt = {}
            self.update(opt)



    def __init__(self, D0, S, lmbda=None, opt=None):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/bpdndl_init.svg
           :width: 20%
           :target: ../_static/jonga/bpdndl_init.svg

        |


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

        if opt is None:
            opt = BPDNDictLearn.Options()
        self.opt = opt

        # Normalise dictionary according to D update options
        D0 = cmod.getPcn(opt['CMOD', 'ZeroMean'])(D0)

        # Modify D update options to include initial values for Y and U
        Nc = D0.shape[1]
        opt['CMOD'].update({'Y0': D0, 'U0': np.zeros((S.shape[0], Nc))})

        # Create X update object
        xstep = bpdn.BPDN(D0, S, lmbda, opt['BPDN'])

        # Create D update object
        Nm = S.shape[1]
        dstep = cmod.CnstrMOD(xstep.Y, S, (Nc, Nm), opt['CMOD'])

        # Configure iteration statistics reporting
        if self.opt['AccurateDFid']:
            isxmap = {'XPrRsdl': 'PrimalRsdl', 'XDlRsdl': 'DualRsdl',
                      'XRho': 'Rho'}
            evlmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1'}
        else:
            isxmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1',
                      'XPrRsdl': 'PrimalRsdl', 'XDlRsdl': 'DualRsdl',
                      'XRho': 'Rho'}
            evlmap = {}
        isc = dictlrn.IterStatsConfig(
            isfld=['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr', 'XPrRsdl',
                   'XDlRsdl', 'XRho', 'DPrRsdl', 'DDlRsdl', 'DRho', 'Time'],
            isxmap=isxmap,
            isdmap={'Cnstr':  'Cnstr', 'DPrRsdl': 'PrimalRsdl',
                    'DDlRsdl': 'DualRsdl', 'DRho': 'Rho'},
            evlmap=evlmap,
            hdrtxt=['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr', 'r_X', 's_X',
                    u('ρ_X'), 'r_D', 's_D', u('ρ_D')],
            hdrmap={'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
                    u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr', 'r_X': 'XPrRsdl',
                    's_X': 'XDlRsdl', u('ρ_X'): 'XRho', 'r_D': 'DPrRsdl',
                    's_D': 'DDlRsdl', u('ρ_D'): 'DRho'}
            )

        # Call parent constructor
        super(BPDNDictLearn, self).__init__(xstep, dstep, opt, isc)



    def evaluate(self):
        """Evaluate functional value of previous iteration"""

        if self.opt['AccurateDFid']:
            D = self.dstep.var_y()
            X = self.xstep.var_y()
            S = self.xstep.S
            dfd = 0.5*np.linalg.norm((D.dot(X) - S))**2
            rl1 = np.sum(np.abs(X))
            return dict(DFid=dfd, RegL1=rl1, ObjFun=dfd+self.xstep.lmbda*rl1)
        else:
            return None
