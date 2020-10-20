# -*- coding: utf-8 -*-
# Copyright (C) 2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Dictionary learning based on weighted BPDN sparse coding"""

from __future__ import print_function, absolute_import

import copy
import numpy as np

from sporco.util import u
from sporco.array import atleast_nd
from sporco.pgm import bpdn
from sporco.pgm import cmod
from sporco.dictlrn import dictlrn

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class WeightedBPDNDictLearn(dictlrn.DictLearn):
    r"""
    Dictionary learning based on weighted BPDN and CnstrMOD

    |

    .. inheritance-diagram:: WeightedBPDNDictLearn
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{D, X} \; (1/2) \| D X - S \|_W^2 + \lambda \|
       X \|_1 \quad \text{such that} \quad \|\mathbf{d}_m\|_2 = 1

    via interleaved alternation between the PGM steps of the
    :class:`.pgm.bpdn.WeightedBPDN` and
    :class:`.pgm.cmod.WeightedCnstrMOD` problems. Note that
    :math:`\| \cdot \|_W` denotes a non-standard definition of the
    weighted Frobenius norm defined as

    .. math::
       \| X \|_W^2 = \| W^{1/2} \odot X \|_F

    so that

    .. math::
       \| X \|_W^2 = \sum_i \| W_i^{1/2} \mathbf{x}_i \|_2^2 =
                     \sum_i \| \mathbf{x}_i \|_{W_i}^2 \;,

    where :math:`\mathbf{x}_i` and :math:`\mathbf{w}_i` are the
    :math:`i^{\text{th}}` columns of :math:`X` and :math:`W`
    respectively, and :math:`W_i = \mathrm{diag}(\mathbf{w}_i)`.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| D X - S \|_W^2`

       ``RegL1`` : Value of regularisation term :math:`\| X \|_1`

       ``Cnstr`` : Constraint violation measure

       ``XRsdl`` : Norm of X residual

       ``XL`` : X inverse of gradient step parameter

       ``DRsdl`` : Norm of D residual

       ``DL`` : D inverse of gradient step parameter

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

          ``BPDN`` : Options :class:`sporco.pgm.bpdn.WeightedBPDN.Options`

          ``CMOD`` : Options :class:`sporco.pgm.cmod.WeightedCnstrMOD.Options`
        """

        defaults = copy.deepcopy(dictlrn.DictLearn.Options.defaults)
        defaults.update(
            {'AccurateDFid': False,
             'BPDN': copy.deepcopy(bpdn.WeightedBPDN.Options.defaults),
             'CMOD': copy.deepcopy(cmod.WeightedCnstrMOD.Options.defaults)})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              WeightedBPDNDictLearn algorithm options
            """

            dictlrn.DictLearn.Options.__init__(
                self, {'BPDN': bpdn.WeightedBPDN.Options(
                    {'MaxMainIter': 1}),
                       'CMOD': cmod.WeightedCnstrMOD.Options(
                    {'MaxMainIter': 1})
                      })

            if opt is None:
                opt = {}
            self.update(opt)



    def __init__(self, D0, S, lmbda=None, W=None, opt=None):
        """
        Parameters
        ----------
        D0 : array_like, shape (N, M)
          Initial dictionary matrix
        S : array_like, shape (N, K)
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        W : array_like, shape (N, K)
          Weight matrix
        opt : :class:`WeightedBPDNDictLearn.Options` object
          Algorithm options
        """

        if opt is None:
            opt = WeightedBPDNDictLearn.Options()
        self.opt = opt

        # Normalise dictionary according to D update options
        D0 = cmod.getPcn(opt['CMOD', 'ZeroMean'],
                         opt['CMOD', 'NonNegCoef'])(D0)

        # Modify D update options to include initial values for Y and U
        Nc = D0.shape[1]
        opt['CMOD'].update({'X0': D0})

        # Create X update object
        xstep = bpdn.WeightedBPDN(D0, S, lmbda, W=W, opt=opt['BPDN'])

        # Create D update object
        Nm = S.shape[1]
        dstep = cmod.WeightedCnstrMOD(xstep.Y, S, W=W, dsz=(Nc, Nm),
                                      opt=opt['CMOD'])

        if W is None:
            W = np.array([1.0], dtype=xstep.dtype)
        if W.ndim > 0:
            W = atleast_nd(2, W)
        self.W = np.asarray(W, dtype=xstep.dtype)

        # Configure iteration statistics reporting
        if self.opt['AccurateDFid']:
            isxmap = {'XRsdl': 'Rsdl', 'XL': 'L'}
            evlmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1'}
        else:
            isxmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1',
                      'XRsdl': 'Rsdl', 'XL': 'L'}
            evlmap = {}
        isc = dictlrn.IterStatsConfig(
            isfld=['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr', 'XRsdl',
                   'XL', 'DRsdl', 'DL', 'Time'],
            isxmap=isxmap,
            isdmap={'Cnstr': 'Cnstr', 'DRsdl': 'Rsdl', 'DL': 'L'},
            evlmap=evlmap,
            hdrtxt=['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr', 'X_Rsdl',
                    'X_L', 'D_Rsdl', 'D_L'],
            hdrmap={'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
                    u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr', 'X_Rsdl': 'XRsdl',
                    'X_L': 'XL', 'D_Rsdl': 'DRsdl', 'D_L': 'DL'}
            )

        # Call parent constructor
        super(WeightedBPDNDictLearn, self).__init__(xstep, dstep, opt, isc)



    def evaluate(self):
        """Evaluate functional value of previous iteration"""

        if self.opt['AccurateDFid']:
            D = self.dstep.var_x()
            X = self.xstep.var_x()
            S = self.xstep.S
            dfd = 0.5*np.linalg.norm(np.sqrt(self.W) * (D.dot(X) - S))**2
            rl1 = np.sum(np.abs(X))
            return dict(DFid=dfd, RegL1=rl1, ObjFun=dfd+self.xstep.lmbda*rl1)
        else:
            return None
