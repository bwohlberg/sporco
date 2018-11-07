# -*- coding: utf-8 -*-
# Copyright (C) 2016-2018 by Brendt Wohlberg <brendt@ieee.org>
#                            Cristina Garcia-Cardona <cgarciac@lanl.gov>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for FISTA algorithm for the BPDN problem"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import copy
import numpy as np
from scipy import linalg

from sporco.fista import fista
import sporco.linalg as sl
from sporco.util import u



__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""



class BPDN(fista.FISTA):
    r"""
    Base class for FISTA algorithm for the Basis Pursuit DeNoising (BPDN)
    :cite:`chen-1998-atomic` problem.

    |

    .. inheritance-diagram:: BPDN
       :parts: 2

    |

    The generic problem form is

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
        f( \{ \mathbf{x}_m \} ) + \lambda g( \{ \mathbf{x}_m \} )

    where :math:`f = (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2`, and
    :math:`g(\cdot)` is a penalty term or the indicator function of a
    constraint; with input image :math:`\mathbf{s}`, dictionary filters
    :math:`D`, and coefficient maps :math:`\mathbf{x}`. It is solved via
    the FISTA formulation

    Proximal step

    .. math::
       \mathbf{x}_k = \mathrm{prox}_{t_k}(g) (\mathbf{y}_k - 1/L \nabla
       f(\mathbf{y}_k) ) \;\;.

    Combination step

    .. math::
       \mathbf{y}_{k+1} = \mathbf{x}_k + \left( \frac{t_k - 1}{t_{k+1}}
       \right) (\mathbf{x}_k - \mathbf{x}_{k-1}) \;\;,

    with :math:`t_{k+1} = \frac{1 + \sqrt{1 + 4 t_k^2}}{2}`.


    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| D
       \mathbf{x} - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\lambda \|
       \mathbf{x} \|_1`

       ``Rsdl`` : Residual

       ``L`` : Inverse of gradient step parameter

       ``Time`` : Cumulative run time
    """


    class Options(fista.FISTA.Options):
        r"""BPDN algorithm options

        Options include all of those defined in
        :class:`.fista.FISTA.Options`, together with
        additional options:

          ``L1Weight`` : An array of weights for the :math:`\ell_1`
          norm. The array shape must be such that the array is
          compatible for multiplication with the X/Y variables. If this
          option is defined, the regularization term is :math:`\lambda
          \| \mathbf{w}_m \odot \mathbf{x} \|_1` where
          :math:`\mathbf{w}` denotes slices of the weighting array.

        """

        defaults = copy.deepcopy(fista.FISTADFT.Options.defaults)
        defaults.update({'L1Weight': 1.0})
        defaults.update({'L': 500.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              BPDN algorithm options
            """

            if opt is None:
                opt = {}
            fista.FISTA.Options.__init__(self, opt)


        def __setitem__(self, key, value):
            """Set options."""

            fista.FISTA.Options.__setitem__(self, key, value)


    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1'}



    def __init__(self, D, S, lmbda=None, opt=None):
        """
        This class supports an arbitrary number of spatial dimensions,
        `dimN`, with a default of 2. The input dictionary `D` is either
        `dimN` + 1 dimensional, in which case each spatial component
        (image in the default case) is assumed to consist of a single
        channel, or `dimN` + 2 dimensional, in which case the final
        dimension is assumed to contain the channels (e.g. colour
        channels in the case of images). The input signal set `S` is
        either `dimN` dimensional (no channels, only one signal),
        `dimN` + 1 dimensional (either multiple channels or multiple
        signals), or `dimN` + 2 dimensional (multiple channels and
        multiple signals). Determination of problem dimensions is
        handled by :class:`.cnvrep.CSC_ConvRepIndexing`.


        Parameters
        ----------
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        lmbda : float
          Regularisation parameter
        opt : :class:`BPDN.Options` object
          Algorithm options
        """

        # Set default options if none specified
        if opt is None:
            opt = BPDN.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        # Set default lambda value if not specified
        if lmbda is None:
            DTS = D.T.dot(S)
            lmbda = 0.1*abs(DTS).max()

        # Set l1 term scaling and weight array
        self.lmbda = self.dtype.type(lmbda)
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)

        # Call parent class __init__
        Nc = D.shape[1]
        Nm = S.shape[1]

        xshape = (Nc, Nm)
        super(BPDN, self).__init__(Nc * Nm, xshape, S.dtype, opt)

        self.S = np.asarray(S, dtype=self.dtype)

        self.store_prev()
        self.Y = self.X.copy()
        self.Yprv = self.Y.copy() + 1e5

        self.setdict(D)



    def setdict(self, D):
        """Set dictionary array."""

        self.D = np.asarray(D, dtype=self.dtype)



    def getcoef(self):
        """Get final coefficient array."""

        return self.X



    def eval_grad(self):
        """Compute gradient in spatial domain for variable Y."""

        # Compute D^T(D Y - S)
        return self.D.T.dot(self.D.dot(self.Y) - self.S)



    def eval_proxop(self, V):
        """Compute proximal operator of :math:`g`."""

        return np.asarray(sl.shrink1(V, (self.lmbda/self.L) * self.wl1),
                          dtype=self.dtype)



    def eval_R(self, V):
        """Evaluate smooth term in V."""

        return self.D.dot(V) - self.S



    def rsdl(self):
        """Compute fixed point residual."""

        return linalg.norm((self.X - self.Yprv).ravel())



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_f()
        reg = self.obfn_reg()
        obj = dfd + reg[0]
        return (obj, dfd) + reg[1:]



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = linalg.norm((self.wl1 * self.X).ravel(), 1)
        return (self.lmbda*rl1, rl1)



    def obfn_f(self, X=None):
        r"""Compute data fidelity term :math:`(1/2) \| D \mathbf{x} -
        \mathbf{s} \|_2^2`.
        """
        if X is None:
            X = self.X

        return 0.5 * linalg.norm((self.D.dot(X) - self.S).ravel())**2



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            X = self.X
        return self.D.dot(self.X)
