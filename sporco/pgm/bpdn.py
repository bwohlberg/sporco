# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 by Brendt Wohlberg <brendt@ieee.org>
#                            Cristina Garcia-Cardona <cgarciac@lanl.gov>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for PGM algorithm for the BPDN problem"""

from __future__ import division, absolute_import, print_function

import copy
import numpy as np

from sporco.util import u
from sporco.array import atleast_nd
from sporco.prox import prox_l1
from sporco.pgm import pgm


__author__ = """\n""".join(['Cristina Garcia-Cardona <cgarciac@lanl.gov>',
                            'Brendt Wohlberg <brendt@ieee.org>'])


class BPDN(pgm.PGM):
    r"""
    Class for PGM algorithm for the Basis Pursuit DeNoising (BPDN)
    :cite:`chen-1998-atomic` problem.

    |

    .. inheritance-diagram:: BPDN
       :parts: 2

    |

    The problem form is

    .. math::
       \mathrm{argmin}_\mathbf{x} \; (1/2) \| D \mathbf{x} - \mathbf{s}
       \|_2^2  + \lambda \| \mathbf{x} \|_1

    where :math:`\mathbf{s}` is the input vector/matrix, :math:`D` is
    the dictionary, and :math:`\mathbf{x}` is the sparse representation.

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


    class Options(pgm.PGM.Options):
        r"""BPDN algorithm options

        Options include all of those defined in
        :class:`.pgm.PGM.Options`, together with
        additional options:

          ``NonNegCoef`` : If ``True``, force solution to be non-negative.

          ``L1Weight`` : An array of weights for the :math:`\ell_1`
          norm. The array shape must be such that the array is
          compatible for multiplication with the X/Y variables. If this
          option is defined, the regularization term is :math:`\lambda
          \| \mathbf{w} \odot \mathbf{x} \|_1` where
          :math:`\mathbf{w}` denotes the weighting array.

        """

        defaults = copy.deepcopy(pgm.PGM.Options.defaults)
        defaults.update({'NonNegCoef': False, 'L1Weight': 1.0, 'L': 500.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              BPDN algorithm options
            """

            if opt is None:
                opt = {}
            pgm.PGM.Options.__init__(self, opt)


        def __setitem__(self, key, value):
            """Set options."""

            pgm.PGM.Options.__setitem__(self, key, value)


    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1'}



    def __init__(self, D, S, lmbda=None, opt=None):
        """
        Parameters
        ----------
        D : array_like
          Dictionary array (2d)
        S : array_like
          Signal array (1d or 2d)
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
            lmbda = 0.1 * abs(DTS).max()

        # Set l1 term scaling and weight array
        self.lmbda = self.dtype.type(lmbda)
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)

        # Call parent class __init__
        Nc = D.shape[1]
        Nm = S.shape[1]

        xshape = (Nc, Nm)
        super(BPDN, self).__init__(xshape, S.dtype, opt)

        self.S = np.asarray(S, dtype=self.dtype)

        self.Y = self.X.copy()
        self.Yprv = self.Y.copy() + 1e3

        self.setdict(D)



    def setdict(self, D):
        """Set dictionary array."""

        self.D = np.asarray(D, dtype=self.dtype)



    def getcoef(self):
        """Get final coefficient array."""

        return self.X



    def grad_f(self, V=None):
        """Compute gradient of data fidelity for variable V or self.Y."""

        if V is None:
            V = self.Y
        # Compute D^T(D V - S)
        return self.D.T.dot(self.D.dot(V) - self.S)



    def prox_g(self, V):
        """Compute proximal operator of :math:`g`."""

        U = np.asarray(prox_l1(V, (self.lmbda / self.L) * self.wl1),
                       dtype=self.dtype)
        if self.opt['NonNegCoef']:
            U[U < 0.0] = 0.0
        return U



    def hessian_f(self, V):
        """Compute Hessian of :math:`f` applied to V."""

        hessfv = self.D.T.dot(self.D.dot(V))
        return hessfv



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

        rl1 = np.linalg.norm((self.wl1 * self.X).ravel(), 1)
        return (self.lmbda * rl1, rl1)



    def obfn_f(self, X=None):
        r"""Compute data fidelity term :math:`(1/2) \| D \mathbf{x} -
        \mathbf{s} \|_2^2`.
        """
        if X is None:
            X = self.X
        return 0.5 * np.linalg.norm((self.D.dot(X) - self.S).ravel())**2



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            X = self.X
        return self.D.dot(self.X)





class WeightedBPDN(BPDN):
    r"""
    Class for PGM algorithm for variant of BPDN with a weighted
    :math:`\ell_2` data fidelity term.

    |

    .. inheritance-diagram:: WeightedBPDN
       :parts: 2

    |

    The problem form is

    .. math::
       \mathrm{argmin}_\mathbf{x} \; (1/2) \| D \mathbf{x} - \mathbf{s}
       \|_W^2  + \lambda \| \mathbf{x} \|_1

    where :math:`\mathbf{s}` is the input vector/matrix, :math:`D` is
    the dictionary, :math:`\mathbf{x}` is the sparse representation,
    and :math:`\| \cdot \|_W` denotes the weighted :math:`\ell_2`
    norm defined as

    .. math::
       \| \mathbf{x} \|_W = \| W^{1/2} \mathbf{x} \|_2 \;.

    While this norm is defined for any symmetric positive definite
    :math:`W`, the interface of this class only supports diagonal
    :math:`W` in that the `W` parameter of the constructor is actually
    a vector :math:`\mathbf{w}` such that
    :math:`W = \mathrm{diag}(\mathbf{w})`.

    When the input is a matrix, i.e. the problem is of the form

    .. math::
       \mathrm{argmin}_X \; (1/2) \| D X - S \|_W^2  + \lambda \| S \|_1

    where :math:`S` and :math:`X` are matrices rather than vectors,
    it is important to note that :math:`\| \cdot \|_W` does *not*
    denote the standard weighted Frobenius norm :math:`\| X \|_W =
    \| W^{1/2} X W^{1/2} \|_F`, and is instead defined as

    .. math::
       \| X \|_W^2 = \| W^{1/2} \odot X \|_F

    so that

    .. math::
       \| X \|_W^2 = \sum_i \| W_i^{1/2} \mathbf{x}_i \|_2^2 =
                     \sum_i \| \mathbf{x}_i \|_{W_i}^2 \;,

    where :math:`\mathbf{x}_i` and :math:`\mathbf{w}_i` are the
    :math:`i^{\text{th}}` columns of :math:`X` and :math:`W`
    respectively, and :math:`W_i = \mathrm{diag}(\mathbf{w}_i)`.


    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| D
       \mathbf{x} - \mathbf{s} \|_W^2`

       ``RegL1`` : Value of regularisation term :math:`\lambda \|
       \mathbf{x} \|_1`

       ``Rsdl`` : Residual

       ``L`` : Inverse of gradient step parameter

       ``Time`` : Cumulative run time
    """


    def __init__(self, D, S, lmbda=None, W=None, opt=None):
        """
        Parameters
        ----------
        D : array_like
          Dictionary array (2d)
        S : array_like
          Signal array (1d or 2d)
        lmbda : float
          Regularisation parameter
        W : array_like
          Weight array (1d or 2d)
        opt : :class:`WeightedBPDN.Options` object
          Algorithm options
        """

        super(WeightedBPDN, self).__init__(D, S, lmbda, opt)

        if W is None:
            W = np.array([1.0], dtype=self.dtype)
        if W.ndim > 0:
            W = atleast_nd(2, W)
        self.W = np.asarray(W, dtype=self.dtype)



    def grad_f(self, V=None):
        """Compute gradient of data fidelity for variable V or self.Y."""

        if V is None:
            V = self.Y
        # Compute D^T (W \odot (D V - S))
        return self.D.T.dot(self.W * (self.D.dot(V) - self.S))



    def obfn_f(self, X=None):
        r"""Compute data fidelity term :math:`(1/2) \| D \mathbf{x} -
        \mathbf{s} \|_W^2`.
        """
        if X is None:
            X = self.X
        return 0.5 * np.linalg.norm(
            (np.sqrt(self.W) * (self.D.dot(X) - self.S)).ravel())**2
