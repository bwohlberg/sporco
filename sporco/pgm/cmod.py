# -*- coding: utf-8 -*-
# Copyright (C) 2025 by Brendt Wohlberg <brendt@ieee.org>
#                       Cristina Garcia-Cardona <cgarciac@lanl.gov>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""PGM algorithm for the CMOD problem"""

from __future__ import division, absolute_import, print_function

import copy
import numpy as np

from sporco.array import atleast_nd
from sporco.pgm import pgm


__author__ = """\n""".join(['Brendt Wohlberg <brendt@ieee.org>',
                            'Cristina Garcia-Cardona <cgarciac@lanl.gov>'])


class CnstrMOD(pgm.PGM):
    r"""
    PGM algorithm for a constrained variant of the Method of Optimal
    Directions (MOD) :cite:`engan-1999-method` problem, referred to here
    as Constrained MOD (CMOD).

    |

    .. inheritance-diagram:: CnstrMOD
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_D \| D X - S \|_2^2 + \iota_C(D)

    where :math:`\iota_C(\cdot)` is the indicator function of feasible
    set :math:`C` consisting of matrices with unit-norm columns.

    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| D X - S
       \|_2^2`

       ``Cnstr`` : Constraint violation measure

       ``Rsdl`` : Residual

       ``L`` : Inverse of gradient step parameter

       ``Time`` : Cumulative run time
    """


    class Options(pgm.PGM.Options):
        r"""CnstrMOD algorithm options

        Options include all of those defined in
        :class:`.pgm.PGM.Options`, together with
        additional options:

         ``ZeroMean`` : Flag indicating whether the solution
          dictionary :math:`D` should have zero-mean components.

         ``NonNegCoef`` : Flag indicating whether the solution
          dictionary :math:`D` should have non-negative coefficients.

        Note that ``ZeroMean`` and ``NonNegCoef`` may not both be True.
        """

        defaults = copy.deepcopy(pgm.PGM.Options.defaults)
        defaults.update({'ZeroMean': False, 'NonNegCoef': False,
                         'L': 500.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              CnstrMOD algorithm options
            """

            if opt is None:
                opt = {}
            pgm.PGM.Options.__init__(self, opt)


        def __setitem__(self, key, value):
            """Set options."""

            pgm.PGM.Options.__setitem__(self, key, value)


    itstat_fields_objfn = ('DFid', 'Cnstr')
    hdrtxt_objfn = ('DFid', 'Cnstr')
    hdrval_objfun = {'DFid': 'DFid', 'Cnstr': 'Cnstr'}



    def __init__(self, Z, S, dsz=None, opt=None):
        """
        Parameters
        ----------
        Z : array_like, shape (M, K)
          Sparse representation coefficient matrix
        S : array_like, shape (N, K)
          Signal vector or matrix
        dsz : tuple
          Dictionary size
        opt : :class:`CnstrMOD.Options` object
          Algorithm options
        """

        # Set default options if none specified
        if opt is None:
            opt = CnstrMOD.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        # Call parent class __init__
        Nc = S.shape[0]
        # If Z not specified, get dictionary size from dsz
        if Z is None:
            Nm = dsz[0]
        else:
            Nm = Z.shape[0]

        xshape = (Nc, Nm)
        super(CnstrMOD, self).__init__(xshape, S.dtype, opt)

        self.S = np.asarray(S, dtype=self.dtype)

        # Create constraint set projection function
        self.Pcn = getPcn(opt['ZeroMean'], opt['NonNegCoef'])

        self.Y = self.X.copy()
        self.Yprv = self.Y.copy() + 1e3

        if Z is not None:
            self.setcoef(Z)



    def setcoef(self, Z):
        """Set coefficient array."""

        self.Z = np.asarray(Z, dtype=self.dtype)



    def getdict(self):
        """Get final coefficient array."""

        return self.X



    def grad_f(self, V=None):
        """Compute gradient of data fidelity for variable V or self.Y."""

        if V is None:
            V = self.Y

        # Compute (V Z - S) Z^T
        return (V.dot(self.Z) - self.S).dot(self.Z.T)



    def prox_g(self, V):
        """Compute proximal operator of :math:`g`."""

        return self.Pcn(V)



    def hessian_f(self, V):
        """Compute Hessian of :math:`f` applied to V."""

        hessfv = V.dot(self.Z).dot(self.Z.T)

        return hessfv



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_f()
        cns = self.obfn_cns()
        return (dfd, cns)



    def obfn_f(self, X=None):
        r"""Compute data fidelity term :math:`(1/2) \| D \mathbf{x} -
        \mathbf{s} \|_2^2`.
        """

        if X is None:
            X = self.X
        return 0.5 * np.linalg.norm((X.dot(self.Z) - self.S).ravel())**2



    def obfn_cns(self):
        r"""Compute constraint violation measure :math:`\| P(\mathbf{y}) -
        \mathbf{y}\|_2`.
        """

        return np.linalg.norm((self.Pcn(self.X) - self.X))



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            X = self.X
        return self.D.dot(self.X)





class WeightedCnstrMOD(CnstrMOD):
    r"""
    PGM algorithm for a weighted  variant of Constrained MOD (CMOD)
    problem.

    |

    .. inheritance-diagram:: WeightedCnstrMOD
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_D \| D X - S \|_W^2 + \iota_C(D)

    where :math:`\iota_C(\cdot)` is the indicator function of feasible
    set :math:`C` consisting of matrices with unit-norm columns and
    :math:`\| \cdot \|_W` denotes the weighted Frobenius norm defined as
    (note that this is *not* the standard definition)

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

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| D X - S
       \|_W^2`

       ``Cnstr`` : Constraint violation measure

       ``Rsdl`` : Residual

       ``L`` : Inverse of gradient step parameter

       ``Time`` : Cumulative run time
    """

    def __init__(self, Z, S, W=None, dsz=None, opt=None):
        """
        Parameters
        ----------
        Z : array_like, shape (M, K)
          Sparse representation coefficient matrix
        S : array_like, shape (N, K)
          Signal matrix
        W : array_like, shape (N, K)
          Weight matrix
        dsz : tuple
          Dictionary size
        opt : :class:`WeightedCnstrMOD.Options` object
          Algorithm options
        """

        super(WeightedCnstrMOD, self).__init__(Z, S, dsz, opt)

        if W is None:
            W = np.array([1.0], dtype=self.dtype)
        if W.ndim > 0:
            W = atleast_nd(2, W)
        self.W = np.asarray(W, dtype=self.dtype)



    def grad_f(self, V=None):
        """Compute gradient of data fidelity for variable V or self.Y."""

        if V is None:
            V = self.Y

        # Compute (W \odot (V Z - S)) Z^T
        return (self.W * (V.dot(self.Z) - self.S)).dot(self.Z.T)



    def obfn_f(self, X=None):
        r"""Compute data fidelity term :math:`(1/2) \| D \mathbf{x} -
        \mathbf{s} \|_W^2`.
        """

        if X is None:
            X = self.X
        return 0.5 * np.linalg.norm(
            (np.sqrt(self.W) * (X.dot(self.Z) - self.S)).ravel())**2





def getPcn(zm, nnc):
    """Construct constraint set projection function.

    Parameters
    ----------
    zm : bool
      Flag indicating whether the projection function should include
      column mean subtraction
    nnc : bool
      Flag indicating whether the projection function should include
      clipping to positive values

    Returns
    -------
    fn : function
      Constraint set projection function
    """

    if zm:
        if nnc:
            raise ValueError('Parameters zm and nnc may not both be True')
        else:
            return lambda x: normalise(zeromean(x))
    else:
        if nnc:
            return lambda x: normalise(np.clip(x, 0, None))
        else:
            return normalise



def zeromean(v):
    """Subtract mean of each column of matrix.

    Parameters
    ----------
    v : array_like
      Input dictionary array

    Returns
    -------
    vz : ndarray
      Dictionary array with column means subtracted
    """

    return v - np.mean(v, 0)



def normalise(v):
    """Normalise columns of matrix.

    Parameters
    ----------
    v : array_like
      Array with columns to be normalised

    Returns
    -------
    vnrm : ndarray
      Normalised array
    """

    vn = np.sqrt(np.sum(v**2, 0))
    vn[vn == 0] = 1.0
    return np.asarray(v / vn, dtype=v.dtype)
