# -*- coding: utf-8 -*-
# Copyright (C) 2018-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithms for sparse coding with a product of
convolutional and standard dictionaries"""

from __future__ import division, absolute_import, print_function

import copy
import numpy as np

from sporco.admm import cbpdn
import sporco.cnvrep as cr
from sporco.util import u
from sporco.fft import rfftn, irfftn, empty_aligned, rfl2norm2
from sporco.linalg import (dot, inner, solvedbi_sm_c, solvedbi_sm,
                           solvedbd_sm_c, solvedbd_sm, rrs)
from sporco.prox import prox_l1, prox_sl1l2


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class ConvProdDictBPDN(cbpdn.ConvBPDN):
    r"""
    ADMM algorithm for the Convolutional BPDN (CBPDN) for multi-channel
    signals with a dictionary consisting of a product of convolutional
    and standard dictionaries :cite:`garcia-2018-convolutional2`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_X \; (1/2) \left\| D X B^T - S \right\|_2^2 +
       \lambda \| X \|_1

    where :math:`D` is a convolutional dictionary, :math:`B` is a
    standard dictionary, and :math:`S` is a multi-channel input image
    with

    .. math::
       S = \left( \begin{array}{ccc} \mathbf{s}_0 & \mathbf{s}_1 & \ldots
       \end{array} \right) \;.

    where the signal channels form the columns, :math:`\mathbf{s}_c`, of
    :math:`S`. This problem is solved via the ADMM problem
    :cite:`garcia-2018-convolutional2`

    .. math::
       \mathrm{argmin}_{X,Y} \;
       (1/2) \left\| D X B^T - S \right\|_2^2 + \lambda \| Y \|_1
       \quad \text{such that} \quad X = Y \;\;.
    """

    def __init__(self, D, B, S, lmbda, opt=None, dimK=None, dimN=2):
        """
        Parameters
        ----------
        D : array_like
          Convolutional dictionary array
        B : array_like
          Standard dictionary array
        S : array_like
          Signal array
        lmbda : float
          Regularisation parameter
        opt : :class:`ConvProdDictBPDN.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvProdDictBPDN.Options()

        # Since D operates on X B^T, the number of channels in X is equal
        # to the number of columns in B rather than the number of channels
        # in S. Here we initialise the object representing the problem
        # dimensions and correct the shape of X as required.
        self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)
        if self.cri.Cd > 1:
            raise ValueError('Only single-channel convolutional dictionaries'
                             ' are supported')
        shpX = list(self.cri.shpX)
        shpX[self.cri.axisC] = B.shape[1]
        self.cri.shpX = tuple(shpX)

        # Keep a record of the B dictionary
        self.set_dtype(opt, S.dtype)
        self.B = np.asarray(B, dtype=self.dtype)

        # Call parent constructor
        super(ConvProdDictBPDN, self).__init__(D, S, lmbda, opt, dimK, dimN)



    def setdict(self, D=None, B=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        if B is not None:
            self.B = np.asarray(B, dtype=self.dtype)

        if B is not None or not hasattr(self, 'Gamma'):
            self.Gamma, self.Q = np.linalg.eigh(self.B.T.dot(self.B))
            self.Gamma = np.abs(self.Gamma)

        if D is not None or not hasattr(self, 'Df'):
            self.Df = rfftn(self.D, self.cri.Nv, self.cri.axisN)
            self.DSf = np.conj(self.Df) * self.Sf
            self.DSfBQ = dot(self.B.dot(self.Q).T, self.DSf,
                             axis=self.cri.axisC)

        # Fold square root of Gamma into the dictionary array to enable
        # use of the solvedbi_sm solver
        shpg = [1] * len(self.cri.shpD)
        shpg[self.cri.axisC] = self.Gamma.shape[0]
        Gamma2 = np.sqrt(self.Gamma).reshape(shpg)
        self.gDf = Gamma2 * self.Df

        if self.opt['HighMemSolve']:
            self.c = solvedbi_sm_c(self.gDf, np.conj(self.gDf), self.rho,
                                   self.cri.axisM)
        else:
            self.c = None



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`."""

        self.YU[:] = self.Y - self.U
        Zf = rfftn(self.YU, None, self.cri.axisN)
        ZfQ = dot(self.Q.T, Zf, axis=self.cri.axisC)
        b = self.DSfBQ + self.rho * ZfQ

        Xh = solvedbi_sm(self.gDf, self.rho, b, self.c, axis=self.cri.axisM)
        self.Xf[:] = dot(self.Q, Xh, axis=self.cri.axisC)
        self.X = irfftn(self.Xf, self.cri.Nv, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            DDXf = np.conj(self.Df) *  inner(self.Df, self.Xf,
                                             axis=self.cri.axisM)
            DDXfBB = dot(self.B.T.dot(self.B), DDXf, axis=self.cri.axisC)
            ax = DDXfBB + self.rho * self.Xf
            b = dot(self.B.T, self.DSf, axis=self.cri.axisC) + \
                self.rho * Zf
            self.xrrs = rrs(ax, b)
        else:
            self.xrrs = None



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| D X B - S \|_2^2`.
        """

        DXBf = dot(self.B, inner(self.Df, self.obfn_fvarf(),
                                 axis=self.cri.axisM),
                   axis=self.cri.axisC)
        Ef = DXBf - self.Sf
        return rfl2norm2(Ef, self.S.shape, axis=self.cri.axisN) / 2.0



    def rhochange(self):
        """Updated cached c array when rho changes."""

        if self.opt['HighMemSolve']:
            self.c = solvedbi_sm_c(self.gDf, np.conj(self.gDf), self.rho,
                                   self.cri.axisM)



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            X = self.Y
        Xf = rfftn(X, None, self.cri.axisN)
        Sf = dot(self.B, np.sum(self.Df * Xf, axis=self.cri.axisM),
                    axis=self.cri.axisC)
        return irfftn(Sf, self.cri.Nv, self.cri.axisN)





class ConvProdDictBPDNJoint(ConvProdDictBPDN):
    r"""
    ADMM algorithm for the Convolutional BPDN (CBPDN) for multi-channel
    signals with a dictionary consisting of a product of convolutional
    and standard dictionaries, and with joint sparsity via an
    :math:`\ell_{2,1}` norm term :cite:`garcia-2018-convolutional2`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_X \; (1/2) \left\| D X B^T - S \right\|_2^2 +
       \lambda \| X \|_1 + \mu \| X \|_{2,1}

    where :math:`D` is a convolutional dictionary, :math:`B` is a
    standard dictionary, and :math:`S` is a multi-channel input image
    with

    .. math::
       S = \left( \begin{array}{ccc} \mathbf{s}_0 & \mathbf{s}_1 & \ldots
       \end{array} \right) \;.

    where the signal channels form the columns, :math:`\mathbf{s}_c`, of
    :math:`S`. This problem is solved via the ADMM problem
    :cite:`garcia-2018-convolutional2`

    .. math::
       \mathrm{argmin}_{X,Y} \;
       (1/2) \left\| D X B^T - S \right\|_2^2 + \lambda \| Y \|_1
       + \mu \| Y \|_{2,1} \quad \text{such that} \quad X = Y \;\;.
    """

    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegL21')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), u('Regℓ2,1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid',
                     u('Regℓ1'): 'RegL1', u('Regℓ2,1'): 'RegL21'}


    def __init__(self, D, B, S, lmbda, mu=0.0, opt=None, dimK=None,
                 dimN=2):
        """
        Parameters
        ----------
        D : array_like
          Convolutional dictionary array
        B : array_like
          Standard dictionary array
        S : array_like
          Signal array
        lmbda : float
          Regularisation parameter (l1)
        mu : float
          Regularisation parameter (l2,1)
        opt : :class:`ConvProdDictBPDNJoint.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        super(ConvProdDictBPDNJoint, self).__init__(D, B, S, lmbda, opt,
                                                    dimK, dimN)
        self.mu = self.dtype.type(mu)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y = prox_sl1l2(self.AX + self.U,
                            (self.lmbda / self.rho) * self.wl1,
                            (self.mu / self.rho), axis=self.cri.axisC)
        cbpdn.GenericConvBPDN.ystep(self)




    def obfn_reg(self):
        r"""Compute regularisation terms and contribution to objective
        function. Regularisation terms are :math:`\| Y \|_1` and
        :math:`\| Y \|_{2,1}`.
        """

        rl1 = np.linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rl21 = np.sum(np.sqrt(np.sum(self.obfn_gvar()**2,
                                     axis=self.cri.axisC)))
        return (self.lmbda*rl1 + self.mu*rl21, rl1, rl21)





class ConvProdDictL1L1Grd(cbpdn.ConvL1L1Grd):
    r"""
    ADMM algorithm for a Convolutional Sparse Coding problem for
    multi-channel signals with a dictionary consisting of a product
    of convolutional and standard dictionaries and with an :math:`\ell_1`
    data fidelity term and both :math:`\ell_1` and :math:`\ell_2` of
    gradient regularisation terms :cite:`garcia-2018-convolutional2`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_X \; \left\| D X B^T - S \right\|_1 +
       \lambda \| X \|_1 + (\mu / 2) \sum_i \| G_i X \|_2^2

    where :math:`D` is a convolutional dictionary, :math:`B` is a
    standard dictionary, :math:`G_i` is an operator that computes the
    gradient along array axis :math:`i`, and :math:`S` is a multi-channel
    input image with

    .. math::
       S = \left( \begin{array}{ccc} \mathbf{s}_0 & \mathbf{s}_1 & \ldots
       \end{array} \right) \;.

    where the signal channels form the columns, :math:`\mathbf{s}_c`, of
    :math:`S`. This problem is solved via the ADMM problem
    :cite:`garcia-2018-convolutional2`

    .. math::
       \mathrm{argmin}_{X,Y} \;
       \left\| Y_0 \right\|_1 + \lambda \| Y_1 \|_1 + (\mu / 2)
       \sum_i \| G_i X \|_2^2 \quad \text{such that} \quad
       Y_0 = D X B^T - S \;\;\; Y_1 = X \;\;.
    """

    def __init__(self, D, B, S, lmbda, mu, W=None, opt=None, dimK=0,
                 dimN=2):
        """
        Parameters
        ----------
        D : array_like
          Dictionary matrix
        B : array_like
          Standard dictionary array
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter (l1)
        mu : float
          Regularisation parameter (gradient)
        W : array_like
          Mask array. The array shape must be such that the array is
          compatible for multiplication with input array S (see
          :func:`.cnvrep.mskWshape` for more details).
        opt : :class:`ConvProdDictL1L1Grd.Options` object
          Algorithm options
        dimK : 0, 1, optional (default 0)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvProdDictL1L1Grd.Options()

        # Keep a record of the B dictionary
        self.set_dtype(opt, S.dtype)
        self.B = np.asarray(B, dtype=self.dtype)

        # S is an N x C matrix, D is an N x N M_D matrix, B is a C x M_B
        # matrix, and X is an N M x M_B matrix. The base class of this
        # class expects that X is N M x C (i.e. the same number of columns
        # as in S), so we pass its initialiser the product S B, which is
        # a N x M_B matrix, so that it initialises arrays with the correct
        # number of channels. This is the first of many nasty hacks in
        # this class!
        scidx = -2 if dimK == 1 else -1
        SB = dot(B.T, S, axis=scidx)
        super(ConvProdDictL1L1Grd, self).__init__(
            D, SB, lmbda, mu, W=W, opt=opt, dimK=dimK, dimN=dimN)

        # Ensure that the dictionary is single channel
        if self.cri.Cd > 1:
            raise ValueError('Only single-channel convolutional dictionaries'
                             ' are supported')

        # We need to correct the shape of S due to the modified S passed to
        # the base class initialiser
        shpS = list(self.cri.shpS)
        shpS[self.cri.axisC] = S.shape[self.cri.axisC]
        self.cri.shpS = tuple(shpS)
        self.S = np.asarray(S.reshape(shpS), dtype=self.dtype)

        # We also need to correct the shapes of a number of other working
        # arrays because we have to change the mechanism for combining
        # the Y0 and Y1 blocks into a single array. In the base class
        # these arrays can just be concatenated on an appropriate axis,
        # but this is not possible here due to the different array
        # shapes. The solution is that the composite array is one
        # dimensional, with the component blocks being extracted via
        # one dimensional slicing and then reshaped to the appropriate
        # shapes.
        self.y0shp = self.cri.shpS
        self.y1shp = self.cri.shpX
        self.y0I = int(np.prod(np.array(self.y0shp[self.cri.axisC:])))
        self.y1I = int(np.prod(np.array(self.y1shp[self.cri.axisC:])))
        self.yshp = self.cri.shpX[0:self.cri.axisC:] + (self.y0I + self.y1I,)
        self.Y = np.zeros(self.yshp, dtype=self.dtype)
        self.U = np.zeros(self.yshp, dtype=self.dtype)
        self.YU = empty_aligned(self.Y.shape, dtype=self.dtype)



    def block_sep0(self, Y):
        r"""Separate variable into component corresponding to
        :math:`\mathbf{y}_0` in :math:`\mathbf{y}\;\;`.
        """

        # This method is overridden because we have to change the
        # mechanism for combining the Y0 and Y1 blocks into a single
        # array (see comment in the __init__ method).
        shp = Y.shape[0:self.cri.axisC] + self.y0shp[self.cri.axisC:]
        return Y[(slice(None),)*self.cri.axisC +
                 (slice(0, self.y0I),)].reshape(shp)



    def block_sep1(self, Y):
        r"""Separate variable into component corresponding to
        :math:`\mathbf{y}_1` in :math:`\mathbf{y}\;\;`.
        """

        # This method is overridden because we have to change the
        # mechanism for combining the Y0 and Y1 blocks into a single
        # array (see comment in the __init__ method).
        shp = Y.shape[0:self.cri.axisC] + self.y1shp[self.cri.axisC:]
        return Y[(slice(None),)*self.cri.axisC +
                 (slice(self.y0I, None),)].reshape(shp)



    def block_cat(self, Y0, Y1):
        r"""Concatenate components corresponding to :math:`\mathbf{y}_0`
        and :math:`\mathbf{y}_1` to form :math:`\mathbf{y}\;\;`.
        """

        # This method is overridden because we have to change the
        # mechanism for combining the Y0 and Y1 blocks into a single
        # array (see comment in the __init__ method).
        y0shp = Y0.shape[0:self.cri.axisC] + (-1,)
        y1shp = Y1.shape[0:self.cri.axisC] + (-1,)
        return np.concatenate((Y0.reshape(y0shp),
                               Y1.reshape(y1shp)), axis=self.cri.axisC)



    def cnst_A0(self, X, Xf=None):
        r"""Compute :math:`A_0 \mathbf{x}` component of ADMM problem
        constraint.
        """

        if Xf is None:
            Xf = rfftn(X, None, self.cri.axisN)
        return irfftn(
            dot(self.B, inner(self.Df, Xf, axis=self.cri.axisM),
                   axis=self.cri.axisC), self.cri.Nv, self.cri.axisN)



    def cnst_A0T(self, Y0):
        r"""Compute :math:`A_0^T \mathbf{y}_0` component of
        :math:`A^T \mathbf{y}` (see :meth:`.ADMMTwoBlockCnstrnt.cnst_AT`).
        """

        # This calculation involves non-negligible computational cost. It
        # should be possible to disable relevant diagnostic information
        # (dual residual) to avoid this cost.
        Y0f = rfftn(Y0, None, self.cri.axisN)
        return irfftn(
            dot(self.B.T, np.conj(self.Df) * Y0f, axis=self.cri.axisC),
                   self.cri.Nv, self.cri.axisN)



    def setdict(self, D=None, B=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        if B is not None:
            self.B = np.asarray(B, dtype=self.dtype)

        if B is not None or not hasattr(self, 'Gamma'):
            self.Gamma, self.Q = np.linalg.eigh(self.B.T.dot(self.B))
            self.Gamma = np.abs(self.Gamma)

        if D is not None or not hasattr(self, 'Df'):
            self.Df = rfftn(self.D, self.cri.Nv, self.cri.axisN)

        # Fold square root of Gamma into the dictionary array to enable
        # use of the solvedbi_sm solver
        shpg = [1] * len(self.cri.shpD)
        shpg[self.cri.axisC] = self.Gamma.shape[0]
        Gamma2 = np.sqrt(self.Gamma).reshape(shpg)
        self.gDf = Gamma2 * self.Df

        if self.opt['HighMemSolve']:
            self.c = solvedbd_sm_c(
                self.gDf, np.conj(self.gDf),
                (self.mu / self.rho) * self.GHGf + 1.0, self.cri.axisM)
        else:
            self.c = None



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.YU[:] = self.Y - self.U
        self.block_sep0(self.YU)[:] += self.S
        Zf = rfftn(self.YU, None, self.cri.axisN)
        Z0f = self.block_sep0(Zf)
        Z1f = self.block_sep1(Zf)

        DZ0f = np.conj(self.Df) * Z0f
        DZ0fBQ = dot(self.B.dot(self.Q).T, DZ0f, axis=self.cri.axisC)
        Z1fQ = dot(self.Q.T, Z1f, axis=self.cri.axisC)
        b = DZ0fBQ + Z1fQ

        Xh = solvedbd_sm(self.gDf, (self.mu / self.rho) * self.GHGf + 1.0,
                         b, self.c, axis=self.cri.axisM)
        self.Xf[:] = dot(self.Q, Xh, axis=self.cri.axisC)
        self.X = irfftn(self.Xf, self.cri.Nv, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            DDXf = np.conj(self.Df) *  inner(self.Df, self.Xf,
                                                axis=self.cri.axisM)
            DDXfBB = dot(self.B.T.dot(self.B), DDXf, axis=self.cri.axisC)
            ax = self.rho * (DDXfBB + self.Xf) + \
                 self.mu * self.GHGf * self.Xf
            b = self.rho * (dot(self.B.T, DZ0f, axis=self.cri.axisC)
                            + Z1f)
            self.xrrs = rrs(ax, b)
        else:
            self.xrrs = None



    def rhochange(self):
        """Updated cached c array when rho changes."""

        if self.opt['HighMemSolve']:
            self.c = solvedbd_sm_c(
                self.gDf, np.conj(self.gDf),
                (self.mu / self.rho) * self.GHGf + 1.0, self.cri.axisM)



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            X = self.X
        Xf = rfftn(X, None, self.cri.axisN)
        Sf = dot(self.B, np.sum(self.Df * Xf, axis=self.cri.axisM),
                 axis=self.cri.axisC)
        return irfftn(Sf, self.cri.Nv, self.cri.axisN)



    # NB: It's not yet clear whether it's better to use rsdl_s and
    # rsdl_sn definitions below, or those inherited from cbpdn.ConvL1L1Grd
    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        return self.rho * self.cnst_AT(self.U)



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho * np.linalg.norm(U)





class ConvProdDictL1L1GrdJoint(ConvProdDictL1L1Grd):
    r"""
    ADMM algorithm for a Convolutional Sparse Coding problem for
    multi-channel signals with a dictionary consisting of a product
    of convolutional and standard dictionaries and with an :math:`\ell_1`
    data fidelity term and  :math:`\ell_{2,1}`, and :math:`\ell_2` of
    gradient regularisation terms :cite:`garcia-2018-convolutional2`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_X \; \left\| D X B^T - S \right\|_1 +
       \lambda \| X \|_{2,1} + (\mu / 2) \sum_i \| G_i X \|_2^2

    where :math:`D` is a convolutional dictionary, :math:`B` is a
    standard dictionary, :math:`G_i` is an operator that computes the
    gradient along array axis :math:`i`, and :math:`S` is a multi-channel
    input image with

    .. math::
       S = \left( \begin{array}{ccc} \mathbf{s}_0 & \mathbf{s}_1 & \ldots
       \end{array} \right) \;.

    where the signal channels form the columns, :math:`\mathbf{s}_c`, of
    :math:`S`. This problem is solved via the ADMM problem
    :cite:`garcia-2018-convolutional2`

    .. math::
       \mathrm{argmin}_{X,Y} \;
       \left\| Y_0 \right\|_1 + \lambda \| Y_1 \|_{2,1} + (\mu / 2)
       \sum_i \| G_i X \|_2^2 \quad \text{such that} \quad
       Y_0 = D X B^T - S \;\;\; Y_1 = X \;\;.
    """

    class Options(ConvProdDictL1L1Grd.Options):
        r"""ConvBPDNJoint algorithm options

        Options include all of those defined in
        :class:`ConvProdDictL1L1Grd.Options`, together with additional
        options:

          ``L21Weight`` : An array of weights for the :math:`\ell_{2,1}`
          norm. The array shape must be such that the array is
          compatible for multiplication with the X/Y variables *after*
          the sum over ``axisC`` performed during the computation of the
          :math:`\ell_{2,1}` norm. If this option is defined, the
          regularization term is :math:`\mu \sum_i w_i \sqrt{ \sum_c
          \mathbf{x}_{i,c}^2 }` where :math:`w_i` are the elements of the
          weight array, subscript :math:`c` indexes the channel axis and
          subscript :math:`i` indexes all other axes.
        """

        defaults = copy.deepcopy(ConvProdDictL1L1Grd.Options.defaults)
        defaults.update({'L21Weight': 1.0})



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL21', 'RegGrad')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ21'), u('Regℓ2∇'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid',
                     u('Regℓ21'): 'RegL21', u('Regℓ2∇'): 'RegGrad'}


    def __init__(self, D, B, S, lmbda, mu=0.0, opt=None, dimK=None,
                 dimN=2):
        """
        Parameters
        ----------
        D : array_like
          Dictionary matrix
        B : array_like
          Standard dictionary array
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter (l2,1)
        mu : float
          Regularisation parameter (gradient)
        W : array_like
          Mask array. The array shape must be such that the array is
          compatible for multiplication with input array S (see
          :func:`.cnvrep.mskWshape` for more details).
        opt : :class:`ConvProdDictL1L1GrdJoint.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        super(ConvProdDictL1L1GrdJoint, self).__init__(D, B, S, lmbda, mu,
                                            opt=opt, dimK=dimK, dimN=dimN)
        self.wl21 = np.asarray(opt['L21Weight'], dtype=self.dtype)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        AXU = self.AX + self.U
        Y0 = prox_l1(self.block_sep0(AXU) - self.S, (1.0/self.rho)*self.W)
        Y1 = prox_sl1l2(self.block_sep1(AXU), 0.0,
                        (self.lmbda/self.rho)*self.wl21,
                        axis=self.cri.axisC)
        self.Y = self.block_cat(Y0, Y1)
        cbpdn.ConvTwoBlockCnstrnt.ystep(self)



    def obfn_g1(self, Y1):
        r"""Compute :math:`g_1(\mathbf{y_1})` component of ADMM objective
        function.
        """

        return np.sum(self.wl21 * np.sqrt(np.sum(Y1**2, axis=self.cri.axisC)))
