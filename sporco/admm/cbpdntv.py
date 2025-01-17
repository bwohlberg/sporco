# -*- coding: utf-8 -*-
# Copyright (C) 2016-2025 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithms for convolutional sparse coding with
Total Variation regularisation terms"""

from __future__ import division, print_function
from builtins import range

import copy
import numpy as np

from sporco.admm import admm
import sporco.cnvrep as cr
from sporco.admm import cbpdn
import sporco.linalg as sl
import sporco.prox as sp
from sporco.util import u
from sporco.fft import (rfftn, irfftn, empty_aligned, rfftn_empty_aligned,
                        rfl2norm2)
from sporco.signal import gradient_filters


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class ConvBPDNScalarTV(admm.ADMM):
    r"""
    ADMM algorithm for an extension of Convolutional BPDN including
    terms penalising the total variation of each coefficient map
    :cite:`wohlberg-2017-convolutional`.

    |

    .. inheritance-diagram:: ConvBPDNScalarTV
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \; \frac{1}{2}
       \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
       \mu \sum_m \left\| \sqrt{\sum_i (G_i \mathbf{x}_m)^2} \right\|_1
       \;\;,

    where :math:`G_i` is an operator computing the derivative along index
    :math:`i`, via the ADMM problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \; (1/2) \left\| D \mathbf{x} -
       \mathbf{s} \right\|_2^2 + \lambda
       \| \mathbf{y}_L \|_1 + \mu \sum_m \left\| \sqrt{\sum_{i=0}^{L-1}
       \mathbf{y}_i^2} \right\|_1 \quad \text{ such that } \quad
       \left( \begin{array}{c} \Gamma_0 \\ \Gamma_1 \\ \vdots \\ I
       \end{array} \right) \mathbf{x} =
       \left( \begin{array}{c} \mathbf{y}_0 \\
       \mathbf{y}_1 \\ \vdots \\ \mathbf{y}_L \end{array}
       \right)  \;\;,

    where

    .. math::
       D = \left( \begin{array}{ccc} D_0 & D_1 & \ldots \end{array} \right)
       \qquad
       \mathbf{x} = \left( \begin{array}{c} \mathbf{x}_0 \\ \mathbf{x}_1 \\
       \vdots \end{array} \right) \qquad
       \Gamma_i = \left( \begin{array}{ccc}
          G_i & 0 & \ldots \\  0 & G_i & \ldots \\ \vdots & \vdots & \ddots
       \end{array} \right) \;\;.


    For multi-channel signals with a single-channel dictionary, scalar TV is
    applied independently to each coefficient map for channel :math:`c` and
    filter :math:`m`. Since multi-channel signals with a multi-channel
    dictionary also have one coefficient map per filter, the behaviour is
    the same as for single-channel signals.


    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term :math:`(1/2) \| \sum_m
       \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_m \|
       \mathbf{x}_m \|_1`

       ``RegTV`` : Value of regularisation term :math:`\sum_m \left\|
       \sqrt{\sum_i (G_i \mathbf{x}_m)^2} \right\|_1`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Relative residual of X step solver

       ``Time`` : Cumulative run time
    """


    class Options(cbpdn.ConvBPDN.Options):
        r"""ConvBPDNScalarTV algorithm options

        Options include all of those defined in
        :class:`.admm.cbpdn.ConvBPDN.Options`, together with additional
        options:

        ``TVWeight`` : An array of weights :math:`w_m` for the term
        penalising the gradient of the coefficient maps. If this
        option is defined, the regularization term is :math:`\sum_m w_m
        \left\| \sqrt{\sum_i (G_i \mathbf{x}_m)^2} \right\|_1`
        where :math:`w_m` is the weight for filter index :math:`m`. The
        array should be an :math:`M`-vector where :math:`M` is the number
        of filters in the dictionary.
        """

        defaults = copy.deepcopy(cbpdn.ConvBPDN.Options.defaults)
        defaults.update({'TVWeight' : 1.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDNScalarTV algorithm options
            """

            if opt is None:
                opt = {}
            cbpdn.ConvBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegTV')
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), u('RegTV'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid',
                     u('Regℓ1'): 'RegL1', u('RegTV'): 'RegTV'}


    def __init__(self, D, S, lmbda, mu=0.0, opt=None, dimK=None, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdnstv_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdnstv_init.svg

        |


        Parameters
        ----------
        D : array_like
          Dictionary matrix
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter (l1)
        mu : float
          Regularisation parameter (gradient)
        opt : :class:`ConvBPDNScalarTV.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        if opt is None:
            opt = ConvBPDNScalarTV.Options()

        # Infer problem dimensions and set relevant attributes of self
        self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Call parent class __init__
        Nx = np.prod(np.array(self.cri.shpX))
        yshape = self.cri.shpX + (len(self.cri.axisN)+1,)
        super(ConvBPDNScalarTV, self).__init__(Nx, yshape, yshape,
                                               S.dtype, opt)

        # Set l1 term scaling and weight array
        self.lmbda = self.dtype.type(lmbda)
        self.Wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)
        self.Wl1 = self.Wl1.reshape(cr.l1Wshape(self.Wl1, self.cri))

        self.mu = self.dtype.type(mu)
        if hasattr(opt['TVWeight'], 'ndim') and opt['TVWeight'].ndim > 0:
            self.Wtv = np.asarray(opt['TVWeight'].reshape(
                (1,)*(dimN + 2) + opt['TVWeight'].shape), dtype=self.dtype)
        else:
            # Wtv is a scalar: no need to change shape
            self.Wtv = np.asarray(opt['TVWeight'], dtype=self.dtype)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(50.0*self.lmbda + 1.0),
                      dtype=self.dtype)

        # Set rho_xi attribute
        self.set_attr('rho_xi', opt['AutoRho', 'RsdlTarget'], dval=1.0,
                      dtype=self.dtype)

        # Reshape D and S to standard layout
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # Compute signal in DFT domain
        self.Sf = rfftn(self.S, None, self.cri.axisN)

        self.Gf, GHGf = gradient_filters(self.cri.dimN+3, self.cri.axisN,
                                         self.cri.Nv, dtype=self.dtype)
        self.GHGf = self.Wtv**2 * GHGf

        # Initialise byte-aligned arrays for pyfftw
        self.YU = empty_aligned(self.Y.shape, dtype=self.dtype)
        self.Xf = rfftn_empty_aligned(self.cri.shpX, self.cri.axisN,
                                      self.dtype)

        self.setdict()



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = rfftn(self.D, self.cri.Nv, self.cri.axisN)
        # Compute D^H S
        self.DSf = np.conj(self.Df) * self.Sf
        if self.cri.Cd > 1:
            self.DSf = np.sum(self.DSf, axis=self.cri.axisC, keepdims=True)
        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbi_sm_c(
                self.Df, np.conj(self.Df), self.rho*self.GHGf + self.rho,
                self.cri.axisM)
        else:
            self.c = None



    def rhochange(self):
        """Updated cached c array when rho changes."""

        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbi_sm_c(
                self.Df, np.conj(self.Df), self.rho*self.GHGf + self.rho,
                self.cri.axisM)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`."""

        self.YU[:] = self.Y - self.U
        YUf = rfftn(self.YU, None, self.cri.axisN)

        # The sum is over the extra axis indexing spatial gradient
        # operators G_i, *not* over axisM
        b = self.DSf + self.rho*(YUf[..., -1] + self.Wtv * np.sum(
            np.conj(self.Gf) * YUf[..., 0:-1], axis=-1))

        if self.cri.Cd == 1:
            self.Xf[:] = sl.solvedbi_sm(
                self.Df, self.rho*self.GHGf + self.rho, b, self.c,
                self.cri.axisM)
        else:
            self.Xf[:] = sl.solvemdbi_ism(
                self.Df, self.rho*self.GHGf + self.rho, b, self.cri.axisM,
                self.cri.axisC)

        self.X = irfftn(self.Xf, self.cri.Nv, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            Dop = lambda x: sl.inner(self.Df, x, axis=self.cri.axisM)
            if self.cri.Cd == 1:
                DHop = lambda x: np.conj(self.Df) * x
            else:
                DHop = lambda x: sl.inner(np.conj(self.Df), x,
                                          axis=self.cri.axisC)
            ax = DHop(Dop(self.Xf)) + (self.rho*self.GHGf + self.rho)*self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`."""

        AXU = self.AX + self.U
        self.Y[..., 0:-1] = sp.prox_l2(AXU[..., 0:-1], self.mu/self.rho)
        self.Y[..., -1] = sp.prox_l1(AXU[..., -1],
                                     (self.lmbda/self.rho) * self.Wl1)



    def obfn_fvarf(self):
        """Variable to be evaluated in computing data fidelity term,
        depending on ``fEvalX`` option value.
        """

        return self.Xf if self.opt['fEvalX'] else \
            rfftn(self.Y[..., -1], None, self.cri.axisN)



    def var_y0(self):
        r"""Get :math:`\mathbf{y}_0` variable, consisting of all blocks of
        :math:`\mathbf{y}` corresponding to a gradient operator."""

        return self.Y[..., 0:-1]



    def var_y1(self):
        r"""Get :math:`\mathbf{y}_1` variable, the block of
        :math:`\mathbf{y}` corresponding to the identity operator."""

        return self.Y[..., -1:]



    def var_yx(self):
        r"""Get component block of :math:`\mathbf{y}` that is constrained
        to be equal to :math:`\mathbf{x}`."""

        return self.Y[..., -1]



    def var_yx_idx(self):
        r"""Get index expression for component block of :math:`\mathbf{y}`
        that is constrained to be equal to :math:`\mathbf{x}`.
        """

        return np.s_[..., -1]



    def getmin(self):
        """Get minimiser after optimisation."""

        return self.X if self.opt['ReturnX'] else self.var_y1()[..., 0]



    def getcoef(self):
        """Get final coefficient array."""

        return self.getmin()



    def obfn_g0var(self):
        """Variable to be evaluated in computing the TV regularisation
        term, depending on the ``gEvalY`` option value.
        """

        # Use of self.AXnr[..., 0:-1] instead of self.cnst_A0(None, self.Xf)
        # reduces number of calls to self.cnst_A0
        return self.var_y0() if self.opt['gEvalY'] else \
            self.AXnr[..., 0:-1]



    def obfn_g1var(self):
        r"""Variable to be evaluated in computing the :math:`\ell_1`
        regularisation term, depending on the ``gEvalY`` option value.
        """

        # Use of self.AXnr[...,-1:] instead of self.cnst_A1(self.X)
        # reduces number of calls to self.cnst_A1
        return self.var_y1() if self.opt['gEvalY'] else \
            self.AXnr[..., -1:]



    def obfn_gvar(self):
        """Method providing compatibility with the interface of
        :class:`.admm.cbpdn.ConvBPDN` and derived classes in order to make
        this class compatible with classes such as :class:`.AddMaskSim`.
        """

        return self.obfn_g1var()



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_dfd()
        reg = self.obfn_reg()
        obj = dfd + reg[0]
        return (obj, dfd) + reg[1:]



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| \sum_m \mathbf{d}_m *
        \mathbf{x}_m - \mathbf{s} \|_2^2`.
        """

        Ef = sl.inner(self.Df, self.obfn_fvarf(), axis=self.cri.axisM) \
             - self.Sf
        return rfl2norm2(Ef, self.S.shape, axis=self.cri.axisN)/2.0



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = np.linalg.norm((self.Wl1 * self.obfn_g1var()).ravel(), 1)
        rtv = np.sum(np.sqrt(np.sum(self.obfn_g0var()**2, axis=-1)))
        return (self.lmbda*rl1 + self.mu*rtv, rl1, rtv)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs,)



    def cnst_A0(self, X, Xf=None):
        r"""Compute :math:`A_0 \mathbf{x}` component of ADMM problem
        constraint. In this case :math:`A_0 \mathbf{x} = (\Gamma_0^T \;\;
        \Gamma_1^T \;\; \ldots )^T \mathbf{x}`.
        """

        if Xf is None:
            Xf = rfftn(X, axes=self.cri.axisN)
        return self.Wtv[..., np.newaxis] * irfftn(
            self.Gf * Xf[..., np.newaxis], self.cri.Nv, axes=self.cri.axisN)



    def cnst_A0T(self, X):
        r"""Compute :math:`A_0^T \mathbf{x}` where :math:`A_0 \mathbf{x}`
        is a component of the ADMM problem constraint. In this case
        :math:`A_0^T \mathbf{x} = (\Gamma_0^T \;\; \Gamma_1^T \;\; \ldots )
        \mathbf{x}`.
        """

        Xf = rfftn(X, axes=self.cri.axisN)
        return self.Wtv[..., np.newaxis] * irfftn(
            np.conj(self.Gf) * Xf[..., 0:-1], self.cri.Nv, axes=self.cri.axisN)



    def cnst_A1(self, X):
        r"""Compute :math:`A_1 \mathbf{x}` component of ADMM problem
        constraint. In this case :math:`A_1 \mathbf{x} = \mathbf{x}`.
        """

        return X[..., np.newaxis]



    def cnst_A1T(self, X):
        r"""Compute :math:`A_1^T \mathbf{x}` where :math:`A_1 \mathbf{x}`
        is a component of the ADMM problem constraint. In this case
        :math:`A_1^T \mathbf{x} = \mathbf{x}`.
        """

        return X[..., -1]



    def cnst_A(self, X, Xf=None):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint.  In this case :math:`A \mathbf{x} = (\Gamma_0^T \;\;
        \Gamma_1^T \;\; \ldots \;\; I)^T \mathbf{x}`.
        """

        return np.concatenate((self.cnst_A0(X, Xf),
                               self.cnst_A1(X)), axis=-1)



    def cnst_AT(self, X):
        r"""Compute :math:`A^T \mathbf{x}` where :math:`A \mathbf{x}` is
        a component of ADMM problem constraint. In this case
        :math:`A^T \mathbf{x} = (\Gamma_0^T \;\; \Gamma_1^T \;\; \ldots
        \;\; I) \mathbf{x}`.
        """

        return np.sum(self.cnst_A0T(X), axis=-1) + self.cnst_A1T(X)



    def cnst_B(self, Y):
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem constraint.
        In this case :math:`B \mathbf{y} = -\mathbf{y}`.
        """

        return -Y



    def cnst_c(self):
        r"""Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = \mathbf{0}`.
        """

        return 0.0



    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        # We need to keep the non-relaxed version of AX since it is
        # required for computation of primal residual r
        self.AXnr = self.cnst_A(self.X, self.Xf)
        if self.rlx == 1.0:
            # If RelaxParam option is 1.0 there is no relaxation
            self.AX = self.AXnr
        else:
            # Avoid calling cnst_c() more than once in case it is expensive
            # (e.g. due to allocation of a large block of memory)
            if not hasattr(self, '_cnst_c'):
                self._cnst_c = self.cnst_c()
            # Compute relaxed version of AX
            alpha = self.rlx
            self.AX = alpha*self.AXnr - (1-alpha)*(self.cnst_B(self.Y) -
                                                   self._cnst_c)



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            Xf = self.Xf
        else:
            Xf = rfftn(X, None, self.cri.axisN)
        Sf = np.sum(self.Df * Xf, axis=self.cri.axisM)
        return irfftn(Sf, self.cri.Nv, self.cri.axisN)





class ConvBPDNVectorTV(ConvBPDNScalarTV):
    r"""
    ADMM algorithm for an extension of Convolutional BPDN including
    a term penalising the vector total variation of the coefficient maps
    :cite:`wohlberg-2017-convolutional`.

    |

    .. inheritance-diagram:: ConvBPDNVectorTV
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \; \frac{1}{2}
       \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
       \mu \left\| \sqrt{\sum_m \sum_i (G_i \mathbf{x}_m)^2} \right\|_1
       \;\;,

    where :math:`G_i` is an operator computing the derivative along index
    :math:`i`, via the ADMM problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \; (1/2) \left\| D \mathbf{x} -
       \mathbf{s} \right\|_2^2 + \lambda
       \| \mathbf{y}_L \|_1 + \mu \left\| \sqrt{\sum_{i=0}^{L-1}
       I_B \mathbf{y}_i^2} \right\|_1 \quad \text{ such that } \quad
       \left( \begin{array}{c} \Gamma_0 \\ \Gamma_1 \\ \vdots \\ I
       \end{array} \right) \mathbf{x} =
       \left( \begin{array}{c} \mathbf{y}_0 \\
       \mathbf{y}_1 \\ \vdots \\ \mathbf{y}_L \end{array}
       \right)  \;\;,

    where

    .. math::
       D = \left( \begin{array}{ccc} D_0 & D_1 & \ldots \end{array} \right)
       \qquad
       \mathbf{x} = \left( \begin{array}{c} \mathbf{x}_0 \\ \mathbf{x}_1 \\
       \vdots \end{array} \right) \qquad
       \Gamma_i = \left( \begin{array}{ccc}
          G_i & 0 & \ldots \\  0 & G_i & \ldots \\ \vdots & \vdots & \ddots
       \end{array} \right) \qquad
       I_B = \left( \begin{array}{ccc} I & I & \ldots \end{array} \right)
       \;\;.


    For multi-channel signals with a single-channel dictionary, vector TV is
    applied jointly over the coefficient maps for channel :math:`c` and
    filter :math:`m`. Since multi-channel signals with a multi-channel
    dictionary also have one coefficient map per filter, the behaviour is
    the same as for single-channel signals.


    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term :math:`(1/2) \| \sum_m
       \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_m \|
       \mathbf{x}_m \|_1`

       ``RegTV`` : Value of regularisation term :math:`\left\|
       \sqrt{\sum_m \sum_i (G_i \mathbf{x}_m)^2} \right\|_1`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Relative residual of X step solver

       ``Time`` : Cumulative run time
    """


    def __init__(self, D, S, lmbda, mu=0.0, opt=None, dimK=None, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdnvtv_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdnvtv_init.svg

        |


        Parameters
        ----------
        D : array_like
          Dictionary matrix
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter (l1)
        mu : float
          Regularisation parameter (gradient)
        opt : :class:`ConvBPDNScalarTV.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        super(ConvBPDNVectorTV, self).__init__(D, S, lmbda, mu, opt,
                                               dimK, dimN)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`."""

        AXU = self.AX + self.U
        self.Y[..., 0:-1] = sp.prox_l2(AXU[..., 0:-1], self.mu/self.rho,
                                       axis=(self.cri.axisM, -1))
        self.Y[..., -1] = sp.prox_l1(AXU[..., -1],
                                     (self.lmbda/self.rho) * self.Wl1)



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = np.linalg.norm((self.Wl1 * self.obfn_g1var()).ravel(), 1)
        rtv = np.sum(np.sqrt(np.sum(self.obfn_g0var()**2,
                                    axis=(self.cri.axisM, -1))))
        return (self.lmbda*rl1 + self.mu*rtv, rl1, rtv)





class ConvBPDNRecTV(admm.ADMM):
    r"""
    ADMM algorithm for an extension of Convolutional BPDN including
    terms penalising the total variation of the reconstruction from the
    sparse representation :cite:`wohlberg-2017-convolutional`.

    |

    .. inheritance-diagram:: ConvBPDNRecTV
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \; \frac{1}{2}
       \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
       \mu \left\| \sqrt{\sum_i \left( G_i \left( \sum_m \mathbf{d}_m *
       \mathbf{x}_m  \right) \right)^2} \right\|_1 \;\;,

    where :math:`G_i` is an operator computing the derivative along index
    :math:`i`, via the ADMM problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \; (1/2) \left\| D
       \mathbf{x} - \mathbf{s} \right\|_2^2  +
       \lambda \| \mathbf{y}_0 \|_1 + \mu \left\|
       \sqrt{\sum_{i=1}^L \mathbf{y}_i^2} \right\|_1 \quad \text{ such that }
       \quad \left( \begin{array}{c} I \\ \Gamma_0 \\ \Gamma_1 \\ \vdots \\
       \Gamma_{L-1} \end{array} \right) \mathbf{x} =
       \left( \begin{array}{c} \mathbf{y}_0 \\
       \mathbf{y}_1 \\ \mathbf{y}_2 \\ \vdots \\ \mathbf{y}_L \end{array}
       \right)  \;\;,

    where

    .. math::
       D = \left( \begin{array}{ccc} D_0 & D_1 & \ldots \end{array} \right)
       \qquad
       \mathbf{x} = \left( \begin{array}{c} \mathbf{x}_0 \\ \mathbf{x}_1 \\
       \vdots \end{array} \right) \qquad
       \Gamma_i = \left( \begin{array}{ccc} G_{i,0} & G_{i,1} & \ldots
       \end{array} \right) \;\;,

    and linear operator :math:`G_{i,m}` is defined such that

    .. math::
       G_{i,m} \mathbf{x} = \mathbf{g}_i * \mathbf{d}_m * \mathbf{x}
       \;\;,

    where :math:`\mathbf{g}_i` is the filter corresponding to :math:`G_i`,
    i.e. :math:`G_i \mathbf{x} = \mathbf{g}_i * \mathbf{x}`.


    For multi-channel signals, vector TV is applied jointly over the
    reconstructions of all channels.


    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term :math:`(1/2) \| \sum_m
       \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_m \|
       \mathbf{x}_m \|_1`

       ``RegTV`` : Value of regularisation term :math:`\left\|
       \sqrt{\sum_i \left( G_i \left( \sum_m \mathbf{d}_m *
       \mathbf{x}_m  \right) \right)^2} \right\|_1`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Relative residual of X step solver

       ``Time`` : Cumulative run time
    """


    class Options(cbpdn.ConvBPDN.Options):
        r"""ConvBPDNRecTV algorithm options

        Options include all of those defined in
        :class:`.admm.cbpdn.ConvBPDN.Options`, together with additional
        options:

        ``TVWeight`` : An array of weights :math:`w_m` for the term
        penalising the gradient of the coefficient maps. If this
        option is defined, the regularization term is :math:`\left\|
        \sqrt{\sum_i \left( G_i \left( \sum_m w_m (\mathbf{d}_m *
        \mathbf{x}_m)  \right) \right)^2} \right\|_1` where :math:`w_m`
        is the weight for filter index :math:`m`. The array should be an
        :math:`M`-vector where :math:`M` is the number of filters in the
        dictionary.
        """

        defaults = copy.deepcopy(cbpdn.ConvBPDN.Options.defaults)
        defaults.update({'TVWeight' : 1.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDNRecTV algorithm options
            """

            if opt is None:
                opt = {}
            cbpdn.ConvBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegTV')
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), u('RegTV'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid',
                     u('Regℓ1'): 'RegL1', u('RegTV'): 'RegTV'}


    def __init__(self, D, S, lmbda, mu=0.0, opt=None, dimK=None, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdnrtv_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdnrtv_init.svg

        |


        Parameters
        ----------
        D : array_like
          Dictionary matrix
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter (l1)
        mu : float
          Regularisation parameter (gradient)
        opt : :class:`ConvBPDNRecTV.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        if opt is None:
            opt = ConvBPDNRecTV.Options()

        # Infer problem dimensions and set relevant attributes of self
        self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Call parent class __init__
        Nx = np.prod(np.array(self.cri.shpX))
        yshape = list(self.cri.shpX)
        yshape[self.cri.axisM] += len(self.cri.axisN) * self.cri.Cd
        super(ConvBPDNRecTV, self).__init__(Nx, yshape, yshape,
                                            S.dtype, opt)

        # Set l1 term scaling and weight array
        self.lmbda = self.dtype.type(lmbda)
        self.Wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)
        self.Wl1 = self.Wl1.reshape(cr.l1Wshape(self.Wl1, self.cri))

        self.mu = self.dtype.type(mu)
        if hasattr(opt['TVWeight'], 'ndim') and opt['TVWeight'].ndim > 0:
            self.Wtv = np.asarray(opt['TVWeight'].reshape(
                (1,)*(dimN + 2) + opt['TVWeight'].shape), dtype=self.dtype)
        else:
            # Wtv is a scalar: no need to change shape
            self.Wtv = self.dtype.type(opt['TVWeight'])

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(50.0*self.lmbda + 1.0),
                      dtype=self.dtype)

        # Set rho_xi attribute
        self.set_attr('rho_xi', opt['AutoRho', 'RsdlTarget'], dval=1.0,
                      dtype=self.dtype)

        # Reshape D and S to standard layout
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # Compute signal in DFT domain
        self.Sf = rfftn(self.S, None, self.cri.axisN)

        self.Gf, GHGf = gradient_filters(self.cri.dimN+3, self.cri.axisN,
                                         self.cri.Nv, dtype=self.dtype)

        # Initialise byte-aligned arrays for pyfftw
        self.YU = empty_aligned(self.Y.shape, dtype=self.dtype)
        self.Xf = rfftn_empty_aligned(self.cri.shpX, self.cri.axisN,
                                      self.dtype)

        self.setdict()



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = rfftn(self.D, self.cri.Nv, self.cri.axisN)

        self.GDf = self.Gf * (self.Wtv * self.Df)[..., np.newaxis]

        # Compute D^H S
        self.DSf = np.conj(self.Df) * self.Sf
        if self.cri.Cd > 1:
            self.DSf = np.sum(self.DSf, axis=self.cri.axisC, keepdims=True)



    def block_sep0(self, Y):
        """Separate variable into component corresponding to Y0 in Y."""

        return Y[..., 0:self.cri.M]



    def block_sep1(self, Y):
        """Separate variable into component corresponding to Y1 in Y."""

        Y1 = Y[..., self.cri.M:]

        # If cri.Cd > 1 (multi-channel dictionary), we need to undo the
        # reshape performed in block_cat
        if self.cri.Cd > 1:
            shp = list(Y1.shape)
            shp[self.cri.axisM] = self.cri.dimN
            shp[self.cri.axisC] = self.cri.Cd
            Y1 = Y1.reshape(shp)

        # Axes are swapped here for similar reasons to those
        # motivating swapping in cbpdn.ConvTwoBlockCnstrnt.block_sep0
        Y1 = np.swapaxes(Y1[..., np.newaxis], self.cri.axisM, -1)

        return Y1



    def block_cat(self, Y0, Y1):
        """Concatenate components corresponding to Y0 and Y1 blocks
        into Y.
        """

        # Axes are swapped here for similar reasons to those
        # motivating swapping in cbpdn.ConvTwoBlockCnstrnt.block_cat
        Y1sa = np.swapaxes(Y1, self.cri.axisM, -1)[..., 0]

        # If cri.Cd > 1 (multi-channel dictionary) Y0 has a singleton
        # channel axis but Y1 has a non-singleton channel axis. To make
        # it possible to concatenate Y0 and Y1, we reshape Y1 by a
        # partial ravel of axisM and axisC onto axisM.
        if self.cri.Cd > 1:
            shp = list(Y1sa.shape)
            shp[self.cri.axisM] *= shp[self.cri.axisC]
            shp[self.cri.axisC] = 1
            Y1sa = Y1sa.reshape(shp)

        return np.concatenate((Y0, Y1sa), axis=self.cri.axisM)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`."""

        self.YU[:] = self.Y - self.U
        YUf = rfftn(self.YU, None, self.cri.axisN)
        YUf0 = self.block_sep0(YUf)
        YUf1 = self.block_sep1(YUf)

        b = self.rho * np.sum(np.conj(self.GDf) * YUf1, axis=-1)
        if self.cri.Cd > 1:
            b = np.sum(b, axis=self.cri.axisC, keepdims=True)
        b += self.DSf + self.rho*YUf0

        # Concatenate multiple GDf components on axisC. For
        # single-channel signals, and multi-channel signals with a
        # single-channel dictionary, we end up with sl.solvemdbi_ism
        # solving a linear system of rank dimN+1 (corresponding to the
        # dictionary and a gradient operator per spatial dimension) plus
        # an identity. For multi-channel signals with a multi-channel
        # dictionary, we end up with sl.solvemdbi_ism solving a linear
        # system of rank C.d (dimN+1) (corresponding to the dictionary
        # and a gradient operator per spatial dimension for each
        # channel) plus an identity.

        # The structure of the linear system to be solved depends on the
        # number of channels in the signal and dictionary. Both branches are
        # the same in the single-channel signal case (the choice of handling
        # it via the 'else' branch is somewhat arbitrary).
        if self.cri.C > 1 and self.cri.Cd == 1:
            # Concatenate multiple GDf components on the final axis
            # of GDf (that indexes the number of gradient operators). For
            # multi-channel signals with a single-channel dictionary,
            # sl.solvemdbi_ism has to solve a linear system of rank dimN+1
            # (corresponding to the dictionary and a gradient operator per
            # spatial dimension)
            DfGDf = np.concatenate(
                [self.Df[..., np.newaxis],] +
                [np.sqrt(self.rho)*self.GDf[..., k, np.newaxis] for k
                 in range(self.GDf.shape[-1])], axis=-1)
            self.Xf[:] = sl.solvemdbi_ism(DfGDf, self.rho, b[..., np.newaxis],
                                          self.cri.axisM, -1)[..., 0]
        else:
            # Concatenate multiple GDf components on axisC. For multi-channel
            # signals with a multi-channel dictionary, sl.solvemdbi_ism has
            # to solve a linear system of rank C.d (dimN+1) (corresponding to
            # the dictionary and a gradient operator per spatial dimension
            # for each channel) plus an identity.
            DfGDf = np.concatenate(
                [self.Df,] + [np.sqrt(self.rho)*self.GDf[..., k] for k
                              in range(self.GDf.shape[-1])],
                axis=self.cri.axisC)
            self.Xf[:] = sl.solvemdbi_ism(DfGDf, self.rho, b, self.cri.axisM,
                                          self.cri.axisC)

        self.X = irfftn(self.Xf, self.cri.Nv, self.cri.axisN)

        if self.opt['LinSolveCheck']:
            if self.cri.C > 1 and self.cri.Cd == 1:
                Dop = lambda x: sl.inner(DfGDf, x[..., np.newaxis],
                                         axis=self.cri.axisM)
                DHop = lambda x: sl.inner(np.conj(DfGDf), x, axis=-1)
                ax = DHop(Dop(self.Xf))[..., 0] + self.rho*self.Xf
            else:
                Dop = lambda x: sl.inner(DfGDf, x, axis=self.cri.axisM)
                DHop = lambda x: sl.inner(np.conj(DfGDf), x,
                                          axis=self.cri.axisC)
                ax = DHop(Dop(self.Xf)) + self.rho*self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`."""

        AXU = self.AX + self.U
        self.block_sep0(self.Y)[:] = sp.prox_l1(
            self.block_sep0(AXU), (self.lmbda/self.rho) * self.Wl1)
        self.block_sep1(self.Y)[:] = sp.prox_l2(
            self.block_sep1(AXU), self.mu/self.rho, axis=(self.cri.axisC, -1))



    def obfn_fvarf(self):
        """Variable to be evaluated in computing data fidelity term,
        depending on ``fEvalX`` option value.
        """

        return self.Xf if self.opt['fEvalX'] else \
            rfftn(self.block_sep0(self.Y), None, self.cri.axisN)



    def var_y0(self):
        r"""Get :math:`\mathbf{y}_0` variable, the block of
        :math:`\mathbf{y}` corresponding to the identity operator."""

        return self.block_sep0(self.Y)



    def var_y1(self):
        r"""Get :math:`\mathbf{y}_1` variable, consisting of all blocks of
        :math:`\mathbf{y}` corresponding to a gradient operator."""

        return self.block_sep1(self.Y)



    def var_yx(self):
        r"""Get component block of :math:`\mathbf{y}` that is constrained to
        be equal to :math:`\mathbf{x}`"""

        return self.var_y0()



    def var_yx_idx(self):
        r"""Get index expression for component block of :math:`\mathbf{y}`
        that is constrained to be equal to :math:`\mathbf{x}`.
        """

        return np.s_[..., 0:self.cri.M]



    def getmin(self):
        """Get minimiser after optimisation."""

        return self.X if self.opt['ReturnX'] else self.var_y0()



    def getcoef(self):
        """Get final coefficient array."""

        return self.getmin()



    def obfn_g0var(self):
        """Variable to be evaluated in computing the TV regularisation
        term, depending on the ``gEvalY`` option value.
        """

        # Use of self.block_sep0(self.AXnr) instead of self.cnst_A0(self.X)
        # reduces number of calls to self.cnst_A0
        return self.var_y0() if self.opt['gEvalY'] else \
            self.block_sep0(self.AXnr)



    def obfn_g1var(self):
        r"""Variable to be evaluated in computing the :math:`\ell_1`
        regularisation term, depending on the ``gEvalY`` option value.
        """

        # Use of self.block_sep1(self.AXnr) instead of self.cnst_A1(self.X)
        # reduces number of calls to self.cnst_A0
        return self.var_y1() if self.opt['gEvalY'] else \
            self.block_sep1(self.AXnr)



    def obfn_gvar(self):
        """Method providing compatibility with the interface of
        :class:`.admm.cbpdn.ConvBPDN` and derived classes in order to make
        this class compatible with classes such as :class:`.AddMaskSim`.
        """

        return self.obfn_g1var()



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_dfd()
        reg = self.obfn_reg()
        obj = dfd + reg[0]
        return (obj, dfd) + reg[1:]



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| \sum_m \mathbf{d}_m *
        \mathbf{x}_m - \mathbf{s} \|_2^2`.
        """

        Ef = sl.inner(self.Df, self.obfn_fvarf(), axis=self.cri.axisM) \
             - self.Sf
        return rfl2norm2(Ef, self.S.shape, axis=self.cri.axisN)/2.0



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = np.linalg.norm((self.Wl1 * self.obfn_g0var()).ravel(), 1)
        rtv = np.sum(np.sqrt(np.sum(self.obfn_g1var()**2,
                                    axis=(self.cri.axisC, -1))))
        return (self.lmbda*rl1 + self.mu*rtv, rl1, rtv)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs,)


    def cnst_A0(self, X):
        r"""Compute :math:`A_0 \mathbf{x}` component of ADMM problem
        constraint. In this case :math:`A_0 \mathbf{x} = \mathbf{x}`.
        """

        return X



    def cnst_A0T(self, Y0):
        r"""Compute :math:`A_0^T \mathbf{y}_0` component of
        :math:`A^T \mathbf{y}`. In this case :math:`A_0^T \mathbf{y}_0 =
        \mathbf{y}_0`, i.e. :math:`A_0 = I`.
        """

        return Y0



    def cnst_A1(self, X, Xf=None):
        r"""Compute :math:`A_1 \mathbf{x}` component of ADMM problem
        constraint. In this case :math:`A_1 \mathbf{x} = (\Gamma_0^T \;\;
        \Gamma_1^T \;\; \ldots )^T \mathbf{x}`.
        """

        if Xf is None:
            Xf = rfftn(X, axes=self.cri.axisN)
        return irfftn(sl.inner(
            self.GDf, Xf[..., np.newaxis], axis=self.cri.axisM), self.cri.Nv,
                         self.cri.axisN)



    def cnst_A1T(self, Y1):
        r"""Compute :math:`A_1^T \mathbf{y}_1` component of
        :math:`A^T \mathbf{y}`. In this case :math:`A_1^T \mathbf{y}_1 =
        (\Gamma_0^T \;\; \Gamma_1^T \;\; \ldots) \mathbf{y}_1`.
        """

        Y1f = rfftn(Y1, None, axes=self.cri.axisN)
        return irfftn(np.conj(self.GDf) * Y1f, self.cri.Nv,
                      self.cri.axisN)



    def cnst_A(self, X, Xf=None):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint. In this case :math:`A \mathbf{x} = (I \;\; \Gamma_0^T
        \;\; \Gamma_1^T \;\; \ldots)^T \mathbf{x}`.
        """

        return self.block_cat(self.cnst_A0(X), self.cnst_A1(X, Xf))



    def cnst_AT(self, Y):
        r"""Compute :math:`A^T \mathbf{y}`. In this case
        :math:`A^T \mathbf{y} = (I \;\; \Gamma_0^T \;\; \Gamma_1^T \;\;
        \ldots) \mathbf{y}`.
        """

        return self.cnst_A0T(self.block_sep0(Y)) + \
            np.sum(self.cnst_A1T(self.block_sep1(Y)), axis=-1)



    def cnst_B(self, Y):
        r"""Compute :math:`B \mathbf{y}` component of ADMM problem
        constraint. In this case :math:`B \mathbf{y} = -\mathbf{y}`.
        """

        return -Y



    def cnst_c(self):
        r"""Compute constant component :math:`\mathbf{c}` of ADMM problem
        constraint. In this case :math:`\mathbf{c} = \mathbf{0}`.
        """

        return 0.0



    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        # We need to keep the non-relaxed version of AX since it is
        # required for computation of primal residual r
        self.AXnr = self.cnst_A(self.X, self.Xf)
        if self.rlx == 1.0:
            # If RelaxParam option is 1.0 there is no relaxation
            self.AX = self.AXnr
        else:
            # Avoid calling cnst_c() more than once in case it is expensive
            # (e.g. due to allocation of a large block of memory)
            if not hasattr(self, '_cnst_c'):
                self._cnst_c = self.cnst_c()
            # Compute relaxed version of AX
            alpha = self.rlx
            self.AX = alpha*self.AXnr - (1-alpha)*(self.cnst_B(self.Y) -
                                                   self._cnst_c)



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            Xf = self.Xf
        else:
            Xf = rfftn(X, None, self.cri.axisN)
        Sf = np.sum(self.Df * Xf, axis=self.cri.axisM)
        return irfftn(Sf, self.cri.Nv, self.cri.axisN)
