# -*- coding: utf-8 -*-
# Copyright (C) 2015-2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""ADMM algorithms for the Convolutional Constrained MOD problem with
Mask Decoupling"""

from __future__ import division, absolute_import

import copy
import numpy as np

from sporco.admm import admm
from sporco.admm import ccmod
import sporco.cnvrep as cr
import sporco.linalg as sl
from sporco.common import _fix_dynamic_class_lookup
from sporco.fft import rfftn, irfftn, empty_aligned, rfftn_empty_aligned


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class ConvCnstrMODMaskDcplBase(admm.ADMMTwoBlockCnstrnt):
    r"""
    Base class for ADMM algorithms for Convolutional Constrained MOD
    with Mask Decoupling :cite:`heide-2015-fast`.

    |

    .. inheritance-diagram:: ConvCnstrMODMaskDcplBase
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \left\|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s}\right) \right\|_2^2 \quad \text{such that} \quad
       \mathbf{d}_m \in C \;\; \forall m

    where :math:`C` is the feasible set consisting of filters with unit
    norm and constrained support, and :math:`W` is a mask array, via the
    ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{d},\mathbf{g}_0,\mathbf{g}_1} \;
       (1/2) \| W \mathbf{g}_0 \|_2^2 + \iota_C(\mathbf{g}_1)
       \;\text{such that}\;
       \left( \begin{array}{c} X \\ I \end{array} \right) \mathbf{d}
       - \left( \begin{array}{c} \mathbf{g}_0 \\ \mathbf{g}_1 \end{array}
       \right) = \left( \begin{array}{c} \mathbf{s} \\
       \mathbf{0} \end{array} \right) \;\;,

    where  :math:`\iota_C(\cdot)` is the indicator function of feasible
    set :math:`C`, and :math:`X \mathbf{d} = \sum_m \mathbf{x}_m *
    \mathbf{d}_m`.

   |

    The implementation of this class is substantially complicated by the
    support of multi-channel signals. In the following, the number of
    channels in the signal and dictionary are denoted by ``C`` and ``Cd``
    respectively, the number of signals and the number of filters are
    denoted by ``K`` and ``M`` respectively, ``X``, ``Z``, and ``S`` denote
    the dictionary, coefficient map, and signal arrays respectively, and
    ``Y0`` and ``Y1`` denote blocks 0 and 1 of the auxiliary (split)
    variable of the ADMM problem. We need to consider three different cases:

      1. Single channel signal and dictionary (``C`` = ``Cd`` = 1)
      2. Multi-channel signal, single channel dictionary (``C`` > 1,
         ``Cd`` = 1)
      3. Multi-channel signal and dictionary (``C`` = ``Cd`` > 1)


    The final three (non-spatial) dimensions of the main variables in each
    of these cases are as in the following table:

      ======   ==================   =====================   ==================
      Var.     ``C`` = ``Cd`` = 1   ``C`` > 1, ``Cd`` = 1   ``C`` = ``Cd`` > 1
      ======   ==================   =====================   ==================
      ``X``    1 x 1 x ``M``        1 x 1 x ``M``           ``Cd`` x 1 x ``M``
      ``Z``    1 x ``K`` x ``M``    ``C`` x ``K`` x ``M``   1 x ``K`` x ``M``
      ``S``    1 x ``K`` x 1        ``C`` x ``K`` x 1       ``C`` x ``K`` x 1
      ``Y0``   1 x ``K`` x 1        ``C`` x ``K`` x 1       ``C`` x ``K`` x 1
      ``Y1``   1 x 1 x ``M``        1 x 1 x ``M``           ``C`` x 1 x ``M``
      ======   ==================   =====================   ==================

    In order to combine the block components ``Y0`` and ``Y1`` of
    variable ``Y`` into a single array, we need to be able to
    concatenate the two component arrays on one of the axes, but the shapes
    ``Y0`` and ``Y1`` are not compatible for concatenation. The solution for
    cases 1. and 3. is to swap the ``K`` and ``M`` axes of `Y0`` before
    concatenating, as well as after extracting the ``Y0`` component from the
    concatenated ``Y`` variable. In case 2., since the ``C`` and ``K``
    indices have the same behaviour in the dictionary update equation, we
    combine these axes in :meth:`.__init__`, so that the case 2. array
    shapes become

      ======      =====================
      Var.        ``C`` > 1, ``Cd`` = 1
      ======      =====================
      ``X``       1 x 1 x ``M``
      ``Z``       1 x ``C`` ``K`` x ``M``
      ``S``       1 x ``C`` ``K`` x 1
      ``Y0``      1 x ``C`` ``K`` x 1
      ``Y1``      1 x 1 x ``M``
      ======      =====================

    making it possible to concatenate ``Y0`` and ``Y1`` using the same
    axis swapping strategy as in the other cases. See :meth:`.block_sep0`
    and :meth:`block_cat` for additional details.

    |

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``DFid`` : Value of data fidelity term :math:`(1/2) \sum_k \|
       W (\sum_m \mathbf{d}_m * \mathbf{x}_{k,m} - \mathbf{s}_k) \|_2^2`

       ``Cnstr`` : Constraint violation measure

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


    class Options(admm.ADMMTwoBlockCnstrnt.Options):
        r"""ConvCnstrMODMaskDcplBase algorithm options

        Options include all of those defined in
        :class:`.ADMMTwoBlockCnstrnt.Options`, together with
        additional options:

          ``LinSolveCheck`` : Flag indicating whether to compute
          relative residual of X step solver.

          ``ZeroMean`` : Flag indicating whether the solution
          dictionary :math:`\{\mathbf{d}_m\}` should have zero-mean
          components.
        """

        defaults = copy.deepcopy(admm.ADMMEqual.Options.defaults)
        defaults.update({'AuxVarObj': False, 'fEvalX': True,
                         'gEvalY': False, 'LinSolveCheck': False,
                         'ZeroMean': False, 'RelaxParam': 1.8,
                         'rho': 1.0, 'ReturnVar': 'Y1'})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvCnstrMODMaskDcpl algorithm options
            """

            if opt is None:
                opt = {}
            admm.ADMMTwoBlockCnstrnt.Options.__init__(self, opt)



        def __setitem__(self, key, value):
            """Set options 'fEvalX' and 'gEvalY' appropriately when option
            'AuxVarObj' is set.
            """

            admm.ADMMTwoBlockCnstrnt.Options.__setitem__(self, key, value)

            if key == 'AuxVarObj':
                if value is True:
                    self['fEvalX'] = False
                    self['gEvalY'] = True
                else:
                    self['fEvalX'] = True
                    self['gEvalY'] = False



    itstat_fields_objfn = ('DFid', 'Cnstr')
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('DFid', 'Cnstr')
    hdrval_objfun = {'DFid': 'DFid', 'Cnstr': 'Cnstr'}



    def __init__(self, Z, S, W, dsz, opt=None, dimK=None, dimN=2):
        """
        Parameters
        ----------
        Z : array_like
          Coefficient map array
        S : array_like
          Signal array
        W : array_like
          Mask array. The array shape must be such that the array is
          compatible for multiplication with the *internal* shape of
          input array S (see :class:`.cnvrep.CDU_ConvRepIndexing` for a
          discussion of the distinction between *external* and *internal*
          data layouts) after reshaping to the shape determined by
          :func:`.cnvrep.mskWshape`.
        dsz : tuple
          Filter support size(s)
        opt : :class:`ConvCnstrMODMaskDcplBase.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvCnstrMODMaskDcplBase.Options()

        # Infer problem dimensions and set relevant attributes of self
        self.cri = cr.CDU_ConvRepIndexing(dsz, S, dimK=dimK, dimN=dimN)

        # Convert W to internal shape
        W = np.asarray(W.reshape(cr.mskWshape(W, self.cri)),
                       dtype=S.dtype)

        # Reshape W if necessary (see discussion of reshape of S below)
        if self.cri.Cd == 1 and self.cri.C > 1:
            # In most cases broadcasting rules make it possible for W
            # to have a singleton dimension corresponding to a non-singleton
            # dimension in S. However, when S is reshaped to interleave axisC
            # and axisK on the same axis, broadcasting is no longer sufficient
            # unless axisC and axisK of W are either both singleton or both
            # of the same size as the corresponding axes of S. If neither of
            # these cases holds, it is necessary to replicate the axis of W
            # (axisC or axisK) that does not have the same size as the
            # corresponding axis of S.
            shpw = list(W.shape)
            swck = shpw[self.cri.axisC] * shpw[self.cri.axisK]
            if swck > 1 and swck < self.cri.C * self.cri.K:
                if W.shape[self.cri.axisK] == 1 and self.cri.K > 1:
                    shpw[self.cri.axisK] = self.cri.K
                else:
                    shpw[self.cri.axisC] = self.cri.C
                W = np.broadcast_to(W, shpw)
            self.W = W.reshape(
                W.shape[0:self.cri.dimN] +
                (1, W.shape[self.cri.axisC] * W.shape[self.cri.axisK], 1))
        else:
            self.W = W

        # Call parent class __init__
        Nx = self.cri.N * self.cri.Cd * self.cri.M
        CK = (self.cri.C if self.cri.Cd == 1 else 1) * self.cri.K
        shpY = list(self.cri.shpX)
        shpY[self.cri.axisC] = self.cri.Cd
        shpY[self.cri.axisK] = 1
        shpY[self.cri.axisM] += CK
        super(ConvCnstrMODMaskDcplBase, self).__init__(
            Nx, shpY, self.cri.axisM, CK, S.dtype, opt)

        # Reshape S to standard layout (Z, i.e. X in cbpdn, is assumed
        # to be taken from cbpdn, and therefore already in standard
        # form). If the dictionary has a single channel but the input
        # (and therefore also the coefficient map array) has multiple
        # channels, the channel index and multiple image index have
        # the same behaviour in the dictionary update equation: the
        # simplest way to handle this is to just reshape so that the
        # channels also appear on the multiple image index.
        if self.cri.Cd == 1 and self.cri.C > 1:
            self.S = S.reshape(self.cri.Nv + (1, self.cri.C*self.cri.K, 1))
        else:
            self.S = S.reshape(self.cri.shpS)
        self.S = np.asarray(self.S, dtype=self.dtype)

        # Create constraint set projection function
        self.Pcn = cr.getPcn(dsz, self.cri.Nv, self.cri.dimN, self.cri.dimCd,
                             zm=opt['ZeroMean'])

        # Initialise byte-aligned arrays for pyfftw
        self.YU = empty_aligned(self.Y.shape, dtype=self.dtype)
        xfshp = list(self.cri.Nv + (self.cri.Cd, 1, self.cri.M))
        self.Xf = rfftn_empty_aligned(xfshp, self.cri.axisN,
                                      self.dtype)

        if Z is not None:
            self.setcoef(Z)



    def uinit(self, ushape):
        """Return initialiser for working variable U."""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            Ub0 = (self.W**2) * self.block_sep0(self.Y) / self.rho
            Ub1 = self.block_sep1(self.Y)
            return self.block_cat(Ub0, Ub1)



    def setcoef(self, Z):
        """Set coefficient array."""

        # If the dictionary has a single channel but the input (and
        # therefore also the coefficient map array) has multiple
        # channels, the channel index and multiple image index have
        # the same behaviour in the dictionary update equation: the
        # simplest way to handle this is to just reshape so that the
        # channels also appear on the multiple image index.
        if self.cri.Cd == 1 and self.cri.C > 1:
            Z = Z.reshape(self.cri.Nv + (1, self.cri.Cx*self.cri.K,
                                         self.cri.M,))
        self.Z = np.asarray(Z, dtype=self.dtype)

        self.Zf = rfftn(self.Z, self.cri.Nv, self.cri.axisN)



    def getdict(self, crop=True):
        """Get final dictionary. If ``crop`` is ``True``, apply
        :func:`.cnvrep.bcrop` to returned array.
        """

        D = self.block_sep1(self.Y)
        if crop:
            D = cr.bcrop(D, self.cri.dsz, self.cri.dimN)
        return D



    def xstep_check(self, b):
        r"""Check the minimisation of the Augmented Lagrangian with
        respect to :math:`\mathbf{x}` by method `xstep` defined in
        derived classes. This method should be called at the end of any
        `xstep` method.
        """

        if self.opt['LinSolveCheck']:
            Zop = lambda x: sl.inner(self.Zf, x, axis=self.cri.axisM)
            ZHop = lambda x: sl.inner(np.conj(self.Zf), x,
                                      axis=self.cri.axisK)
            ax = ZHop(Zop(self.Xf)) + self.Xf
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        AXU = self.AX + self.U
        Y0 = (self.rho*(self.block_sep0(AXU) - self.S)) / (self.W**2 +
                                                           self.rho)
        Y1 = self.Pcn(self.block_sep1(AXU))
        self.Y = self.block_cat(Y0, Y1)



    def relax_AX(self):
        """Implement relaxation if option ``RelaxParam`` != 1.0."""

        self.AXnr = self.cnst_A(self.X, self.Xf)
        if self.rlx == 1.0:
            self.AX = self.AXnr
        else:
            alpha = self.rlx
            self.AX = alpha*self.AXnr + (1-alpha)*self.block_cat(
                self.var_y0() + self.S, self.var_y1())



    def block_sep0(self, Y):
        r"""Separate variable into component corresponding to
        :math:`\mathbf{y}_0` in :math:`\mathbf{y}\;\;`. The method from
        parent class :class:`.ADMMTwoBlockCnstrnt` is overridden here to
        allow swapping of K (multi-image) and M (filter) axes in block 0
        so that it can be concatenated on axis M with block 1. This is
        necessary because block 0 has the dimensions of S while block 1
        has the dimensions of D. Handling of multi-channel signals
        substantially complicate this issue. There are two multi-channel
        cases: multi-channel dictionary and signal (Cd = C > 1), and
        single-channel dictionary with multi-channel signal (Cd = 1, C >
        1). In the former case, S and D shapes are (N x C x K x 1) and
        (N x C x 1 x M) respectively. In the latter case,
        :meth:`.__init__` has already taken care of combining C
        (multi-channel) and K (multi-image) axes in S, so the S and D
        shapes are (N x 1 x C K x 1) and (N x 1 x 1 x M) respectively.
        """

        return np.swapaxes(
            Y[(slice(None),)*self.blkaxis + (slice(0, self.blkidx),)],
            self.cri.axisK, self.cri.axisM)



    def block_cat(self, Y0, Y1):
        r"""Concatenate components corresponding to :math:`\mathbf{y}_0`
        and :math:`\mathbf{y}_1` to form :math:`\mathbf{y}\;\;`. The
        method from parent class :class:`.ADMMTwoBlockCnstrnt` is
        overridden here to allow swapping of K (multi-image) and M
        (filter) axes in block 0 so that it can be concatenated on axis
        M with block 1. This is necessary because block 0 has the
        dimensions of S while block 1 has the dimensions of D. Handling
        of multi-channel signals substantially complicate this
        issue. There are two multi-channel cases: multi-channel
        dictionary and signal (Cd = C > 1), and single-channel
        dictionary with multi-channel signal (Cd = 1, C > 1). In the
        former case, S and D shapes are (N x C x K x 1) and (N x C x 1 x
        M) respectively. In the latter case, :meth:`.__init__` has
        already taken care of combining C (multi-channel) and K
        (multi-image) axes in S, so the S and D shapes are (N x 1 x C K
        x 1) and (N x 1 x 1 x M) respectively.
        """

        return np.concatenate((np.swapaxes(Y0, self.cri.axisK,
                                           self.cri.axisM), Y1),
                              axis=self.blkaxis)



    def cnst_A(self, X, Xf=None):
        r"""Compute :math:`A \mathbf{x}` component of ADMM problem
        constraint.
        """

        return self.block_cat(self.cnst_A0(X, Xf), self.cnst_A1(X))



    def obfn_g0var(self):
        """Variable to be evaluated in computing
        :meth:`.ADMMTwoBlockCnstrnt.obfn_g0`, depending on the ``AuxVarObj``
        option value.
        """

        return self.var_y0() if self.opt['AuxVarObj'] else \
            self.cnst_A0(None, self.Xf) - self.cnst_c0()



    def cnst_A0(self, X, Xf=None):
        r"""Compute :math:`A_0 \mathbf{x}` component of ADMM problem
        constraint.
        """

        # This calculation involves non-negligible computational cost
        # when Xf is None (i.e. the function is not being applied to
        # self.X).
        if Xf is None:
            Xf = rfftn(X, None, self.cri.axisN)
        return irfftn(sl.inner(self.Zf, Xf, axis=self.cri.axisM),
                         self.cri.Nv, self.cri.axisN)



    def cnst_A0T(self, Y0):
        r"""Compute :math:`A_0^T \mathbf{y}_0` component of
        :math:`A^T \mathbf{y}` (see :meth:`.ADMMTwoBlockCnstrnt.cnst_AT`).
        """

        # This calculation involves non-negligible computational cost. It
        # should be possible to disable relevant diagnostic information
        # (dual residual) to avoid this cost.
        Y0f = rfftn(Y0, None, self.cri.axisN)
        return irfftn(sl.inner(np.conj(self.Zf), Y0f,
                                  axis=self.cri.axisK), self.cri.Nv,
                         self.cri.axisN)



    def cnst_c0(self):
        r"""Compute constant component :math:`\mathbf{c}_0` of
        :math:`\mathbf{c}` in the ADMM problem constraint.
        """

        return self.S



    def eval_objfn(self):
        """Compute components of regularisation function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_g0(self.obfn_g0var())
        cns = self.obfn_g1(self.obfn_g1var())
        return (dfd, cns)



    def obfn_g0(self, Y0):
        r"""Compute :math:`g_0(\mathbf{y}_0)` component of ADMM objective
        function.
        """

        return (np.linalg.norm(self.W * Y0)**2) / 2.0



    def obfn_g1(self, Y1):
        r"""Compute :math:`g_1(\mathbf{y_1})` component of ADMM objective
        function.
        """

        return np.linalg.norm((self.Pcn(Y1) - Y1))



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs,)



    def reconstruct(self, D=None):
        """Reconstruct representation."""

        if D is None:
            Df = self.Xf
        else:
            Df = rfftn(D, None, self.cri.axisN)

        Sf = np.sum(self.Zf * Df, axis=self.cri.axisM)
        return irfftn(Sf, self.cri.Nv, self.cri.axisN)



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        return self.rho*np.linalg.norm(self.cnst_AT(self.U))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho*np.linalg.norm(U)





class ConvCnstrMODMaskDcpl_IterSM(ConvCnstrMODMaskDcplBase):
    r"""
    ADMM algorithm for Convolutional Constrained MOD with Mask Decoupling
    :cite:`heide-2015-fast` with the :math:`\mathbf{x}` step solved via
    iterated application of the Sherman-Morrison equation
    :cite:`wohlberg-2016-efficient`.

    |

    .. inheritance-diagram:: ConvCnstrMODMaskDcpl_IterSM
       :parts: 2

    |

    Multi-channel signals/images are supported
    :cite:`wohlberg-2016-convolutional`. See
    :class:`.ConvCnstrMODMaskDcplBase` for interface details.
    """


    class Options(ConvCnstrMODMaskDcplBase.Options):
        """ConvCnstrMODMaskDcpl_IterSM algorithm options

        Options are the same as those defined in
        :class:`.ConvCnstrMODMaskDcplBase.Options`.
        """

        defaults = copy.deepcopy(ConvCnstrMODMaskDcplBase.Options.defaults)


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvCnstrMODMaskDcpl_IterSM algorithm options
            """

            if opt is None:
                opt = {}
            ConvCnstrMODMaskDcplBase.Options.__init__(self, opt)



    def __init__(self, Z, S, W, dsz, opt=None, dimK=1, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/ccmodmdism_init.svg
           :width: 20%
           :target: ../_static/jonga/ccmodmdism_init.svg
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvCnstrMODMaskDcpl_IterSM.Options()

        super(ConvCnstrMODMaskDcpl_IterSM, self).__init__(Z, S, W, dsz,
                                                          opt, dimK, dimN)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.YU[:] = self.Y - self.U
        self.block_sep0(self.YU)[:] += self.S
        YUf = rfftn(self.YU, None, self.cri.axisN)
        b = sl.inner(np.conj(self.Zf), self.block_sep0(YUf),
                     axis=self.cri.axisK) + self.block_sep1(YUf)

        self.Xf[:] = sl.solvemdbi_ism(self.Zf, 1.0, b, self.cri.axisM,
                                      self.cri.axisK)
        self.X = irfftn(self.Xf, self.cri.Nv, self.cri.axisN)
        self.xstep_check(b)





class ConvCnstrMODMaskDcpl_CG(ConvCnstrMODMaskDcplBase):
    r"""
    ADMM algorithm for Convolutional Constrained MOD with Mask Decoupling
    :cite:`heide-2015-fast` with the :math:`\mathbf{x}` step solved via
    Conjugate Gradient (CG) :cite:`wohlberg-2016-efficient`.

    |

    .. inheritance-diagram:: ConvCnstrMODMaskDcpl_CG
       :parts: 2

    |

    Multi-channel signals/images are supported
    :cite:`wohlberg-2016-convolutional`. See
    :class:`.ConvCnstrMODMaskDcplBase` for interface details.
    """


    class Options(ConvCnstrMODMaskDcplBase.Options):
        """ConvCnstrMODMaskDcpl_CG algorithm options

        Options include all of those defined in
        :class:`.ConvCnstrMODMaskDcplBase.Options`, together with
        additional options:

          ``CG`` : CG solver options

            ``MaxIter`` : Maximum CG iterations.

            ``StopTol`` : CG stopping tolerance.
        """

        defaults = copy.deepcopy(ConvCnstrMODMaskDcplBase.Options.defaults)
        defaults.update({'CG': {'MaxIter': 1000, 'StopTol': 1e-3}})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvCnstrMODMaskDcpl_CG algorithm options
            """

            if opt is None:
                opt = {}
            ConvCnstrMODMaskDcplBase.Options.__init__(self, opt)



    itstat_fields_extra = ('XSlvRelRes', 'XSlvCGIt')



    def __init__(self, Z, S, W, dsz, opt=None, dimK=1, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/ccmodmdcg_init.svg
           :width: 20%
           :target: ../_static/jonga/ccmodmdcg_init.svg
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvCnstrMODMaskDcpl_CG.Options()

        super(ConvCnstrMODMaskDcpl_CG, self).__init__(Z, S, W, dsz, opt,
                                                      dimK, dimN)
        self.Xf[:] = 0.0



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.cgit = None

        self.YU[:] = self.Y - self.U
        self.block_sep0(self.YU)[:] += self.S
        YUf = rfftn(self.YU, None, self.cri.axisN)
        b = sl.inner(np.conj(self.Zf), self.block_sep0(YUf),
                     axis=self.cri.axisK) + self.block_sep1(YUf)

        self.Xf[:], cgit = sl.solvemdbi_cg(
            self.Zf, 1.0, b, self.cri.axisM, self.cri.axisK,
            self.opt['CG', 'StopTol'], self.opt['CG', 'MaxIter'], self.Xf)
        self.cgit = cgit
        self.X = irfftn(self.Xf, self.cri.Nv, self.cri.axisN)
        self.xstep_check(b)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs, self.cgit)





class ConvCnstrMODMaskDcpl_Consensus(ccmod.ConvCnstrMOD_Consensus):
    r"""
    Hybrid ADMM Consensus algorithm for Convolutional Constrained MOD with
    Mask Decoupling :cite:`garcia-2018-convolutional1`.

    |

    .. inheritance-diagram:: ConvCnstrMODMaskDcpl_Consensus
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \left\|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right) \right\|_2^2 \quad \text{such that} \quad
       \mathbf{d}_m \in C \;\; \forall m

    where :math:`C` is the feasible set consisting of filters with unit
    norm and constrained support, and :math:`W` is a mask array, via a
    hybrid ADMM Consensus problem.

    See the documentation of :class:`.ConvCnstrMODMaskDcplBase` for a
    detailed discussion of the implementational complications resulting
    from the support of multi-channel signals.
    """


    def __init__(self, Z, S, W, dsz, opt=None, dimK=None, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/ccmodmdcnsns_init.svg
           :width: 20%
           :target: ../_static/jonga/ccmodmdcnsns_init.svg

        |

        Parameters
        ----------
        Z : array_like
          Coefficient map array
        S : array_like
          Signal array
        W : array_like
          Mask array. The array shape must be such that the array is
          compatible for multiplication with input array S (see
          :func:`.cnvrep.mskWshape` for more details).
        dsz : tuple
          Filter support size(s)
        opt : :class:`.ConvCnstrMOD_Consensus.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = ccmod.ConvCnstrMOD_Consensus.Options()

        super(ConvCnstrMODMaskDcpl_Consensus, self).__init__(
            Z, S, dsz, opt=opt, dimK=dimK, dimN=dimN)

        # Convert W to internal shape
        if W is None:
            W = np.array([1.0], dtype=self.dtype)
        W = np.asarray(W.reshape(cr.mskWshape(W, self.cri)),
                       dtype=S.dtype)

        # Reshape W if necessary (see discussion of reshape of S in
        # ccmod.ConvCnstrMOD_Consensus.__init__)
        if self.cri.Cd == 1 and self.cri.C > 1:
            # In most cases broadcasting rules make it possible for W
            # to have a singleton dimension corresponding to a non-singleton
            # dimension in S. However, when S is reshaped to interleave axisC
            # and axisK on the same axis, broadcasting is no longer sufficient
            # unless axisC and axisK of W are either both singleton or both
            # of the same size as the corresponding axes of S. If neither of
            # these cases holds, it is necessary to replicate the axis of W
            # (axisC or axisK) that does not have the same size as the
            # corresponding axis of S.
            shpw = list(W.shape)
            swck = shpw[self.cri.axisC] * shpw[self.cri.axisK]
            if swck > 1 and swck < self.cri.C * self.cri.K:
                if W.shape[self.cri.axisK] == 1 and self.cri.K > 1:
                    shpw[self.cri.axisK] = self.cri.K
                else:
                    shpw[self.cri.axisC] = self.cri.C
                W = np.broadcast_to(W, shpw)
            self.W = W.reshape(
                W.shape[0:self.cri.dimN] +
                (1, W.shape[self.cri.axisC] * W.shape[self.cri.axisK], 1))
        else:
            self.W = W

        # Initialise additional variables required for the different
        # splitting used in combining the consensus solution with mask
        # decoupling
        self.Y1 = np.zeros(self.S.shape, dtype=self.dtype)
        self.U1 = np.zeros(self.S.shape, dtype=self.dtype)
        self.YU1 = empty_aligned(self.S.shape, dtype=self.dtype)



    def setcoef(self, Z):
        """Set coefficient array."""

        # This method largely replicates the method from parent class
        # ConvCnstrMOD_Consensus that it overrides. The inherited
        # method is overridden to avoid the superfluous computation of
        # self.ZSf in that method, which is not required for the
        # modified algorithm with mask decoupling
        if self.cri.Cd == 1 and self.cri.C > 1:
            Z = Z.reshape(self.cri.Nv + (1,) + (self.cri.Cx*self.cri.K,) +
                          (self.cri.M,))
        self.Z = np.asarray(Z, dtype=self.dtype)
        self.Zf = rfftn(self.Z, self.cri.Nv, self.cri.axisN)



    def var_y1(self):
        """Get the auxiliary variable that is constrained to be equal to
        the dictionary. The method is named for compatibility with the
        method of the same name in :class:`.ConvCnstrMODMaskDcpl_IterSM`
        and :class:`.ConvCnstrMODMaskDcpl_CG` (it is *not* variable `Y1`
        in this class).
        """

        return self.Y



    def relax_AX(self):
        """The parent class method that this method overrides only
        implements the relaxation step for the variables of the baseline
        consensus algorithm. This method calls the overridden method and
        then implements the relaxation step for the additional variables
        required for the mask decoupling modification to the baseline
        algorithm.
        """

        super(ConvCnstrMODMaskDcpl_Consensus, self).relax_AX()
        self.AX1nr = irfftn(sl.inner(self.Zf, self.swapaxes(self.Xf),
                                        axis=self.cri.axisM),
                               self.cri.Nv, self.cri.axisN)
        if self.rlx == 1.0:
            self.AX1 = self.AX1nr
        else:
            alpha = self.rlx
            self.AX1 = alpha*self.AX1nr + (1-alpha)*(self.Y1 + self.S)



    def xstep(self):
        """The xstep of the baseline consensus class from which this
        class is derived is re-used to implement the xstep of the
        modified algorithm by replacing ``self.ZSf``, which is constant
        in the baseline algorithm, with a quantity derived from the
        additional variables ``self.Y1`` and ``self.U1``. It is also
        necessary to set the penalty parameter to unity for the duration
        of the x step.
        """

        self.YU1[:] = self.Y1 - self.U1
        self.ZSf = np.conj(self.Zf) * (self.Sf + rfftn(
            self.YU1, None, self.cri.axisN))
        rho = self.rho
        self.rho = 1.0
        super(ConvCnstrMODMaskDcpl_Consensus, self).xstep()
        self.rho = rho



    def ystep(self):
        """The parent class ystep method is overridden to allow also
        performing the ystep for the additional variables introduced in
        the modification to the baseline algorithm.
        """

        super(ConvCnstrMODMaskDcpl_Consensus, self).ystep()
        AXU1 = self.AX1 + self.U1
        self.Y1 = self.rho*(AXU1 - self.S) / (self.W**2 + self.rho)



    def ustep(self):
        """The parent class ystep method is overridden to allow also
        performing the ystep for the additional variables introduced in
        the modification to the baseline algorithm.
        """

        super(ConvCnstrMODMaskDcpl_Consensus, self).ustep()
        self.U1 += self.AX1 - self.Y1 - self.S



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| W \left( \sum_m
        \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \right) \|_2^2`.
        """

        Ef = sl.inner(self.Zf, self.obfn_fvarf(), axis=self.cri.axisM) \
          - self.Sf
        return (np.linalg.norm(self.W * irfftn(Ef, self.cri.Nv,
                                                  self.cri.axisN))**2) / 2.0



    def compute_residuals(self):
        """Compute residuals and stopping thresholds. The parent class
        method is overridden to ensure that the residual calculations
        include the additional variables introduced in the modification
        to the baseline algorithm.
        """

        # The full primary residual is straightforward to compute from
        # the primary residuals for the baseline algorithm and for the
        # additional variables
        r0 = self.rsdl_r(self.AXnr, self.Y)
        r1 = self.AX1nr - self.Y1 - self.S
        r = np.sqrt(np.sum(r0**2) + np.sum(r1**2))

        # The full dual residual is more complicated to compute than the
        # full primary residual
        ATU = self.swapaxes(self.U) + irfftn(
            np.conj(self.Zf) * rfftn(self.U1, self.cri.Nv, self.cri.axisN),
            self.cri.Nv, self.cri.axisN)
        s = self.rho * np.linalg.norm(ATU)

        # The normalisation factor for the full primal residual is also not
        # straightforward
        nAX = np.sqrt(np.linalg.norm(self.AXnr)**2 +
                      np.linalg.norm(self.AX1nr)**2)
        nY = np.sqrt(np.linalg.norm(self.Y)**2 +
                     np.linalg.norm(self.Y1)**2)
        rn = max(nAX, nY, np.linalg.norm(self.S))

        # The normalisation factor for the full dual residual is
        # straightforward to compute
        sn = self.rho * np.sqrt(np.linalg.norm(self.U)**2 +
                                np.linalg.norm(self.U1)**2)

        # Final residual values and stopping tolerances depend on
        # whether standard or normalised residuals are specified via the
        # options object
        if self.opt['AutoRho', 'StdResiduals']:
            epri = np.sqrt(self.Nc)*self.opt['AbsStopTol'] + \
                rn*self.opt['RelStopTol']
            edua = np.sqrt(self.Nx)*self.opt['AbsStopTol'] + \
                sn*self.opt['RelStopTol']
        else:
            if rn == 0.0:
                rn = 1.0
            if sn == 0.0:
                sn = 1.0
            r /= rn
            s /= sn
            epri = np.sqrt(self.Nc)*self.opt['AbsStopTol']/rn + \
                self.opt['RelStopTol']
            edua = np.sqrt(self.Nx)*self.opt['AbsStopTol']/sn + \
                self.opt['RelStopTol']

        return r, s, epri, edua





def ConvCnstrMODMaskDcpl(*args, **kwargs):
    """A wrapper function that dynamically defines a class derived from
    one of the implementations of the Convolutional Constrained MOD
    with Mask Decoupling problems, and returns an object instantiated
    with the provided. parameters. The wrapper is designed to allow the
    appropriate object to be created by calling this function using the
    same syntax as would be used if it were a class. The specific
    implementation is selected by use of an additional keyword
    argument 'method'. Valid values are:

    - ``'ism'`` :
      Use the implementation defined in :class:`.ConvCnstrMODMaskDcpl_IterSM`.
      This method works well for a small number of training images, but is
      very slow for larger training sets.
    - ``'cg'`` :
      Use the implementation defined in :class:`.ConvCnstrMODMaskDcpl_CG`.
      This method is slower than ``'ism'`` for small training sets, but has
      better run time scaling as the training set grows.
    - ``'cns'`` :
      Use the implementation defined in
      :class:`.ConvCnstrMODMaskDcpl_Consensus`. This method is the best choice
      for large training sets.

    The default value is ``'cns'``.
    """

    # Extract method selection argument or set default
    if 'method' in kwargs:
        method = kwargs['method']
        del kwargs['method']
    else:
        method = 'cns'

    # Assign base class depending on method selection argument
    if method == 'ism':
        base = ConvCnstrMODMaskDcpl_IterSM
    elif method == 'cg':
        base = ConvCnstrMODMaskDcpl_CG
    elif method == 'cns':
        base = ConvCnstrMODMaskDcpl_Consensus
    else:
        raise ValueError('Unknown ConvCnstrMODMaskDcpl solver method %s'
                         % method)

    # Nested class with dynamically determined inheritance
    class ConvCnstrMODMaskDcpl(base):
        def __init__(self, *args, **kwargs):
            super(ConvCnstrMODMaskDcpl, self).__init__(*args, **kwargs)

    # Allow pickling of objects of type ConvCnstrMODMaskDcpl
    _fix_dynamic_class_lookup(ConvCnstrMODMaskDcpl, method)

    # Return object of the nested class type
    return ConvCnstrMODMaskDcpl(*args, **kwargs)




def ConvCnstrMODMaskDcplOptions(opt=None, method='cns'):
    """A wrapper function that dynamically defines a class derived from
    the Options class associated with one of the implementations of
    the Convolutional Constrained MOD with Mask Decoupling  problem,
    and returns an object instantiated with the provided parameters.
    The wrapper is designed to allow the appropriate object to be
    created by calling this function using the same syntax as would be
    used if it were a class. The specific implementation is selected
    by use of an additional keyword argument 'method'. Valid values are
    as specified in the documentation for :func:`ConvCnstrMODMaskDcpl`.
    """

    # Assign base class depending on method selection argument
    if method == 'ism':
        base = ConvCnstrMODMaskDcpl_IterSM.Options
    elif method == 'cg':
        base = ConvCnstrMODMaskDcpl_CG.Options
    elif method == 'cns':
        base = ConvCnstrMODMaskDcpl_Consensus.Options
    else:
        raise ValueError('Unknown ConvCnstrMODMaskDcpl solver method %s'
                         % method)

    # Nested class with dynamically determined inheritance
    class ConvCnstrMODMaskDcplOptions(base):
        def __init__(self, opt):
            super(ConvCnstrMODMaskDcplOptions, self).__init__(opt)

    # Allow pickling of objects of type ConvCnstrMODMaskDcplOptions
    _fix_dynamic_class_lookup(ConvCnstrMODMaskDcplOptions, method)

    # Return object of the nested class type
    return ConvCnstrMODMaskDcplOptions(opt)
