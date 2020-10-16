# -*- coding: utf-8 -*-
# Copyright (C) 2016-2019 by Brendt Wohlberg <brendt@ieee.org>
#                            Cristina Garcia-Cardona <cgarciac@lanl.gov>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""PGM algorithms for the CCMOD problem"""

from __future__ import division, absolute_import

import copy
import numpy as np

from sporco.pgm import pgm
from sporco.array import atleast_nd
from sporco.fft import (rfftn, irfftn, empty_aligned, rfftn_empty_aligned,
                        rfl2norm2)
from sporco.linalg import inner
from sporco.cnvrep import CDU_ConvRepIndexing, getPcn, bcrop


__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""



class ConvCnstrMOD(pgm.PGMDFT):
    r"""
    Base class for PGM algorithm for Convolutional Constrained MOD
    problem :cite:`garcia-2018-convolutional1`.

    |

    .. inheritance-diagram:: ConvCnstrMOD
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_k \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \right\|_2^2 \quad \text{such that} \quad
       \mathbf{d}_m \in C

    via the PGM problem

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_k \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \right\|_2^2 + \sum_m \iota_C(\mathbf{d}_m) \;\;,

    where :math:`\iota_C(\cdot)` is the indicator function of feasible
    set :math:`C` consisting of filters with unit norm and constrained
    support. Multi-channel problems with input image channels
    :math:`\mathbf{s}_{c,k}` are also supported, either as

    .. math::
       \mathrm{argmin}_\mathbf{d} \; (1/2) \sum_c \sum_k \left\| \sum_m
       \mathbf{d}_m * \mathbf{x}_{c,k,m} - \mathbf{s}_{c,k} \right\|_2^2
       \quad \text{such that} \quad \mathbf{d}_m \in C

    with single-channel dictionary filters :math:`\mathbf{d}_m` and
    multi-channel coefficient maps :math:`\mathbf{x}_{c,k,m}`, or

    .. math::
       \mathrm{argmin}_\mathbf{d} \; (1/2) \sum_c \sum_k \left\| \sum_m
       \mathbf{d}_{c,m} * \mathbf{x}_{k,m} - \mathbf{s}_{c,k}
       \right\|_2^2 \quad \text{such that} \quad \mathbf{d}_{c,m} \in C

    with multi-channel dictionary filters :math:`\mathbf{d}_{c,m}` and
    single-channel coefficient maps :math:`\mathbf{x}_{k,m}`. In this
    latter case, normalisation of filters :math:`\mathbf{d}_{c,m}` is
    performed jointly over index :math:`c` for each filter :math:`m`.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``DFid`` : Value of data fidelity term :math:`(1/2) \sum_k \|
       \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} - \mathbf{s}_k \|_2^2`

       ``Cnstr`` : Constraint violation measure

       ``Rsdl`` : Residual

       ``L`` : Inverse of gradient step parameter

       ``Time`` : Cumulative run time
    """



    class Options(pgm.PGMDFT.Options):
        r"""ConvCnstrMOD algorithm options

        Options include all of those defined in
        :class:`.pgm.PGMDFT.Options`, together with
        additional options:

          ``ZeroMean`` : Flag indicating whether the solution
          dictionary :math:`\{\mathbf{d}_m\}` should have zero-mean
          components.
        """

        defaults = copy.deepcopy(pgm.PGMDFT.Options.defaults)
        defaults.update({'ZeroMean': False})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvCnstrMOD algorithm options
            """

            if opt is None:
                opt = {}
            pgm.PGMDFT.Options.__init__(self, opt)


        def __setitem__(self, key, value):
            """Set options."""

            pgm.PGMDFT.Options.__setitem__(self, key, value)


    itstat_fields_objfn = ('DFid', 'Cnstr')
    hdrtxt_objfn = ('DFid', 'Cnstr')
    hdrval_objfun = {'DFid': 'DFid', 'Cnstr': 'Cnstr'}



    def __init__(self, Z, S, dsz, opt=None, dimK=1, dimN=2):
        """
        This class supports an arbitrary number of spatial dimensions,
        `dimN`, with a default of 2. The input coefficient map array `Z`
        (usually labelled X, but renamed here to avoid confusion with
        the X and Y variables in the PGM base class) is expected to
        be in standard form as computed by the GenericConvBPDN class.

        The input signal set `S` is either `dimN` dimensional (no
        channels, only one signal), `dimN` +1 dimensional (either
        multiple channels or multiple signals), or `dimN` +2 dimensional
        (multiple channels and multiple signals). Parameter `dimK`, with
        a default value of 1, indicates the number of multiple-signal
        dimensions in `S`:

        ::

          Default dimK = 1, i.e. assume input S is of form
            S(N0,  N1,   C,   K)  or  S(N0,  N1,   K)
          If dimK = 0 then input S is of form
            S(N0,  N1,   C,   K)  or  S(N0,  N1,   C)

        The internal data layout for S, D (X here), and X (Z here) is:
        ::

          dim<0> - dim<Nds-1> : Spatial dimensions, product of N0,N1,... is N
          dim<Nds>            : C number of channels in S and D
          dim<Nds+1>          : K number of signals in S
          dim<Nds+2>          : M number of filters in D

            sptl.      chn  sig  flt
          S(N0,  N1,   C,   K,   1)
          D(N0,  N1,   C,   1,   M)   (X here)
          X(N0,  N1,   1,   K,   M)   (Z here)

        The `dsz` parameter indicates the desired filter supports in the
        output dictionary, since this cannot be inferred from the
        input variables. The format is the same as the `dsz` parameter
        of :func:`.cnvrep.bcrop`.


        |

        **Call graph**

        .. image:: ../_static/jonga/ccmodpgm_init.svg
           :width: 20%
           :target: ../_static/jonga/ccmodpgm_init.svg

        |


        Parameters
        ----------
        Z : array_like
          Coefficient map array
        S : array_like
          Signal array
        dsz : tuple
          Filter support size(s)
        opt : ccmod.Options object
          Algorithm options
        dimK : int, optional (default 1)
          Number of dimensions for multiple signals in input S
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvCnstrMOD.Options()

        # Infer problem dimensions and set relevant attributes of self
        self.cri = CDU_ConvRepIndexing(dsz, S, dimK=dimK, dimN=dimN)

        # Call parent class __init__
        xshape = self.cri.shpD
        Nv = self.cri.Nv
        axisN = self.cri.axisN
        super(ConvCnstrMOD, self).__init__(xshape, Nv, axisN, S.dtype, opt)

        # Set gradient step parameter
        self.set_attr('L', opt['L'], dval=self.cri.K * 14.0, dtype=self.dtype)

        # Reshape S to standard layout (Z, i.e. X in cbpdn, is assumed
        # to be taken from cbpdn, and therefore already in standard
        # form). If the dictionary has a single channel but the input
        # (and therefore also the coefficient map array) has multiple
        # channels, the channel index and multiple image index have
        # the same behaviour in the dictionary update equation: the
        # simplest way to handle this is to just reshape so that the
        # channels also appear on the multiple image index.
        if self.cri.Cd == 1 and self.cri.C > 1:
            self.S = S.reshape(self.cri.Nv + (1,) +
                               (self.cri.C * self.cri.K,) + (1,))
        else:
            self.S = S.reshape(self.cri.shpS)
        self.S = np.asarray(self.S, dtype=self.dtype)

        # Compute signal S in DFT domain
        self.Sf = rfftn(self.S, None, self.cri.axisN)

        # Create constraint set projection function
        self.Pcn = getPcn(dsz, self.cri.Nv, self.cri.dimN, self.cri.dimCd,
                          zm=opt['ZeroMean'])

        # Create byte aligned arrays for FFT calls
        self.Y = self.X
        self.X = empty_aligned(self.Y.shape, dtype=self.dtype)
        self.X[:] = self.Y

        # Initialise auxiliary variable Vf: Create byte aligned arrays
        # for FFT calls
        self.Vf = rfftn_empty_aligned(self.X.shape, self.cri.axisN,
                                      self.dtype)

        self.Xf = rfftn(self.X, None, self.cri.axisN)
        self.Yf = self.Xf
        self.Yfprv = self.Yf.copy() + 1e5

        if Z is not None:
            self.setcoef(Z)



    def setcoef(self, Z):
        """Set coefficient array."""

        # If the dictionary has a single channel but the input (and
        # therefore also the coefficient map array) has multiple
        # channels, the channel index and multiple image index have
        # the same behaviour in the dictionary update equation: the
        # simplest way to handle this is to just reshape so that the
        # channels also appear on the multiple image index.
        if self.cri.Cd == 1 and self.cri.C > 1:
            Z = Z.reshape(self.cri.Nv + (1,)
                          + (self.cri.Cx * self.cri.K,)
                          + (self.cri.M,))
        self.Z = np.asarray(Z, dtype=self.dtype)

        self.Zf = rfftn(self.Z, self.cri.Nv, self.cri.axisN)



    def getdict(self, crop=True):
        """Get final dictionary. If ``crop`` is ``True``, apply
        :func:`.cnvrep.bcrop` to returned array.
        """

        D = self.X
        if crop:
            D = bcrop(D, self.cri.dsz, self.cri.dimN)
        return D



    def grad_f(self, Vf=None):
        """Compute gradient in Fourier domain."""

        if Vf is None:
            Vf = self.Yf
        # Compute Zf Vf - Sf
        Ryf = self.eval_Rf(Vf)

        gradf = inner(np.conj(self.Zf), Ryf, axis=self.cri.axisK)

        # Multiple channel signal, single channel dictionary
        if self.cri.C > 1 and self.cri.Cd == 1:
            gradf = np.sum(gradf, axis=self.cri.axisC, keepdims=True)

        return gradf



    def eval_Rf(self, Vf):
        """Evaluate smooth term in Vf."""

        return inner(self.Zf, Vf, axis=self.cri.axisM) - self.Sf



    def prox_g(self, V):
        """Compute proximal operator of :math:`g`."""

        return self.Pcn(V)



    def hessian_f(self, V):
        """Compute Hessian of :math:`f` applied to V."""

        hessfv = inner(self.Zf, V, axis=self.cri.axisM)
        hessfv = inner(np.conj(self.Zf), hessfv, axis=self.cri.axisK)

        # Multiple channel signal, single channel dictionary
        if self.cri.C > 1 and self.cri.Cd == 1:
            hessfv = np.sum(hessfv, axis=self.cri.axisC, keepdims=True)

        return hessfv



    def rsdl(self):
        """Compute fixed point residual in Fourier domain."""

        diff = self.Xf - self.Yfprv
        return rfl2norm2(diff, self.X.shape, axis=self.cri.axisN)



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_dfd()
        cns = self.obfn_cns()
        return (dfd, cns)



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| \sum_m
        \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`.
        """

        Ef = self.eval_Rf(self.Xf)
        return rfl2norm2(Ef, self.S.shape, axis=self.cri.axisN) / 2.0



    def obfn_cns(self):
        r"""Compute constraint violation measure :math:`\|
        P(\mathbf{y}) - \mathbf{y}\|_2`.
        """

        return np.linalg.norm((self.Pcn(self.X) - self.X))



    def obfn_f(self, Xf=None):
        r"""Compute data fidelity term :math:`(1/2) \| \sum_m
        \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`.
        This is used for backtracking. Since the backtracking is
        computed in the DFT, it is important to preserve the
        DFT scaling.
        """
        if Xf is None:
            Xf = self.Xf

        Rf = self.eval_Rf(Xf)
        return 0.5 * np.linalg.norm(Rf.flatten(), 2)**2



    def reconstruct(self, D=None):
        """Reconstruct representation."""

        if D is None:
            Df = self.Xf
        else:
            Df = rfftn(D, None, self.cri.axisN)

        Sf = np.sum(self.Zf * Df, axis=self.cri.axisM)
        return irfftn(Sf, self.cri.Nv, self.cri.axisN)




class ConvCnstrMODMask(ConvCnstrMOD):
    r"""
    PGM algorithm for Convolutional Constrained MOD problem
    with a spatial mask :cite:`garcia-2018-convolutional1`.

    |

    .. inheritance-diagram:: ConvCnstrMODMask
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
    PGM problem

    .. math::
       \mathrm{argmin}_{\mathbf{d}} \; (1/2) \left\|  W \left(X
       \mathbf{d} - \mathbf{s}\right) \right\|_2^2 +
       \iota_C(\mathbf{d}_m) \;\;,

    where  :math:`\iota_C(\cdot)` is the indicator function of feasible
    set :math:`C`, and :math:`X \mathbf{d} = \sum_m \mathbf{x}_m *
    \mathbf{d}_m`.

    See :class:`ConvCnstrMOD` for interface details.
    """


    class Options(ConvCnstrMOD.Options):
        """ConvCnstrMODMask algorithm options

        Options include all of those defined in
        :class:`.pgm.PGM.Options`.
        """

        defaults = copy.deepcopy(ConvCnstrMOD.Options.defaults)


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvCnstrMODMask algorithm options
            """

            if opt is None:
                opt = {}
            ConvCnstrMOD.Options.__init__(self, opt)


    def __init__(self, Z, S, W, dsz, opt=None, dimK=None, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/ccmodmdpgm_init.svg
           :width: 20%
           :target: ../_static/jonga/ccmodmdpgm_init.svg

        |


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
            discussion of the distinction between *external* and
            *internal* data layouts).
        dsz : tuple
            Filter support size(s)
        opt : :class:`ConvCnstrMODMask.Options` object
            Algorithm options
        dimK : 0, 1, or None, optional (default None)
            Number of dimensions in input signal corresponding to multiple
            independent signals
        dimN : int, optional (default 2)
            Number of spatial dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvCnstrMODMask.Options()

        # Infer problem dimensions and set relevant attributes of self
        self.cri = CDU_ConvRepIndexing(dsz, S, dimK=dimK, dimN=dimN)

        # Append singleton dimensions to W if necessary
        if hasattr(W, 'ndim'):
            W = atleast_nd(self.cri.dimN + 3, W)

        # Reshape W if necessary (see discussion of reshape of S in
        # ccmod base class)
        if self.cri.Cd == 1 and self.cri.C > 1 and hasattr(W, 'ndim'):
            # In most cases broadcasting rules make it possible for W
            # to have a singleton dimension corresponding to a
            # non-singleton dimension in S. However, when S is
            # reshaped to interleave axisC and axisK on the same axis,
            # broadcasting is no longer sufficient unless axisC and
            # axisK of W are either both singleton or both of the same
            # size as the corresponding axes of S. If neither of these
            # cases holds, it is necessary to replicate the axis of W
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
                W.shape[0:self.cri.dimN] + (1, W.shape[self.cri.axisC] *
                                            W.shape[self.cri.axisK], 1))
        else:
            self.W = W

        super(ConvCnstrMODMask, self).__init__(Z, S, dsz, opt, dimK, dimN)

        # Create byte aligned arrays for FFT calls
        self.WRy = empty_aligned(self.S.shape, dtype=self.dtype)
        self.Ryf = rfftn_empty_aligned(self.S.shape, self.cri.axisN,
                                       self.dtype)



    def grad_f(self, Vf=None):
        """Compute gradient in Fourier domain."""

        if Vf is not None:
            # Compute X V - S
            self.Ryf[:] = self.eval_Rf(Vf)
        else:
            # Compute X D - S
            self.Ryf[:] = self.eval_Rf(self.Yf)

        # Map to spatial domain to multiply by mask
        Ry = irfftn(self.Ryf, self.cri.Nv, self.cri.axisN)
        # Multiply by mask
        self.WRy[:] = (self.W**2) * Ry
        # Map back to frequency domain
        WRyf = rfftn(self.WRy, self.cri.Nv, self.cri.axisN)

        gradf = inner(np.conj(self.Zf), WRyf, axis=self.cri.axisK)

        # Multiple channel signal, single channel dictionary
        if self.cri.C > 1 and self.cri.Cd == 1:
            gradf = np.sum(gradf, axis=self.cri.axisC, keepdims=True)

        return gradf



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \sum_k \| W (\sum_m
        \mathbf{d}_m * \mathbf{x}_{k,m} - \mathbf{s}_k) \|_2^2`
        """

        Ef = self.eval_Rf(self.Xf)
        E = irfftn(Ef, self.cri.Nv, self.cri.axisN)

        return (np.linalg.norm(self.W * E)**2) / 2.0



    def obfn_f(self, Xf=None):
        r"""Compute data fidelity term :math:`(1/2) \sum_k \| W (\sum_m
        \mathbf{d}_m * \mathbf{x}_{k,m} - \mathbf{s}_k) \|_2^2`.
        This is used for backtracking. Since the backtracking is
        computed in the DFT, it is important to preserve the
        DFT scaling.
        """

        if Xf is None:
            Xf = self.Xf

        Rf = self.eval_Rf(Xf)
        R = irfftn(Rf, self.cri.Nv, self.cri.axisN)
        WRf = rfftn(self.W * R, self.cri.Nv, self.cri.axisN)

        return 0.5 * np.linalg.norm(WRf.flatten(), 2)**2
