# -*- coding: utf-8 -*-
# Copyright (C) 2016-2020 by Brendt Wohlberg <brendt@ieee.org>
#                            Cristina Garcia-Cardona <cgarciac@lanl.gov>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for PGM algorithm for the Convolutional BPDN problem"""

from __future__ import division, absolute_import, print_function

import copy
import numpy as np

from sporco.util import u
from sporco.cnvrep import CSC_ConvRepIndexing, mskWshape
from sporco.linalg import inner
from sporco.fft import (rfftn, irfftn, empty_aligned, rfftn_empty_aligned,
                        rfl2norm2)
from sporco.prox import prox_l1
from sporco.pgm import pgm


__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""



class ConvBPDN(pgm.PGMDFT):
    r"""
    Base class for PGM algorithm for the Convolutional BPDN (CBPDN)
    :cite:`garcia-2018-convolutional1` problem.

    |

    .. inheritance-diagram:: ConvBPDN
       :parts: 2

    |

    The generic problem form is

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
        f( \{ \mathbf{x}_m \} ) + \lambda g( \{ \mathbf{x}_m \} )

    where :math:`f = (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
    \mathbf{s} \right\|_2^2`, and :math:`g(\cdot)` is a penalty
    term or the indicator function of a constraint; with input
    image :math:`\mathbf{s}`, dictionary filters :math:`\mathbf{d}_m`,
    and coefficient maps :math:`\mathbf{x}_m`. It is solved via the
    PGM formulation

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

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| \sum_m
       \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_m \|
       \mathbf{x}_m \|_1`

       ``Rsdl`` : Residual

       ``L`` : Inverse of gradient step parameter

       ``Time`` : Cumulative run time
    """


    class Options(pgm.PGMDFT.Options):
        r"""ConvBPDN algorithm options

        Options include all of those defined in
        :class:`.pgm.PGMDFT.Options`, together with
        additional options:

          ``NonNegCoef`` : Flag indicating whether to force solution to
          be non-negative.

          ``NoBndryCross`` : Flag indicating whether all solution
          coefficients corresponding to filters crossing the image
          boundary should be forced to zero.

          ``L1Weight`` : An array of weights for the :math:`\ell_1`
          norm. The array shape must be such that the array is
          compatible for multiplication with the X/Y variables. If this
          option is defined, the regularization term is :math:`\lambda
          \sum_m \| \mathbf{w}_m \odot \mathbf{x}_m \|_1` where
          :math:`\mathbf{w}_m` denotes slices of the weighting array on
          the filter index axis.

        """

        defaults = copy.deepcopy(pgm.PGMDFT.Options.defaults)
        defaults.update({'NonNegCoef': False, 'NoBndryCross': False})
        defaults.update({'L1Weight': 1.0})
        defaults.update({'L': 500.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDN algorithm options
            """

            if opt is None:
                opt = {}
            pgm.PGMDFT.Options.__init__(self, opt)



        def __setitem__(self, key, value):
            """Set options."""

            pgm.PGMDFT.Options.__setitem__(self, key, value)


    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1'}



    def __init__(self, D, S, lmbda=None, opt=None, dimK=None, dimN=2):
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


        |

        **Call graph**

        .. image:: ../_static/jonga/pgm_cbpdn_init.svg
           :width: 20%
           :target: ../_static/jonga/pgm_cbpdn_init.svg

        |


        Parameters
        ----------
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        lmbda : float
          Regularisation parameter
        opt : :class:`ConvBPDN.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvBPDN.Options()

        # Infer problem dimensions and set relevant attributes of self
        if not hasattr(self, 'cri'):
            self.cri = CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        # Set default lambda value if not specified
        if lmbda is None:
            cri = CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)
            Df = rfftn(D.reshape(cri.shpD), cri.Nv, axes=cri.axisN)
            Sf = rfftn(S.reshape(cri.shpS), axes=cri.axisN)
            b = np.conj(Df) * Sf
            lmbda = 0.1 * abs(b).max()

        # Set l1 term scaling and weight array
        self.lmbda = self.dtype.type(lmbda)
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)

        # Call parent class __init__
        self.Xf = None
        xshape = self.cri.shpX
        Nv = self.cri.Nv
        axisN = self.cri.axisN
        super(ConvBPDN, self).__init__(xshape, Nv, axisN, S.dtype, opt)

        # Reshape D and S to standard layout
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # Compute signal in DFT domain
        self.Sf = rfftn(self.S, None, self.cri.axisN)

        # Create byte aligned arrays for FFT calls
        self.Y = self.X.copy()
        self.X = empty_aligned(self.Y.shape, dtype=self.dtype)
        self.X[:] = self.Y

        # Initialise auxiliary variable Vf: Create byte aligned arrays
        # for FFT calls
        self.Vf = rfftn_empty_aligned(self.X.shape, self.cri.axisN,
                                      self.dtype)


        self.Xf = rfftn(self.X, None, self.cri.axisN)
        self.Yf = self.Xf.copy()
        self.Yfprv = self.Yf.copy() + 1e5

        self.setdict()



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = rfftn(self.D, self.cri.Nv, self.cri.axisN)



    def getcoef(self):
        """Get final coefficient array."""

        return self.X



    def grad_f(self, Vf=None):
        """Compute gradient in Fourier domain."""

        if Vf is None:
            Vf = self.Yf
        # Compute Df Vf - Sf
        Ryf = self.eval_Rf(Vf)
        # Compute D^H Ryf
        gradf = np.conj(self.Df) * Ryf

        # Multiple channel signal, multiple channel dictionary
        if self.cri.Cd > 1:
            gradf = np.sum(gradf, axis=self.cri.axisC, keepdims=True)

        return gradf



    def eval_Rf(self, Vf):
        """Evaluate smooth term in Vf."""

        return inner(self.Df, Vf, axis=self.cri.axisM) - self.Sf



    def prox_g(self, V):
        """Compute proximal operator of :math:`g`."""

        U = prox_l1(V, (self.lmbda / self.L) * self.wl1)
        if self.opt['NonNegCoef']:
            U[U < 0.0] = 0.0
        if self.opt['NoBndryCross']:
            for n in range(0, self.cri.dimN):
                U[(slice(None),) * n +
                  (slice(1 - self.D.shape[n], None),)] = 0.0
        return U



    def hessian_f(self, V):
        """Compute Hessian of :math:`f` applied to V."""

        hessfv = np.conj(self.Df) * inner(self.Df, V, axis=self.cri.axisM)
        # Multiple channel signal, multiple channel dictionary
        if self.cri.Cd > 1:
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
        reg = self.obfn_reg()
        obj = dfd + reg[0]
        return (obj, dfd) + reg[1:]



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| \sum_m
        \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`.
        This function takes into account the unnormalised DFT scaling,
        i.e. given that the variables are the DFT of multi-dimensional
        arrays computed via :func:`.rfftn`, this returns the data
        fidelity term in the original (spatial) domain.
        """

        Ef = self.eval_Rf(self.Xf)
        return rfl2norm2(Ef, self.S.shape, axis=self.cri.axisN) / 2.0



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = np.linalg.norm((self.wl1 * self.X).ravel(), 1)
        return (self.lmbda * rl1, rl1)



    def obfn_f(self, Xf=None):
        r"""Compute data fidelity term :math:`(1/2) \| \sum_m
        \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`
        This is used for backtracking. Since the backtracking is
        computed in the DFT, it is important to preserve the
        DFT scaling.
        """

        if Xf is None:
            Xf = self.Xf

        Rf = self.eval_Rf(Xf)
        return 0.5 * np.linalg.norm(Rf.flatten(), 2)**2



    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            X = self.X
        Xf = rfftn(X, None, self.cri.axisN)
        Sf = np.sum(self.Df * Xf, axis=self.cri.axisM)
        return irfftn(Sf, self.cri.Nv, self.cri.axisN)





class ConvBPDNMask(ConvBPDN):
    r"""
    PGM algorithm for Convolutional BPDN with a spatial mask.

    |

    .. inheritance-diagram:: ConvBPDNMask
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s}\right) \right\|_2^2 + \lambda \sum_m
       \| \mathbf{x}_m \|_1 \;\;,

    where :math:`W` is a mask array.

    See :class:`ConvBPDN` for interface details.
    """


    def __init__(self, D, S, lmbda, W=None, opt=None, dimK=None, dimN=2):
        """

        |


        Parameters
        ----------
        D : array_like
          Dictionary matrix
        S : array_like
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        W : array_like
          Mask array. The array shape must be such that the array is
          compatible for multiplication with input array S (see
          :func:`.cnvrep.mskWshape` for more details).
        opt : :class:`ConvBPDNMask.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        super(ConvBPDNMask, self).__init__(D, S, lmbda, opt, dimK=dimK,
                                           dimN=dimN)

        if W is None:
            W = np.array([1.0], dtype=self.dtype)
        self.W = np.asarray(W.reshape(mskWshape(W, self.cri)),
                            dtype=self.dtype)

        # Create byte aligned arrays for FFT calls
        self.WRy = empty_aligned(self.S.shape, dtype=self.dtype)
        self.Ryf = rfftn_empty_aligned(self.S.shape, self.cri.axisN,
                                       self.dtype)



    def grad_f(self, Vf=None):
        """Compute gradient in Fourier domain."""

        if Vf is not None:
            # Compute D V - S
            self.Ryf[:] = self.eval_Rf(Vf)
        else:
            # Compute D X - S
            self.Ryf[:] = self.eval_Rf(self.Yf)

        # Map to spatial domain to multiply by mask
        Ry = irfftn(self.Ryf, self.cri.Nv, self.cri.axisN)
        # Multiply by mask
        self.WRy[:] = (self.W**2) * Ry
        # Map back to frequency domain
        WRyf = rfftn(self.WRy, self.cri.Nv, self.cri.axisN)

        gradf = np.conj(self.Df) * WRyf

        # Multiple channel signal, multiple channel dictionary
        if self.cri.Cd > 1:
            gradf = np.sum(gradf, axis=self.cri.axisC, keepdims=True)

        return gradf



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| W (\sum_m
        \mathbf{d}_m * \mathbf{x}_{m} - \mathbf{s}) \|_2^2`
        """

        Ef = self.eval_Rf(self.Xf)
        E = irfftn(Ef, self.cri.Nv, self.cri.axisN)

        return (np.linalg.norm(self.W * E)**2) / 2.0



    def obfn_f(self, Xf=None):
        r"""Compute data fidelity term :math:`(1/2) \| W (\sum_m
        \mathbf{d}_m * \mathbf{x}_{m} - \mathbf{s}) \|_2^2`.
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
