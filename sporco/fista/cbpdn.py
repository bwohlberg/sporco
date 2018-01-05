# -*- coding: utf-8 -*-
# Copyright (C) 2016-2017 by Brendt Wohlberg <brendt@ieee.org>
#                            Cristina Garcia-Cardona <cgarciac@lanl.gov>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for FISTA algorithm for the Convolutional BPDN problem"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from builtins import range

import copy
from types import MethodType
import numpy as np
from scipy import linalg

import sporco.cnvrep as cr
import sporco.linalg as sl
from sporco.util import u

from sporco.fista import fista

__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""



class ConvBPDN(fista.FISTADFT):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ConvBPDN
       :parts: 2

    |

    Base class for FISTA algorithm for the Convolutional BPDN (CBPDN)
    :cite:`garcia-2017-convolutional` problem.

    The generic problem form is

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
        f( \{ \mathbf{x}_m \} ) + \lambda g( \{ \mathbf{x}_m \} )

    where :math:`f = (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
    \mathbf{s} \right\|_2^2`, and :math:`g(\cdot)` is a penalty
    term or the indicator function of a constraint; with input
    image :math:`\mathbf{s}`, dictionary filters :math:`\mathbf{d}_m`,
    and coefficient maps :math:`\mathbf{x}_m`. It is solved via the
    FISTA formulation

    Proximal step

    .. math::
       \mathbf{x}_k = \mathrm{prox}_{t_k}(g) (\mathbf{y}_k - 1/L \nabla
       f(\mathbf{y}_k) ) \;\;.

    Combination step

    .. math::
       \mathbf{y}_{k+1} = \mathbf{x}_k + \left( \frac{t_k - 1}{t_{k+1}} \right)
       (\mathbf{x}_k - \mathbf{x}_{k-1}) \;\;,

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


    class Options(fista.FISTADFT.Options):
        r"""ConvBPDN algorithm options

        Options include all of those defined in
        :class:`.fista.FISTADFT.Options`, together with
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

        defaults = copy.deepcopy(fista.FISTADFT.Options.defaults)
        defaults.update({'NonNegCoef': False, 'NoBndryCross': False})
        defaults.update({'L1Weight': 1.0})


        def __init__(self, opt=None):
            """Initialise ConvBPDN algorithm options object."""

            if opt is None:
                opt = {}
            fista.FISTADFT.Options.__init__(self, opt)


        def __setitem__(self, key, value):
            """Set options.
            """

            fista.FISTADFT.Options.__setitem__(self, key, value)


    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1'}



    def __init__(self, D, S, lmbda=None, opt=None, dimK=None, dimN=2):
        """
        Initialise a ConvBPDN object with problem parameters.

        This class supports an arbitrary number of spatial dimensions,
        `dimN`, with a default of 2. The input dictionary `D` is either
        `dimN` + 1 dimensional, in which case each spatial component
        (image in the default case) is assumed to consist of a single
        channel, or `dimN` + 2 dimensional, in which case the final
        dimension is assumed to contain the channels (e.g. colour
        channels in the case of images). The input signal set `S` is
        either `dimN` dimensional (no channels, only one signal), `dimN` + 1
        dimensional (either multiple channels or multiple signals), or
        `dimN` + 2 dimensional (multiple channels and multiple signals).
        Determination of problem dimensions is handled by
        :class:`.cnvrep.CSC_ConvRepIndexing`.


        |

        **Call graph**

        .. image:: _static/jonga/fista_cbpdn_init.svg
           :width: 20%
           :target: _static/jonga/fista_cbpdn_init.svg

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
            self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        # Set default lambda value if not specified
        if lmbda is None:
            cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)
            Df = sl.rfftn(D.reshape(cri.shpD), cri.Nv, axes=cri.axisN)
            Sf = sl.rfftn(S.reshape(cri.shpS), axes=cri.axisN)
            b = np.conj(Df) * Sf
            lmbda = 0.1*abs(b).max()

        # Set l1 term scaling and weight array
        self.lmbda = self.dtype.type(lmbda)
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)

        # Call parent class __init__
        xshape = self.cri.shpX
        super(ConvBPDN, self).__init__(xshape, S.dtype, opt)
        if self.opt['BackTrack', 'Enabled']:
            self.L /= self.lmbda

        # Reshape D and S to standard layout
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # Compute signal in DFT domain
        self.Sf = sl.rfftn(self.S, None, self.cri.axisN)

        # Create byte aligned arrays for FFT calls
        xfshp = list(self.X.shape)
        xfshp[dimN-1] = xfshp[dimN-1]//2 + 1
        self.Xf = sl.pyfftw_empty_aligned(xfshp,
                            dtype=sl.complex_dtype(self.dtype))

        # Initialise auxiliary variable Yf
        self.Yf = sl.pyfftw_empty_aligned(xfshp,
                            dtype=sl.complex_dtype(self.dtype))


        self.Ryf = -self.Sf

        self.Xf = sl.rfftn(self.X, None, self.cri.axisN)
        self.Yf = self.Xf
        self.store_prev()

        self.setdict()



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = sl.rfftn(self.D, self.cri.Nv, self.cri.axisN)


    def getcoef(self):
        """Get final coefficient array."""

        return self.X


    def eval_gradf(self):
        """ Compute gradient in Fourier domain """

        # Compute X D - S
        self.Ryf = self.eval_Rf(self.Yf)

        gradf = np.conj(self.Df) * self.Ryf

        # Multiple channel signal, multiple channel dictionary
        if self.cri.Cd > 1:
            gradf = np.sum(gradf, axis=self.cri.axisC, keepdims=True)

        return gradf


    def eval_proxop(self, V):
        """ Compute proximal operator of :math:`g` ."""

        return sl.shrink1(V, (self.lmbda/self.L) * self.wl1)



    def eval_Rf(self, Vf):
        """Evaluate smooth term in Vf."""

        return sl.inner(self.Df, Vf, axis=self.cri.axisM) - self.Sf



    def rsdl(self):
        """Compute fixed point residual in Fourier domain."""

        return linalg.norm(self.Xf - self.Yfprv)


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

        Ef = sl.inner(self.Df, self.Xf, axis=self.cri.axisM) - \
          self.Sf
        return sl.rfl2norm2(Ef, self.S.shape, axis=self.cri.axisN)/2.0



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = linalg.norm((self.wl1 * self.X).ravel(), 1)
        return (self.lmbda*rl1, rl1)


    def reconstruct(self, X=None):
        """Reconstruct representation."""

        if X is None:
            X = self.X
        Xf = sl.rfftn(X, None, self.cri.axisN)
        Sf = np.sum(self.Df * Xf, axis=self.cri.axisM)
        return sl.irfftn(Sf, self.cri.Nv, self.cri.axisN)
