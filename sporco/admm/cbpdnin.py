# -*- coding: utf-8 -*-
# Copyright (C) 2020 by Frank Cwitkowitz <fcwitkow@ur.rochester.edu>
#                       Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Class for ADMM algorithm for convolutional sparse coding with
inhibition terms"""

from __future__ import division, print_function
from builtins import range

import copy
import numpy as np
from scipy import signal

from sporco.admm import cbpdn
import sporco.prox as sp
from sporco.util import u
from sporco.fft import (rfftn, irfftn)


__author__ = """Frank Cwitkowitz <fcwitkow@ur.rochester.edu>"""


class ConvBPDNInhib(cbpdn.ConvBPDN):
    r"""
    ADMM algorithm for Convolutional BPDN with inhibition via
    weighted :math:`\ell_{1}` norm terms :cite:`cogliati-2017-piano`

    |

    .. inheritance-diagram:: ConvBPDNInhib
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1
       + \mu \sum_m \boldsymbol{\omega}^T_m | \mathbf{x}_m | +
       \gamma \sum_m \mathbf{z}^T_m | \mathbf{x}_m |

    for input image :math:`\mathbf{s}`, dictionary filters
    :math:`\mathbf{d}_m`, and coefficient maps :math:`\mathbf{x}_m`,
    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s} \right\|_2^2 + \lambda \sum_m \| \mathbf{y}_m \|_1 +
       \mu \sum_m \boldsymbol{\omega}^T_m | \mathbf{y}_m | +
       \gamma \sum_m \mathbf{z}^T_m | \mathbf{y}_m |
       \quad \text{such that} \quad \mathbf{x}_m = \mathbf{y}_m \;\;.

    Here, :math:`\boldsymbol{\omega}^T_m = \sum_n c_{m,n} (| \mathbf{x}_n *
    \mathbf{h} |)^T` and :math:`\mathbf{z}^T_m = \sum_m (| \mathbf{x}_m
    * \mathbf{h}' |)^T`, where :math:`c_{m,n}` is a square matrix with
    non-zero entries where elements :math:`m` and :math:`n` share the
    same group and :math:`m != n`, :math:`\mathbf{h}` is a spatial
    weighting matrix non-zero around the origin with radius
    :math:`\frac{T}{2}`, and :math:`\mathbf{h}'` is the same matrix with
    zero at the origin.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| \sum_m
       \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_m \|
       \mathbf{x}_m \|_1`

       ``RegLat`` : Value of regularisation term :math:`\sum_m
       \boldsymbol{\omega}^T_m | \mathbf{x}_m |`

       ``RegSelf`` : Value of regularisation term :math:`\sum_m
       \mathbf{z}^T_m | \mathbf{x}_m |`

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
        r"""ConvBPDNInhib algorithm options

        Options include all of those defined in
        :class:`.admm.cbpdn.ConvBPDN.Options`, together with additional
        options:

          ``SmoothWeight`` : Smoothing for the weighted :math:`\ell_1`
          norms. The value acts as the percentage of the previous weights
          to superimpose with the new weights before iterating.
        """

        defaults = copy.deepcopy(cbpdn.ConvBPDN.Options.defaults)
        defaults.update({'SmoothWeight': 0.9})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDNInhib algorithm options
            """

            if opt is None:
                opt = {}
            cbpdn.ConvBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegLat', 'RegSelf')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), 'RegLat', 'RegSelf')
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u(
        'Regℓ1'): 'RegL1', 'RegLat': 'RegLat', 'RegSelf': 'RegSelf'}



    def __init__(self, D, S, Wg=None, Whn=None, win_args=None,
                 lmbda=None, mu=None, gamma=None, opt=None, dimK=None, dimN=2):
        """
        This class supports an arbitrary number of spatial dimensions,
        `dimN`, with a default of 2. The input dictionary `D` is either
        `dimN` + 1 dimensional, in which case each spatial component
        (image in the default case) is assumed to consist of a single
        channel, or `dimN` + 2 dimensional, in which case the final
        dimension is assumed to contain the channels (e.g. colour
        channels in the case of images). The input signal set `S` is
        either `dimN` dimensional (no channels, only one signal), `dimN`
        + 1 dimensional (either multiple channels or multiple signals),
        or `dimN` + 2 dimensional (multiple channels and multiple
        signals). Determination of problem dimensions is handled by
        :class:`.cnvrep.CSC_ConvRepIndexing`.


        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdnin_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdnin_init.svg

        |


        Parameters
        ----------
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        Wg : array_like
          Ng x M grouping matrix with rows representing groups where non-zero
          values indicate strength of element membership - dimensionality of
          input signal does not affect the shape of this matrix
        Whn: int
          Diameter of inhibition window (in samples) across each dimension
        win_args: tuple
          Window function parameters for inhibition window, passed to
          :func:`scipy.signal.get_window`
        lmbda : float
          Regularisation parameter for sparsity
        mu : float
          Regularisation parameter for lateral inhibition - this discourages
          grouped elements from being active within the same windowed area
        gamma : float
          Regularisation parameter for self inhibition - this discourages
          single elements from being active more than once within the same
          windowed area, leading to more impulse-like activations
        opt : :class:`ConvBPDNInhib.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvBPDNInhib.Options()

        # Call parent class __init__
        super(ConvBPDNInhib, self).__init__(D, S, lmbda, opt, dimK, dimN)

        # Add the groups to the class instance
        self.Wg = Wg

        # Set default lateral inhibition scaling term
        if mu is None:
            mu = 10 * self.lmbda
        self.mu = self.dtype.type(mu)

        # Set default self inhibition scaling term
        if gamma is None:
            # No self inhibition by default
            gamma = 0.0
        self.gamma = self.dtype.type(gamma)

        # Initialize lateral and self inhibition weights, respectively,
        # as zero
        self.wml, self.wms = 0, 0

        # If no grouping scheme and no gamma are provided, nothing
        # indicates inhibition, and standard CBPDN is executed.
        if (self.Wg is not None and self.mu != 0) or self.gamma:
            if self.Wg is not None:
                # Set the type of the grouping weights
                self.Wg = self.Wg.astype(self.dtype)

                # Add the number of groups to the indexer
                self.cri.Ng = self.Wg.shape[0]

                # Determine the number of filters per group and add to
                # the indexer
                self.cri.Mgs = np.sum((self.Wg != 0), axis=1)

            if Whn is None:
                # Make inhibition window size of dictionary elements by
                # default
                Whn = self.D.shape[self.cri.axisN[0]]

            if win_args is None:
                win_args = ('tukey', 0.5)

            # Create generalized spatial weighting matrix. This matrix
            # is convolved with activations during inhibition to window
            # and enforce locality of inhibition.
            whl_shape = self.cri.Nv + (1,) * 3
            Whl = np.zeros(whl_shape, dtype=self.dtype)
            # Ensure inhibition sample length is odd for symmetric
            # inhibition and so the origin is the first element of the
            # matrix
            Whn += not Whn % 2
            # Create N-dimensional window function
            nDimInd = tuple(np.meshgrid(*([np.arange(Whn)] * dimN)))
            nDimWin = np.meshgrid(
                *([np.array(signal.get_window(win_args, Whn))] * dimN))
            nDimWin = np.concatenate(
                [np.expand_dims(nDimWin[i], axis=0)
                 for i in range(len(nDimWin))])
            nDimWin = np.power(np.prod(nDimWin, axis=0), 1 / dimN)
            Whl[nDimInd] = np.reshape(nDimWin, Whl[nDimInd].shape)

            # Center window around origin in each dimension
            for i in range(dimN):
                Whl = np.roll(Whl, -Whn // 2 + 1, axis=i)
            # Create a spatial weighting matrix for self inhibition
            # (zero-out t=0)
            Whs = Whl.copy()
            Whs[tuple([0] * dimN)] = 0

            # Obtain the lateral and self inhibition windows in
            # frequency domain
            self.Whfl = rfftn(Whl, self.cri.Nv, self.cri.axisN)
            self.Whfs = rfftn(Whs, self.cri.Nv, self.cri.axisN)

            # Initialize previous values for lateral and self inhibition
            # weights
            self.wml_prev, self.wms_prev = None, None

            # Initialize smoothing for inhibition terms
            self.smooth = opt['SmoothWeight']



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`."""

        if (self.Wg is None or self.mu == 0) and self.gamma == 0:
            # Skip unnecessary inhibition steps and run standard CBPDN
            super(ConvBPDNInhib, self).ystep()

        else:
            # Perform soft-thresholding step of Y subproblem using l1 weights
            self.Y = sp.prox_l1(self.AX + self.U,
                                (self.lmbda * self.wl1 + self.mu * self.wml +
                                 self.gamma * self.wms) / self.rho)

            # Compute the frequency domain representation of the
            # magnitude of X
            Xaf = rfftn(np.abs(self.X), self.cri.Nv, self.cri.axisN)

            if self.mu > 0 and self.Wg is not None:
                # Update previous lateral inhibition term
                self.wml_prev = self.wml
                # Convolve the lateral spatial weighting matrix with the
                # magnitude of X
                WhXal = irfftn(self.Whfl * Xaf, self.cri.Nv, self.cri.axisN)
                # Sum the weights across in-group members for each element
                self.wml = np.dot(np.dot(WhXal, self.Wg.T),
                                  self.Wg) - np.sum(self.Wg, axis=0) * WhXal
                # Smooth lateral inhibition term
                self.wml = self.smooth * self.wml_prev + \
                    (1 - self.smooth) * self.wml

            if self.gamma > 0:
                # Update previous self inhibition term
                self.wms_prev = self.wms
                # Convolve the self spatial weighting matrix with the
                # magnitude of X
                self.wms = irfftn(
                    self.Whfs * Xaf, self.cri.Nv, self.cri.axisN)
                # Smooth self inhibition term
                self.wms = self.smooth * self.wms_prev + \
                    (1 - self.smooth) * self.wms

            # Handle negative coefficients and boundary crossings
            super(cbpdn.ConvBPDN, self).ystep()



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        # Sparsity term
        rl = np.linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        # Lateral inhibition term
        rm = np.linalg.norm((self.wml * self.obfn_gvar()).ravel(), 1)
        # Self inhibition term
        rg = np.linalg.norm((self.wms * self.obfn_gvar()).ravel(), 1)
        return (self.lmbda * rl + self.mu * rm + self.gamma * rg, rl, rm, rg)
