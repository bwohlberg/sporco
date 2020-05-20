# -*- coding: utf-8 -*-
# Copyright (C) 2016-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithms for convolutional sparse coding with
Lateral Inhibition terms"""

from __future__ import division, print_function
from builtins import range

import copy
import numpy as np

from sporco.admm import cbpdn
import sporco.linalg as sl
import sporco.prox as sp
from scipy import signal


__author__ = """Frank Cwitkowitz <fcwitkow@ur.rochester.edu>"""


class ConvBPDNLatInh(cbpdn.ConvBPDN):
    r"""
    TODO - this block comment will need to be updated
    ADMM algorithm for Convolutional BPDN with lateral inhibition via a
    weighted :math:`\ell_{1}` norm term :cite:`wohlberg-2016-convolutional2`

    |

    .. inheritance-diagram:: ConvBPDNLatInh
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_c \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{c,m} -
       \mathbf{s}_c \right\|_2^2 + \lambda \sum_c \sum_m
       \| \mathbf{x}_{c,m} \|_1 + \mu \| \{ \mathbf{x}_{c,m} \} \|_{2,1}

    with input images :math:`\mathbf{s}_c`, dictionary filters
    :math:`\mathbf{d}_m`, and coefficient maps
    :math:`\mathbf{x}_{c,m}`, via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \sum_c \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{c,m} -
       \mathbf{s}_c \right\|_2^2 + \lambda \sum_c \sum_m
       \| \mathbf{y}_{c,m} \|_1 + \mu \| \{ \mathbf{y}_{c,m} \} \|_{2,1}
       \quad \text{such that} \quad \mathbf{x}_{c,m} = \mathbf{y}_{c,m} \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term :math:`(1/2) \sum_c
       \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} - \mathbf{s}_c
       \right\|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_c \sum_m
       \| \mathbf{x}_{c,m} \|_1`

       ``RegL21`` : Value of regularisation term :math:`\| \{
       \mathbf{x}_{c,m} \} \|_{2,1}`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual Residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """


    class Options(cbpdn.ConvBPDN.Options):
        r"""ConvBPDNLatInh algorithm options
        TODO - this block comment will need to be updated

        Options include all of those defined in
        :class:`.cbpdn.ConvBPDN.Options`, together with additional options:

          ``LISmWeight`` : Smoothing for the weighted :math:`\ell_1`
          norm (lateral inhibition). The value acts as the percentage of the
          previous value which is added to the.
        """

        defaults = copy.deepcopy(cbpdn.ConvBPDN.Options.defaults)
        defaults.update({'LISmWeight': 0.9})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDNLatInh algorithm options
            """

            if opt is None:
                opt = {}
            cbpdn.ConvBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'l', 'm', 'g')
    hdrtxt_objfn = ('Fnc', 'DFid', 'l', 'm', 'g')
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', 'l': 'l', 'm': 'm', 'g': 'g'}



    def __init__(self, D, S, Wg=None, Whn=2205, win_args=('tukey', 0.5), lmbda=None, mu=None, gamma=None, opt=None, dimK=None, dimN=2):
        """
        TODO - this block comment will need to be updated
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

        .. image:: ../_static/jonga/cbpdn_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdn_init.svg

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
            opt = ConvBPDNLatInh.Options()

        # Call parent class __init__
        super(ConvBPDNLatInh, self).__init__(D, S, lmbda, opt, dimK, dimN)

        # Add the groups to the class instance
        self.Wg = Wg

        # Set default lateral inhibition scaling term
        if mu is None:
            mu = 10 * self.lmbda

        # Set weighted l1 (lateral inhibition) term scaling
        self.mu = self.dtype.type(mu)

        # Set default self inhibition scaling term
        if gamma is None:
            # No self inhibition by default
            gamma = 0.0

        # Set weighted l1 (self inhibition) term scaling
        self.gamma = self.dtype.type(gamma)

        self.wml, self.wms = 0, 0

        # Check to see if a grouping scheme is provided. If not, we assume
        # each filter is independent, i.e. there is no grouping and no inhibition.
        # If no grouping scheme and no gamma, nothing indicates inhibition, and we skip the following.
        if (self.Wg is not None and self.mu != 0) or self.gamma:

            # Set the type of the grouping weights
            self.Wg = self.Wg.astype(self.dtype)

            # Add the number of groups to the indexer
            self.cri.Ng = self.Wg.shape[0]

            # Determine the number of filters per group and add to the indexer
            self.cri.Mgs = np.sum((self.Wg != 0), axis=1)

            # Initialize the grouping matrix
            self.cmn = np.zeros((self.cri.M, self.cri.M), dtype=self.dtype)

            # Obtain the grouping matrix c_{m,n}, where:
            #     c_{m,n} = 1 if m != n and filter m and n belong to the same group,
            #             = 0 otherwise.
            for g in range(self.cri.Ng):
                temp_groups = np.zeros(self.cmn.shape, dtype=self.dtype)
                mmbr_indces = (self.Wg[g] != 0)
                temp_groups[mmbr_indces, :] = self.Wg[g]
                self.cmn += temp_groups

            # Remove self-grouping
            self.cmn = self.cmn * (1 - np.eye(self.cri.M))

            # Protect against same pair in multiple groups
            self.cmn[self.cmn != 0] = 1

            # Obtain a matrix of the indices where c_{m,n} is non-zero, i.e. the filter
            # ids n of those which belong to the same group as filter m where m != n
            self.cmni = self.cmn == 1

            # Create generalized spatial weighting matrix. This matrix is convolved with
            # weights during lateral inhibition to enforce locality of inhibition.
            Whl = np.zeros(self.cri.shpS, dtype=self.dtype)
            # Ensure inhibition sample length is odd for symmetric inhibition
            Whn += not Whn % 2
            # Create N-dimensional window function
            nDimInd = tuple(np.meshgrid(*([np.arange(Whn)]*dimN)))
            nDimWin = np.meshgrid(*([np.array(signal.get_window(win_args, Whn))]*dimN))
            nDimWin = np.concatenate([np.expand_dims(nDimWin[i], axis=0) for i in range(len(nDimWin))])
            nDimWin = np.power(np.prod(nDimWin, axis=0), 1/dimN)
            Whl[nDimInd] = np.reshape(nDimWin, Whl[nDimInd].shape)
            # Center window around origin in each dimension
            for i in range(dimN):
                Whl = np.roll(Whl, -Whn//2+1, axis=i)
            # Create a spatial weighting matrix for self inhibition (Zero-out t=0)
            Whs = Whl.copy()
            Whs[tuple([0]*dimN)] = 0

            # Obtain the lateral and self inhibition window frequency representations
            self.Whfl = sl.rfftn(Whl, self.cri.Nv, self.cri.axisN)
            self.Whfs = sl.rfftn(Whs, self.cri.Nv, self.cri.axisN)

            # Initialize pre-summed weights for weight l1 norm (lateral inhibition)
            self.WhXal = None

            # Initialize previous values
            self.wml_prev, self.wms_prev = None, None

            # Initialize smoothing for weighted l1 terms
            self.wmsm = opt['LISmWeight']



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`."""

        if (self.Wg is None or self.mu == 0) and self.gamma == 0:
            # Skip unnecessary inhibition steps and run standard CBPDN
            super(ConvBPDNLatInh, self).ystep()

        else:
            # Perform soft-thresholding step of Y subproblem using l1 weights
            self.Y = sp.prox_l1(self.AX + self.U, (self.lmbda * self.wl1 + self.mu * self.wml + self.gamma * self.wms) / self.rho)

            # Update previous weighted l1 term
            self.wml_prev = self.wml

            # Compute the frequency domain representation of the magnitude of X
            Xaf = sl.rfftn(np.abs(self.X), self.cri.Nv, self.cri.axisN)

            # Convolve the spatial weighting matrix with the magnitude of X
            self.WhXal = sl.irfftn(self.Whfl * Xaf, self.cri.Nv, self.cri.axisN)

            # Sum the weights across in-group members for each element
            self.wml = np.dot(np.dot(self.WhXal, self.Wg.T), self.Wg) - np.sum(self.Wg, axis=0) * self.WhXal

            # Smooth weighted l1 term
            self.wml = self.wmsm * self.wml_prev + (1 - self.wmsm) * self.wml

            if self.gamma > 0:
                self.wms_prev = self.wms
                self.wms = sl.irfftn(self.Whfs * Xaf, self.cri.Nv, self.cri.axisN)
                self.wms = self.wmsm * self.wms_prev + (1 - self.wmsm) * self.wms

            # Handle negative coefficients and boundary crossings
            super(cbpdn.ConvBPDN, self).ystep()



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl = np.linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rm = np.linalg.norm((self.wml * self.obfn_gvar()).ravel(), 1)
        rg = np.linalg.norm((self.wms * self.obfn_gvar()).ravel(), 1)
        return (self.lmbda*rl + self.mu*rm + self.gamma*rg, rl, rm, rg)
