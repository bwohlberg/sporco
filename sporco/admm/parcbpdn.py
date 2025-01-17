# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 by Erik Skau <ewskau@gmail.com>
#                            Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Parallel ADMM algorithm for Convolutional BPDN"""

from __future__ import division, absolute_import, print_function
from builtins import range

import platform
if platform.system() == 'Windows' or platform.system() == 'Darwin':
    raise RuntimeError('Module %s is not supported under Windows or '
                       'MacOS' % __name__)
import copy
import multiprocessing as mp
import numpy as np

import sporco.linalg as sl
import sporco.prox as sp
from sporco.util import u
from sporco.admm.cbpdn import GenericConvBPDN
import sporco.fft
from sporco.fft import rfftn, irfftn
import sporco.cnvrep as cr
# Required due to pyFFTW bug #135 - see "Notes" section of SPORCO docs.
sporco.fft.pyfftw_threads = 1



__all__ = ['ParConvBPDN']


# Initialise global variables required and used my multiprocessing

# Conv Rep Indexing and parameter values for multiprocessing
mp_nproc = None  # Number of processes
mp_ngrp = None  # Number of groups in the partition of M
mp_Nv = None  # Tuple of the signal dimensions
mp_axisN = None  # Axis that indexes the signal
mp_C = None  # Number of channels in the signal
mp_Cd = None  # Number of channels in the dictionary
mp_axisC = None  # Axis that indexes the channels
mp_axisM = None  # Axis that indexes the filters
mp_Dshp = None  # Shape of the dictionary D
mp_NonNegCoef = None  # Flag for non neg coef option
mp_NoBndryCross = None  # Flag for no boundary crossing

# Parameters for optimization
mp_lmbda = None   # Regularisation parameter lambda
mp_rls = None  # Relaxation parameter
mp_rho = None  # Penalty parameter of splits
mp_alpha = None  # scaling factor for X=Y1 relative to DX=Y0
mp_wl1 = None  # L1Weight matrix or scalar

# Matrices used in optimization
mp_S = None  # Training data array
mp_Df = None  # Dictionary variable (in DFT domain) used by X step
mp_X = None  # Sparsity variable
mp_Xnr = None  # Sparsity variable
mp_Y0 = None  # Split variable DX=Y0
mp_Y1 = None  # Split variable X=Y1
mp_U0 = None  # Lagrange multiplier of DX=Y0
mp_U1 = None  # Lagrange multiplier of X=Y1
mp_DX = None  # DX in spatial domain
mp_DXnr = None  # DX in spatial domain

# Variables used to solve the optimization efficiently
mp_inv_off_diag = None  # The off diagonal element of inverse matrix off
                        # the Y0 update
mp_b = None   # The rhs of the Y0 step equation, calculated in serial,
              # used in parallel
mp_grp = None  # A list of indices that partition M into approximately
               # L groups
mp_cache = None  # The cached component for solvedbi_sm for the X step

# Residual and stopping criteria variables
mp_ry0 = None  # Primal residual components of Y0
mp_ry1 = None  # Primal residual components of Y1
mp_sy0 = None  # Dual residual components of Y0
mp_sy1 = None  # Dual residual components of Y1
mp_nrmAx = None  # Components of norm of AX for computing epsilon primal
mp_nrmBy = None  # Components of norm of BY for computing epsilon primal
mp_nrmu = None  # Components of norm of U for computing epsilon dual


def mpraw_as_np(shape, dtype):
    """Construct a numpy array of the specified shape and dtype for
    which the underlying storage is a multiprocessing RawArray in shared
    memory.

    Parameters
    ----------
    shape : tuple
      Shape of numpy array
    dtype : data-type
      Data type of array

    Returns
    -------
    arr : ndarray
      Numpy array
    """

    sz = int(np.prod(shape))
    csz = sz * np.dtype(dtype).itemsize
    raw = mp.RawArray('c', csz)
    return np.frombuffer(raw, dtype=dtype, count=sz).reshape(shape)



def init_mpraw(mpv, npv):
    """Set a global variable as a multiprocessing RawArray in shared
    memory with a numpy array wrapper and initialise its value.

    Parameters
    ----------
    mpv : string
      Name of global variable to set
    npv : ndarray
      Numpy array to use as initialiser for global variable value
    """

    globals()[mpv] = mpraw_as_np(npv.shape, npv.dtype)
    globals()[mpv][:] = npv



def par_xstep(i):
    r"""Minimise Augmented Lagrangian with respect to
    :math:`\mathbf{x}_{G_i}`, one of the disjoint problems of optimizing
    :math:`\mathbf{x}`.

    Parameters
    ----------
    i : int
      Index of grouping to update

    """
    global mp_X
    global mp_DX
    YU0f = rfftn(mp_Y0[[i]] - mp_U0[[i]], mp_Nv, mp_axisN)
    YU1f = rfftn(mp_Y1[mp_grp[i]:mp_grp[i+1]] -
                    1/mp_alpha*mp_U1[mp_grp[i]:mp_grp[i+1]], mp_Nv, mp_axisN)
    if mp_Cd == 1:
        b = np.conj(mp_Df[mp_grp[i]:mp_grp[i+1]]) * YU0f + mp_alpha**2*YU1f
        Xf = sl.solvedbi_sm(mp_Df[mp_grp[i]:mp_grp[i+1]], mp_alpha**2, b,
                            mp_cache[i], axis=mp_axisM)
    else:
        b = sl.inner(np.conj(mp_Df[mp_grp[i]:mp_grp[i+1]]), YU0f,
                     axis=mp_C) + mp_alpha**2*YU1f
        Xf = sl.solvemdbi_ism(mp_Df[mp_grp[i]:mp_grp[i+1]], mp_alpha**2, b,
                              mp_axisM, mp_axisC)
    mp_X[mp_grp[i]:mp_grp[i+1]] = irfftn(Xf, mp_Nv,
                                            mp_axisN)
    mp_DX[i] = irfftn(sl.inner(mp_Df[mp_grp[i]:mp_grp[i+1]], Xf,
                                  mp_axisM), mp_Nv, mp_axisN)



def par_relax_AX(i):
    """Parallel implementation of relaxation if option ``RelaxParam`` !=
    1.0.
    """

    global mp_X
    global mp_Xnr
    global mp_DX
    global mp_DXnr
    mp_Xnr[mp_grp[i]:mp_grp[i+1]] = mp_X[mp_grp[i]:mp_grp[i+1]]
    mp_DXnr[i] = mp_DX[i]
    if mp_rlx != 1.0:
        grpind = slice(mp_grp[i], mp_grp[i+1])
        mp_X[grpind] = mp_rlx * mp_X[grpind] + (1-mp_rlx)*mp_Y1[grpind]
        mp_DX[i] = mp_rlx*mp_DX[i] + (1-mp_rlx)*mp_Y0[i]



def y0astep():
    r"""The serial component of the step to minimise the augmented
    Lagrangian with respect to :math:`\mathbf{y}_0`.
    """

    global mp_b
    mp_b[:] = mp_inv_off_diag * np.sum((mp_S + mp_rho*(mp_DX+mp_U0)),
                                       axis=mp_axisM, keepdims=True)



def par_y0bstep(i):
    r"""The parallel component of the step to minimise the augmented
    Lagrangian with respect to :math:`\mathbf{y}_0`.

    Parameters
    ----------
    i : int
      Index of grouping to update
    """

    global mp_Y0
    mp_Y0[i] = 1/mp_rho*mp_S + mp_DX[i] + mp_U0[i] + mp_b



def par_y1step(i):
    r"""Minimise Augmented Lagrangian with respect to
    :math:`\mathbf{y}_{1,G_i}`, one of the disjoint problems of
    optimizing :math:`\mathbf{y}_1`.

    Parameters
    ----------
    i : int
      Index of grouping to update
    """

    global mp_Y1
    grpind = slice(mp_grp[i], mp_grp[i+1])
    XU1 = mp_X[grpind] + 1/mp_alpha*mp_U1[grpind]
    if mp_wl1.shape[mp_axisM] == 1:
        gamma = mp_lmbda/(mp_alpha**2*mp_rho)*mp_wl1
    else:
        gamma = mp_lmbda/(mp_alpha**2*mp_rho)*mp_wl1[grpind]
    Y1 = sp.prox_l1(XU1, gamma)
    if mp_NonNegCoef:
        Y1[Y1 < 0.0] = 0.0
    if mp_NoBndryCross:
        for n in range(len(mp_Nv)):
            Y1[(slice(None),) + (slice(None),)*n +
               (slice(1-mp_Dshp[n], None),)] = 0.0
    mp_Y1[mp_grp[i]:mp_grp[i+1]] = Y1



def par_u0step(i):
    r"""Dual variable update for :math:`\mathbf{u}_{0,i}`, one of the
    disjoint problems for updating :math:`\mathbf{u}_0`.

    Parameters
    ----------
    i : int
      Index of grouping to update
    """

    global mp_U0
    mp_U0[i] += mp_DX[i] - mp_Y0[i]


def par_u1step(i):
    r"""Dual variable update for :math:`\mathbf{u}_{1,G_i}`, one of the
    disjoint problems for updating :math:`\mathbf{u}_1`.

    Parameters
    ----------
    i : int
      Index of grouping to update
    """

    global mp_U1
    grpind = slice(mp_grp[i], mp_grp[i+1])
    mp_U1[grpind] += mp_alpha*(mp_X[grpind] - mp_Y1[grpind])



def par_initial_stepgrp(i):
    """The parallel step grouping of the initial iteration in solve. A
    cyclic permutation of the steps is done to require only one merge
    per iteration, requiring unique initial and final step groups.

    Parameters
    ----------
    i : int
      Index of grouping to update
    """

    par_xstep(i)
    par_relax_AX(i)



def par_stepgrp(i):
    """The parallel step grouping of internal (not initial or final)
    iterations in solve. A cyclic permutation of the steps is done to
    require only one merge per iteration, requiring unique initial and
    final step groups.

    Parameters
    ----------
    i : int
      Index of grouping to update
    """

    par_final_stepgrp(i)
    par_initial_stepgrp(i)



def par_final_stepgrp(i):
    """The parallel step grouping of the final iteration in solve. A
    cyclic permutation of the steps is done to require only one merge
    per iteration, requiring unique initial and final step groups.

    Parameters
    ----------
    i : int
      Index of grouping to update
    """

    par_y0bstep(i)
    par_y1step(i)
    par_u0step(i)
    par_u1step(i)



def par_compute_residuals(i):
    """Compute components of the residual and stopping thresholds that
    can be done in parallel.

    Parameters
    ----------
    i : int
      Index of group to compute
    """

    # Compute the residuals in parallel, need to check if the residuals
    # depend on alpha
    global mp_ry0
    global mp_ry1
    global mp_sy0
    global mp_sy1
    global mp_nrmAx
    global mp_nrmBy
    global mp_nrmu
    mp_ry0[i] = np.sum((mp_DXnr[i] - mp_Y0[i])**2)
    mp_ry1[i] = mp_alpha**2*np.sum((mp_Xnr[mp_grp[i]:mp_grp[i+1]]-
                                    mp_Y1[mp_grp[i]:mp_grp[i+1]])**2)
    mp_sy0[i] = np.sum((mp_Y0old[i] - mp_Y0[i])**2)
    mp_sy1[i] = mp_alpha**2*np.sum((mp_Y1old[mp_grp[i]:mp_grp[i+1]]-
                                    mp_Y1[mp_grp[i]:mp_grp[i+1]])**2)
    mp_nrmAx[i] = np.sum(mp_DXnr[i]**2) + mp_alpha**2 * np.sum(
        mp_Xnr[mp_grp[i]:mp_grp[i+1]]**2)
    mp_nrmBy[i] = np.sum(mp_Y0[i]**2) + mp_alpha**2 * np.sum(
        mp_Y1[mp_grp[i]:mp_grp[i+1]]**2)
    mp_nrmu[i] = np.sum(mp_U0[i]**2) + np.sum(mp_U1[mp_grp[i]:mp_grp[i+1]]**2)





class ParConvBPDN(GenericConvBPDN):
    r"""
    Parallel ADMM algorithm for Convolutional BPDN (CBPDN) with or
    without a spatial mask :cite:`skau-2018-fast`.

    |

    .. inheritance-diagram:: ParConvBPDN
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s}\right) \right\|_2^2 + \lambda \sum_m
       \| \mathbf{x}_m \|_1 \;\;,

    where :math:`W` is a mask array, via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
       (1/2) \| W \left( \sum_l \mathbf{y}_{0,l} - \mathbf{s} \right)
       \|_2^2 + \lambda \| \mathbf{y}_1 \|_1 \;\text{such that}\;
       \left( \begin{array}{c} D_{G_0} \\ \vdots \\ D_{G_{L-1}} \\
       \alpha I \end{array} \right) \mathbf{x} - \left( \begin{array}{c}
       \mathbf{y}_{0,0} \\ \vdots \\ \mathbf{y}_{0,L-1} \\ \alpha
       \mathbf{y}_1 \end{array} \right) = \left( \begin{array}{c}
       \mathbf{0} \\ \vdots \\ \mathbf{0} \\ \mathbf{0} \end{array}
       \right) \;\;,

    where the :math:`M` dictionary filters are partitioned into
    :math:`L` groups, :math:`\{G_l\}_{l \in \{0,\dots,L-1\}}` where

    .. math::
       G_i \cap G_j = \emptyset \text{ for } i \neq j \text{
       and } \bigcup_l G_l = \{0, \dots, M-1\} \;,

    and :math:`D_{G_l}` is a linear operator such that :math:`D_{G_l}
    \mathbf{x} = \sum_{g \in G_l} \mathbf{d}_g * \mathbf{x}_g`.

    Multi-image and multi-channel problems are also supported. The
    multi-image problem is

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_k \left\| W_k \left( \sum_m \mathbf{d}_m *
       \mathbf{x}_{k,m} - \mathbf{s}_k \right) \right\|_2^2 + \lambda
       \sum_k \sum_m \| \mathbf{x}_{k,m} \|_1

    with input images :math:`\mathbf{s}_k`, masks :math:`W_k`, and
    coefficient maps :math:`\mathbf{x}_{k,m}`. The multi-channel
    problem with input image channels :math:`\mathbf{s}_c` and a
    multi-channel mask :math:`W_c` is either

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_c \left\| W_c \left( \sum_m \mathbf{d}_m *
       \mathbf{x}_{c,m} - \mathbf{s}_c \right) \right\|_2^2 +
       \lambda \sum_c \sum_m \| \mathbf{x}_{c,m} \|_1

    with single-channel dictionary filters :math:`\mathbf{d}_m` and
    multi-channel coefficient maps :math:`\mathbf{x}_{c,m}`, or

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_c \left\| W_c \left( \sum_m \mathbf{d}_{c,m} *
       \mathbf{x}_m - \mathbf{s}_c \right) \right\|_2^2 + \lambda
       \sum_m \| \mathbf{x}_m \|_1

    with multi-channel dictionary filters :math:`\mathbf{d}_{c,m}` and
    single-channel coefficient maps :math:`\mathbf{x}_m`.

    After termination of the :meth:`solve` method, AttributeError
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| W \left(
       \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \right) \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_m \|
       \mathbf{x}_m \|_1`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Not Implemented (relative residual of X step solver)

       ``Time`` : Cumulative run time
    """

    class Options(GenericConvBPDN.Options):
        r"""ParConvBPDN algorithm options

        Options include all of those defined in
        :class:`.admm.ADMMEqual.Options`, together with additional options:

          ``alpha`` : A float indicating the relative weight between
          the constraint :math:`D_{G_l} \mathbf{x} = \mathbf{y}_{0,l}`
          and :math:`\alpha \mathbf{x} = \mathbf{y}_1`. None value
          effectively defaults to no weight or :math:`\alpha = 1`.

          ``Y0`` : Initial value for :math:`\mathbf{y}_0`.

          ``U0`` : Initial value for :math:`\mathbf{u}_0`.

          ``Y1`` : Initial value for :math:`\mathbf{y}_1`.

          ``U1`` : Initial value for :math:`\mathbf{u}_1`.


        and the exceptions:

          ``AutoRho`` : Not implemented.

          ``LinSolveCheck`` : Not implemented.

        """
        defaults = copy.deepcopy(GenericConvBPDN.Options.defaults)
        defaults.update({'L1Weight': 1.0, 'alpha': None, 'Y1': None,
                         'U1': None})

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
               ParConvBPDN algorithm options
            """

            if opt is None:
                opt = {}
            GenericConvBPDN.Options.__init__(self, opt)


    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regl1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regl1'): 'RegL1'}


    def __init__(self, D, S, lmbda=None, W=None, opt=None, nproc=None,
                 ngrp=None, dimK=None, dimN=2):
        """
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
        opt : :class:`ParConvBPDN.Options` object
          Algorithm options
        nproc : int
          Number of processes
        ngrp : int
          Number of groups in partition of filter indices
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial dimensions
        """

        self.pool = None

        # Set default options if none specified
        if opt is None:
            opt = ParConvBPDN.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        # Set default lambda value if not specified
        if lmbda is None:
            cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)
            Df = rfftn(D.reshape(cri.shpD), cri.Nv, axes=cri.axisN)
            Sf = rfftn(S.reshape(cri.shpS), axes=cri.axisN)
            b = np.conj(Df) * Sf
            lmbda = 0.1*abs(b).max()

        # Set l1 term scaling and weight array
        self.lmbda = self.dtype.type(lmbda)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(50.0*self.lmbda + 1.0),
                      dtype=self.dtype)
        self.set_attr('alpha', opt['alpha'], dval=1.0,
                      dtype=self.dtype)

        # Set rho_xi attribute (see Sec. VI.C of wohlberg-2015-adaptive)
        # if self.lmbda != 0.0:
        #     rho_xi = (1.0 + (18.3)**(np.log10(self.lmbda) + 1.0))
        # else:
        #     rho_xi = 1.0
        # self.set_attr('rho_xi', opt['AutoRho', 'RsdlTarget'], dval=rho_xi,
        #               dtype=self.dtype)

        # Call parent class __init__
        super(ParConvBPDN, self).__init__(D, S, opt, dimK, dimN)

        if nproc is None:
            if ngrp is None:
                self.nproc = min(16, mp.cpu_count(), self.cri.M)
                self.ngrp = self.nproc
            else:
                self.nproc = min(mp.cpu_count(), ngrp, self.cri.M)
                self.ngrp = ngrp
        else:
            if ngrp is None:
                self.ngrp = nproc
                self.nproc = nproc
            else:
                self.ngrp = ngrp
                self.nproc = nproc

        if W is None:
            W = np.array([1.0], dtype=self.dtype)
        self.W = np.asarray(W.reshape(cr.mskWshape(W, self.cri)),
                            dtype=self.dtype)
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)
        self.wl1 = self.wl1.reshape(cr.l1Wshape(self.wl1, self.cri))

        self.xrrs = None

        # Initialise global variables
        # Conv Rep Indexing and parameter values for multiprocessing
        global mp_nproc
        mp_nproc = self.nproc
        global mp_ngrp
        mp_ngrp = self.ngrp
        global mp_Nv
        mp_Nv = self.cri.Nv
        global mp_axisN
        mp_axisN = tuple(i+1 for i in self.cri.axisN)
        global mp_C
        mp_C = self.cri.C
        global mp_Cd
        mp_Cd = self.cri.Cd
        global mp_axisC
        mp_axisC = self.cri.axisC+1
        global mp_axisM
        mp_axisM = 0
        global mp_NonNegCoef
        mp_NonNegCoef = self.opt['NonNegCoef']
        global mp_NoBndryCross
        mp_NoBndryCross = self.opt['NoBndryCross']
        global mp_Dshp
        mp_Dshp = self.D.shape

        # Parameters for optimization
        global mp_lmbda
        mp_lmbda = self.lmbda
        global mp_rho
        mp_rho = self.rho
        global mp_alpha
        mp_alpha = self.alpha
        global mp_rlx
        mp_rlx = self.rlx
        global mp_wl1
        init_mpraw('mp_wl1', np.moveaxis(self.wl1, self.cri.axisM, mp_axisM))

        # Matrices used in optimization
        global mp_S
        init_mpraw('mp_S', np.moveaxis(self.S*self.W**2, self.cri.axisM,
                                       mp_axisM))
        global mp_Df
        init_mpraw('mp_Df', np.moveaxis(self.Df, self.cri.axisM, mp_axisM))
        global mp_X
        init_mpraw('mp_X', np.moveaxis(self.Y, self.cri.axisM, mp_axisM))
        shp_X = list(mp_X.shape)
        global mp_Xnr
        mp_Xnr = mpraw_as_np(mp_X.shape, mp_X.dtype)
        global mp_Y0
        shp_Y0 = shp_X[:]
        shp_Y0[0] = self.ngrp
        shp_Y0[mp_axisC] = mp_C
        if self.opt['Y0'] is not None:
            init_mpraw('Y0', np.moveaxis(
                self.opt['Y0'].astype(self.dtype, copy=True),
                self.cri.axisM, mp_axisM))
        else:
            mp_Y0 = mpraw_as_np(shp_Y0, mp_X.dtype)
        global mp_Y0old
        mp_Y0old = mpraw_as_np(shp_Y0, mp_X.dtype)
        global mp_Y1
        if self.opt['Y1'] is not None:
            init_mpraw('Y1', np.moveaxis(
                self.opt['Y1'].astype(self.dtype, copy=True),
                self.cri.axisM, mp_axisM))
        else:
            mp_Y1 = mpraw_as_np(shp_X, mp_X.dtype)
        global mp_Y1old
        mp_Y1old = mpraw_as_np(shp_X, mp_X.dtype)
        global mp_U0
        if self.opt['U0'] is not None:
            init_mpraw('U0', np.moveaxis(
                self.opt['U0'].astype(self.dtype, copy=True),
                self.cri.axisM, mp_axisM))
        else:
            mp_U0 = mpraw_as_np(shp_Y0, mp_X.dtype)
        global mp_U1
        if self.opt['U1'] is not None:
            init_mpraw('U1', np.moveaxis(
                self.opt['U1'].astype(self.dtype, copy=True),
                self.cri.axisM, mp_axisM))
        else:
            mp_U1 = mpraw_as_np(shp_X, mp_X.dtype)
        global mp_DX
        mp_DX = mpraw_as_np(shp_Y0, mp_X.dtype)
        global mp_DXnr
        mp_DXnr = mpraw_as_np(shp_Y0, mp_X.dtype)

        # Variables used to solve the optimization efficiently
        global mp_inv_off_diag
        if self.W.ndim is self.cri.axisM+1:
            init_mpraw('mp_inv_off_diag', np.moveaxis(
                -self.W**2/(mp_rho*(mp_rho+self.W**2*mp_ngrp)),
                self.cri.axisM, mp_axisM))
        else:
            init_mpraw('mp_inv_off_diag',
                       -self.W**2/(mp_rho*(mp_rho+self.W**2*mp_ngrp)))
        global mp_grp
        mp_grp = [np.min(i) for i in
                  np.array_split(np.array(range(self.cri.M)),
                                 mp_ngrp)] + [self.cri.M, ]
        global mp_cache
        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            mp_cache = [sl.solvedbi_sm_c(mp_Df[k], np.conj(mp_Df[k]),
                                         mp_alpha**2, mp_axisM) for k in
                        np.array_split(np.array(range(self.cri.M)), self.ngrp)]
        else:
            mp_cache = [None for k in mp_grp]
        global mp_b
        shp_b = shp_Y0[:]
        shp_b[0] = 1
        mp_b = mpraw_as_np(shp_b, mp_X.dtype)

        # Residual and stopping criteria variables
        global mp_ry0
        mp_ry0 = mpraw_as_np((self.ngrp,), mp_X.dtype)
        global mp_ry1
        mp_ry1 = mpraw_as_np((self.ngrp,), mp_X.dtype)
        global mp_sy0
        mp_sy0 = mpraw_as_np((self.ngrp,), mp_X.dtype)
        global mp_sy1
        mp_sy1 = mpraw_as_np((self.ngrp,), mp_X.dtype)
        global mp_nrmAx
        mp_nrmAx = mpraw_as_np((self.ngrp,), mp_X.dtype)
        global mp_nrmBy
        mp_nrmBy = mpraw_as_np((self.ngrp,), mp_X.dtype)
        global mp_nrmu
        mp_nrmu = mpraw_as_np((self.ngrp,), mp_X.dtype)



    def solve(self):
        """Start (or re-start) optimisation. This method implements the
        framework for the iterations of an ADMM algorithm.

        If option ``Verbose`` is ``True``, the progress of the
        optimisation is displayed at every iteration. At termination
        of this method, attribute :attr:`itstat` is a list of tuples
        representing statistics of each iteration, unless option
        ``FastSolve`` is ``True`` and option ``Verbose`` is ``False``.

        Attribute :attr:`timer` is an instance of :class:`.util.Timer`
        that provides the following labelled timers:

          ``init``: Time taken for object initialisation by
          :meth:`__init__`

          ``solve``: Total time taken by call(s) to :meth:`solve`

          ``solve_wo_func``: Total time taken by call(s) to
          :meth:`solve`, excluding time taken to compute functional
          value and related iteration statistics

          ``solve_wo_rsdl`` : Total time taken by call(s) to
          :meth:`solve`, excluding time taken to compute functional
          value and related iteration statistics as well as time take
          to compute residuals and implemented ``AutoRho`` mechanism
        """

        global mp_Y0old
        global mp_Y1old

        self.init_pool()

        fmtstr, nsep = self.display_start()

        # Start solve timer
        self.timer.start(['solve', 'solve_wo_func', 'solve_wo_rsdl'])

        first_iteration = self.k
        last_iteration = self.k + self.opt['MaxMainIter'] - 1
        # Main optimisation iterations
        for self.k in range(self.k, self.k + self.opt['MaxMainIter']):
            mp_Y0old[:] = np.copy(mp_Y0)
            mp_Y1old[:] = np.copy(mp_Y1)

            # Perform the variable updates.
            if self.k is first_iteration:
                self.distribute(par_initial_stepgrp, mp_ngrp)
            y0astep()
            if self.k is last_iteration:
                self.distribute(par_final_stepgrp, mp_ngrp)
            else:
                self.distribute(par_stepgrp, mp_ngrp)

            # Compute the residual variables
            self.timer.stop('solve_wo_rsdl')
            if self.opt['AutoRho', 'Enabled'] or not self.opt['FastSolve']:
                self.distribute(par_compute_residuals, mp_ngrp)
                r = np.sqrt(np.sum(mp_ry0) + np.sum(mp_ry1))
                s = np.sqrt(np.sum(mp_sy0) + np.sum(mp_sy1))

                epri = np.sqrt(self.Nc) * self.opt['AbsStopTol'] + \
                  np.max([np.sqrt(np.sum(mp_nrmAx)),
                          np.sqrt(np.sum(mp_nrmBy))]) * self.opt['RelStopTol']

                edua = np.sqrt(self.Nx) * self.opt['AbsStopTol'] + \
                  np.sqrt(np.sum(mp_nrmu)) * self.opt['RelStopTol']

            # Compute and record other iteration statistics and
            # display iteration stats if Verbose option enabled
            self.timer.stop(['solve_wo_func', 'solve_wo_rsdl'])
            if not self.opt['FastSolve']:
                itst = self.iteration_stats(self.k, r, s, epri, edua)
                self.itstat.append(itst)
                self.display_status(fmtstr, itst)
            self.timer.start(['solve_wo_func', 'solve_wo_rsdl'])

            # Automatic rho adjustment
            # self.timer.stop('solve_wo_rsdl')
            # if self.opt['AutoRho', 'Enabled'] or not self.opt['FastSolve']:
            #     self.update_rho(self.k, r, s)
            # self.timer.start('solve_wo_rsdl')

            # Call callback function if defined
            if self.opt['Callback'] is not None:
                if self.opt['Callback'](self):
                    break

            # Stop if residual-based stopping tolerances reached
            if self.opt['AutoRho', 'Enabled'] or not self.opt['FastSolve']:
                if r < epri and s < edua:
                    break

        # Increment iteration count
        self.k += 1

        # Record solve time
        self.timer.stop(['solve', 'solve_wo_func', 'solve_wo_rsdl'])

        # Print final separator string if Verbose option enabled
        self.display_end(nsep)

        self.Y = np.moveaxis(mp_Y1, mp_axisM, self.cri.axisM)
        self.X = np.moveaxis(mp_X, mp_axisM, self.cri.axisM)

        self.terminate_pool()

        return self.getmin()



    def init_pool(self):
        """Initialize multiprocessing pool if necessary."""

        # initialize the pool if needed
        if self.pool is None:
            if self.nproc > 1:
                self.pool = mp.Pool(processes=self.nproc)
            else:
                self.pool = None
        else:
            print('pool already initialized?')



    def distribute(self, f, n):
        """Distribute the computations amongst the multiprocessing pools

        Parameters
        ----------
        f : function
          Function to be distributed to the processors
        n : int
          The values in range(0,n) will be passed as arguments to the
          function f.
        """

        if self.pool is None:
            return [f(i) for i in range(n)]
        else:
            return self.pool.map(f, range(n))



    def terminate_pool(self):
        """Terminate and close the multiprocessing pool if necessary."""

        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            del(self.pool)
            self.pool = None



    def obfn_gvar(self):
        """Variable to be evaluated in computing :meth:`ADMM.obfn_g`,
        depending on the ``gEvalY`` option value.
        """

        return mp_Y1 if self.opt['gEvalY'] else mp_X



    def obfn_fvar(self):
        """Variable to be evaluated in computing :meth:`ADMM.obfn_f`,
        depending on the ``fEvalX`` option value.
        """
        return mp_X if self.opt['fEvalX'] else mp_Y1



    def obfn_reg(self):
        r"""Compute regularisation term, :math:`\| x \|_1`, and
        contribution to objective function.
        """
        l1 = np.sum(mp_wl1*np.abs(self.obfn_gvar()))
        return (self.lmbda*l1, l1)



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| W \left( \sum_m
        \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \right) \|_2^2`.
        """
        XF = rfftn(self.obfn_fvar(), mp_Nv, mp_axisN)
        DX = np.moveaxis(irfftn(sl.inner(mp_Df, XF, mp_axisM),
                                   mp_Nv, mp_axisN), mp_axisM,
                         self.cri.axisM)
        return np.sum((self.W*(DX-self.S))**2)/2.0
