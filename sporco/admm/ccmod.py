#-*- coding: utf-8 -*-
# Copyright (C) 2015-2016 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""ADMM algorithm for the CCMOD problem"""

from __future__ import division
from __future__ import absolute_import
from builtins import range

import numpy as np
from scipy import linalg
import copy
import collections

from sporco.admm import admm
import sporco.linalg as sl

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class ConvRepIndexing(object):
    """Manage the inference of problem dimensions and the roles of ndarray
    indices for convolutional representations as in :class:`.ConvBPDN`
    and related classes.
    """

    def __init__(self, S, dsz, dimN=2, dimK=1):
        """Initialise a ConvRepIndexing object, inferring the problem
        dimensions from input dictionary and signal arrays D and S
        respectively.

        The internal data layout for S, D (X here), and X (A here) is:
        ::

          dim<0> - dim<Nds-1> : Spatial dimensions, product of N0,N1,... is N
          dim<Nds>            : C number of channels in S and D
          dim<Nds+1>          : K number of signals in S
          dim<Nds+2>          : M number of filters in D

            sptl.      chn  sig  flt
          S(N0,  N1,   C,   K,   1)
          D(N0,  N1,   C,   1,   M)   (X here)
          X(N0,  N1,   1,   K,   M)   (A here)
        """

        # Numbers of spatial, channel, and signal dimensions in
        # external X (A here) and S. These need to be calculated since
        # inputs D and S do not already have the standard data layout
        # above, i.e. singleton dimensions will not be present
        self.dimN = dimN                  # Number of spatial dimensions
        self.dimC = S.ndim - dimN - dimK  # Number of channel dimensions in S
        self.dimK = dimK                  # Number of signal dimensions in S

        # Number of channels in D and S
        if self.dimC == 1:
            self.C = S.shape[dimN]
        else:
            self.C = 1

        # Number of signals in S
        if self.dimK == 1:
            self.K = S.shape[dimN+self.dimC]
        else:
            self.K = 1

        # Number of filters
        self.dsz = dsz
        if isinstance(dsz[0], tuple):
            self.M = sum(x[self.dimN] for x in dsz)
        else:
            self.M = dsz[self.dimN]

        # Number of spatial samples
        self.Nv = S.shape[0:dimN]
        self.N = np.prod(np.array(self.Nv))

        # Axis indices for each component of S, D (X here), and X (A here)
        self.axisN = tuple(range(0, dimN))
        self.axisC = dimN
        self.axisK = dimN + 1
        self.axisM = dimN + 2

        # Shapes of internal S, D (X here), and X (A here)
        self.shpD = self.Nv + (self.C,) + (1,) + (self.M,)
        self.shpS = self.Nv + (self.C,) + (self.K,) + (1,)
        self.shpX = self.Nv + (1,) + (self.K,) + (self.M,)





class ConvCnstrMOD(admm.ADMMEqual):
    """ADMM algorithm for Convolutional Constrained MOD problem
    :cite:`wohlberg-2016-efficient` :cite:`wohlberg-2016-convolutional`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_k \\left\| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \\right\|_2^2 \quad \\text{such that} \quad
       \|\mathbf{d}_m\|_2 = 1

    via the ADMM problem

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_k \\left\| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \\right\|_2^2 + \sum_m \iota_C(\mathbf{g}_m) \quad
       \\text{such that} \quad \mathbf{d}_m = \mathbf{g}_m \;\;,

    where :math:`\iota_C(\cdot)` is the indicator function of feasible
    set :math:`C` consisting of unit-norm filters.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``DFid`` :  Value of data fidelity term \
       :math:`(1/2) \sum_k \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \|_2^2`

       ``Cnstr`` : Constraint violation measure

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance \
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance \
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``XSlvRelRes`` : Relative residual of X step solver

       ``XSlvCGIt`` : CG iterations used in X step solver

       ``Time`` : Cumulative run time
    """



    class Options(admm.ADMMEqual.Options):
        """CCMOD algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMMEqual.Options`, together with
        additional options:

        ``AuxVarObj`` : Flag indicating whether the objective function \
        should be evaluated using variable X  (``False``) or Y (``True``) \
        as its argument

        ``LinSolveCheck`` : If ``True``, compute relative residual of \
        X step solver

        ``ZeroMean`` : Flag indicating whether the solution dictionary \
        :math:`\{\mathbf{d}_m\}` should have zero-mean components

        ``LinSolve`` : Select linear solver for x step. Options are \
        ``SM`` (Sherman-Morrison) or ``CG`` (Conjugate Gradient)

        ``CG`` : CG solver options

          ``MaxIter`` : Maximum iterations

          ``StopTol`` : Stopping tolerance
        """

        defaults = copy.deepcopy(admm.ADMMEqual.Options.defaults)
        defaults.update({'AuxVarObj' : False, 'ReturnX' : False,
                        'RelaxParam' : 1.8, 'ZeroMean' : False,
                        'LinSolve' : 'SM', 'LinSolveCheck' : False,
                        'CG' : {'MaxIter' : 1000, 'StopTol' : 1e-3}})
        defaults['AutoRho'].update({'Enabled' : True, 'Period' : 1,
                                    'AutoScaling' : True, 'Scaling' : 1000,
                                    'RsdlRatio' : 1.2})


        def __init__(self, opt=None):
            """Initialise CCMOD algorithm options object."""

            if opt is None:
                opt = {}
            admm.ADMMEqual.Options.__init__(self, opt)

            if self['AuxVarObj']:
                self['fEvalX'] = False
                self['gEvalY'] = True
            else:
                self['fEvalX'] = True
                self['gEvalY'] = False

            if self['AutoRho','RsdlTarget'] is None:
                self['AutoRho','RsdlTarget'] = 1.0


        def set_K(self, K, override=False):
            """Set parameters depending on K value"""

            if override or self['rho'] is None:
                self['rho'] = np.float32(K)



    IterationStats = collections.namedtuple('IterationStats',
            ['Iter', 'DFid', 'Cnstr', 'PrimalRsdl', 'DualRsdl',
             'EpsPrimal', 'EpsDual', 'Rho', 'XSlvRelRes', 'XSlvCGIt',
             'Time'])
    """Named tuple type for recording ADMM iteration statistics"""

    hdrtxt = ['Itn', 'DFid', 'Cnstr', 'r', 's', 'rho']
    """Display column header text"""
    hdrval = {'Itn' : 'Iter', 'DFid' : 'DFid', 'Cnstr' : 'Cnstr',
              'r' : 'PrimalRsdl', 's' : 'DualRsdl', 'rho' : 'Rho'}
    """Dictionary mapping display column headers to IterationStats entries"""



    def __init__(self, A, S, dsz, opt=None, dimN=2, dimK=1):
        """Initialise a ConvCnstrMOD object with problem parameters.

        This class supports an arbitrary number of spatial dimensions,
        dimN, with a default of 2. The input coefficient map array A
        (usually labelled X, but renamed here to avoid confusion with
        the X and Y variables in the ADMM base class) is expected to
        be in standard form as computed by the ConvBPDN class.

        The input signal set S is either dimN dimensional (no
        channels, only one signal), dimN+1 dimensional (either
        multiple channels or multiple signals), or dimN+2 dimensional
        (multiple channels and multiple signals). Parameter dimK, with
        a default value of 1, indicates the number of multiple-signal
        dimensions in S:

        ::

          Default dimK = 1, i.e. assume input S is of form
            S(N0,  N1,   C,   K)  or  S(N0,  N1,   K)
          If dimK = 0 then input S is of form
            S(N0,  N1,   C,   K)  or  S(N0,  N1,   C)

        The internal data layout for S, D (X here), and X (A here) is:
        ::

          dim<0> - dim<Nds-1> : Spatial dimensions, product of N0,N1,... is N
          dim<Nds>            : C number of channels in S and D
          dim<Nds+1>          : K number of signals in S
          dim<Nds+2>          : M number of filters in D

            sptl.      chn  sig  flt
          S(N0,  N1,   C,   K,   1)
          D(N0,  N1,   C,   1,   M)   (X here)
          X(N0,  N1,   1,   K,   M)   (A here)

        The dsz parameter indicates the desired filter supports in the
        output dictionary, since this cannot be inferred from the
        input variables. The format is the same as the dsz parameter
        of :func:`bcrop`.

        Parameters
        ----------
        A : array_like
          Coefficient map array
        S : array_like
          Signal array
        dsz : tuple
          Filter support size(s)
        opt : ccmod.Options object
          Algorithm options
        dimN : int, optional (default 2)
          Number of spatial dimensions
        dimK : int, optional (default 1)
          Number of dimensions for multiple signals in input S
        """

        # Set default options if none specified
        if opt is None:
            opt = ConvCnstrMOD.Options()

        # Infer problem dimensions and set relevant attributes of self
        cri = ConvRepIndexing(S, dsz, dimN, dimK)
        for attr in ['dimN', 'dimC', 'dimK', 'C', 'K', 'M', 'Nv', 'N',
                     'axisN', 'axisC', 'axisK', 'axisM', 'dsz']:
            setattr(self, attr, getattr(cri, attr))

        # Call parent class __init__
        Nx = self.M*self.N
        super(ConvCnstrMOD, self).__init__(Nx, opt)

        # Reshape S to standard layout (A, i.e. X in cbpdn, is assumed
        # to be taken from cbpdn, and therefore already in standard
        # form)
        self.S = S.reshape(cri.shpS)

        # Compute signal S in DFT domain
        self.Sf = sl.rfftn(self.S, None, self.axisN)

        # Create constraint set projection function
        self.Pcn = getPcn(opt['ZeroMean'], dsz, self.Nv, self.dimN)

        # Set rho value (computed from K if not specified)
        self.opt.set_K(self.K)
        self.rho = self.opt['rho']

        # Determine working data type
        self.opt.set_dtype(S.dtype)
        dtype = self.opt['DataType']

        # Initial values for Y
        if self.opt['Y0'] is None:
            self.Y = np.zeros(cri.shpD, dtype)
        else:
            self.Y = self.opt['Y0']
        self.Yprev = self.Y

        # Initial value for U
        if  self.opt['U0'] is None:
            if  self.opt['Y0'] is None:
                self.U = self.U = np.zeros(cri.shpD, dtype)
            else:
                # If Y0 is given, but not U0, then choose the initial
                # U so that the relevant dual optimality criterion
                # (see (3.10) in boyd-2010-distributed) is satisfied.
                self.U = self.Y
        else:
            self.U = self.opt['U0']

        # Create byte aligned arrays for FFT calls
        self.YU = sl.pyfftw_empty_aligned(self.Y.shape, dtype=dtype)
        xfshp = list(self.Y.shape)
        xfshp[dimN-1] = xfshp[dimN-1]//2 + 1
        self.Xf = sl.pyfftw_empty_aligned(xfshp, dtype=sl.complex_dtype(dtype))

        self.runtime += self.timer.elapsed()

        if A is not None:
            self.setcoef(A)



    def setcoef(self, A):
        """Set coefficient array."""

        self.timer.start()
        self.A = A
        self.Af = sl.rfftn(self.A, self.Nv, self.axisN)
        # Compute D^H S
        self.ASf = np.sum(np.conj(self.Af) * self.Sf, self.axisK, keepdims=True)
        self.runtime += self.timer.elapsed()



    def getdict(self):
        """Get final dictionary."""

        return bcrop(self.Y, self.dsz)



    def xstep(self):
        """Minimise Augmented Lagrangian with respect to x."""

        self.cgit = None
        self.xrrs = None
        self.YU[:] = self.Y - self.U
        b = self.ASf + self.rho*sl.rfftn(self.YU, None, self.axisN)
        if self.opt['LinSolve'] == 'SM':
            self.Xf[:] = sl.solvemdbi_ism(self.Af, self.rho, b, self.axisM,
                                self.axisK)
        else:
            self.Xf[:], cgit = sl.solvemdbi_cg(self.Af, self.rho, b,
                                self.axisM, self.axisK,
                                self.opt['CG', 'StopTol'],
                                self.opt['CG', 'MaxIter'], self.Xf)
            self.cgit = cgit

        self.X = sl.irfftn(self.Xf, self.Nv, self.axisN)

        if self.opt['LinSolveCheck']:
            Aop = lambda x: np.sum(self.Af * x, axis=self.axisM, keepdims=True)
            AHop = lambda x: np.sum(np.conj(self.Af) * x, axis=self.axisK,
                                    keepdims=True)
            ax = AHop(Aop(self.Xf)) + self.rho*self.Xf
            self.xrrs = sl.rrs(ax, b)



    def ystep(self):
        """Minimise Augmented Lagrangian with respect to y."""

        self.Y = self.Pcn(self.AX + self.U)



    def obfn_fvarf(self):
        """Variable to be evaluated in computing data fidelity term,
        depending on 'fEvalX' option value.
        """

        return self.Xf if self.opt['fEvalX'] else \
            sl.rfftn(self.Y, None, self.axisN)



    def iteration_stats(self, k, r, s, epri, edua, tk):
        """
        Construct iteration stats record tuple. Data fidelity term is
        :math:`(1/2) \|  \sum_m \mathbf{d}_m * \mathbf{x}_m -
        \mathbf{s} \|_2^2` and measure of constraint violation is
        :math:`\| P(\mathbf{y}) -  \mathbf{y}\|_2`.
        """

        Ef = np.sum(self.Af * self.obfn_fvarf(), axis=self.axisM,
                    keepdims=True) - self.Sf
        dfd = sl.rfl2norm2(Ef, self.S.shape, axis=tuple(range(self.dimN))) / 2.0
        cns = linalg.norm((self.Pcn(self.obfn_gvar()) - self.obfn_gvar()))
        itst = type(self).IterationStats(k, dfd, cns, r, s, epri, edua,
                                         self.rho, self.xrrs, self.cgit, tk)
        return itst




def stdformD(D, C, M, dimN=2):
    """Reshape dictionary array (X here, D in cbpdn module) to internal
    standard form.

    Parameters
    ----------
    D : array_like
      Dictionary array
    C : int
      Size of channel index
    M : int
      Number of filters in dictionary
    dimN : int, optional (default 2)
      Number of problem spatial indices

    Returns
    -------
    Dr : ndarray
      Reshaped dictionary array
    """

    return D.reshape(D.shape[0:dimN] + (C,) + (1,) + (M,))



def getPcn0(zm, dsz, dimN=2, dimC=1):
    """Construct constraint set projection function without support
    projection. The dsz parameter specifies the support sizes of each
    filter using the same format as the dsz argument of :func:`bcrop`.

    Parameters
    ----------
    zm : bool
      Flag indicating whether the projection function should include
      filter mean subtraction
    dsz : tuple
      Filter support size(s)
    dimN : int, optional (default 2)
      Number of problem spatial indices
    dimC : int, optional (default 1)
      Number of problem channel indices

    Returns
    -------
    fn : function
      Constraint set projection function
    """

    if zm:
        return lambda x: normalise(zeromean(bcrop(x, dsz), dsz), dimN+dimC)
    else:
        return lambda x: normalise(bcrop(x, dsz), dimN+dimC)



def getPcn(zm, dsz, Nv, dimN=2, dimC=1):
    """Construct the constraint set projection function utilised by
    ystep. The dsz parameter specifies the support sizes of each
    filter using the same format as the dsz argument of :func:`bcrop`.

    Parameters
    ----------
    zm : bool
      Flag indicating whether the projection function should include
      filter mean subtraction
    dsz : tuple
      Filter support size(s)
    Nv : tuple
      Sizes of problem spatial indices
    dimN : int, optional (default 2)
      Number of problem spatial indices
    dimC : int, optional (default 1)
      Number of problem channel indices

    Returns
    -------
    fn : function
      Constraint set projection function
    """

    if zm:
        return lambda x: normalise(zeromean(zpad(bcrop(x, dsz), Nv), dsz),
                                   dimN+dimC)
    else:
        return lambda x: normalise(zpad(bcrop(x, dsz), Nv), dimN+dimC)



def zeromean(v, dsz):
    """Subtract mean value from each filter in the input array v. The dsz
    parameter specifies the support sizes of each filter using the
    same format as the dsz argument of :func:`bcrop`. Support sizes
    must be taken into account to ensure that the mean values are
    computed over the correct number of samples, ingoring the
    zero-padded region in which the filter is embedded.

    Parameters
    ----------
    v : array_like
      Input dictionary array
    dsz : tuple
      Filter support size(s)

    Returns
    -------
    vz : array_like
      Dictionary array with filter means subtracted
    """

    vz = v.copy()
    if isinstance(dsz[0], tuple):
        # Multi-scale dictionary specification
        dimN = len(dsz[0]) - 1
        axisN = tuple(range(0, dimN))
        m0 = 0  # Initial index of current block of equi-sized filters
        # Iterate over distinct filter sizes
        for mb in range(0, len(dsz)):
            m1 = m0 + dsz[mb][-1]  # End index of current block of filters
            # Construct slice corresponding to cropped part of
            # current block of filters in output array and set from
            # input array
            mbslc = tuple([slice(0, x) for x in dsz[mb][0:-1]]) + \
                    (Ellipsis,) + (slice(m0, m1),)
            vz[mbslc] -= np.mean(v[mbslc], axisN)
            m0 = m1  # Update initial index for start of next block
    else:
        # Single scale dictionary specification
        dimN = len(dsz) - 1
        axisN = tuple(range(0, dimN))
        axnslc = tuple([slice(0, x) for x in dsz[0:dimN]])
        vz[axnslc] -= np.mean(v[axnslc], axisN)

    return vz



def normalise(v, dimN=2):
    """Normalise vectors, corresponding to slices along specified number
    of initial spatial dimensions of an array, to have unit
    :math:`\ell^2` norm. The remaining axes enumerate the distinct
    vectors to be normalised.

    Parameters
    ----------
    v : array_like
      Array with components to be normalised
    dimN : int, optional (default 2)
      Number of initial dimensions over which norm should be computed

    Returns
    -------
    vp : array_like
      Normalised array
    """

    axisN = tuple(range(0,dimN))
    vn = np.sqrt(np.sum(v**2, axisN, keepdims=True))
    vn[vn == 0] = 1.0
    return v / vn



def zpad(v, Nv):
    """Zero-pad initial axes of array to specified size. Padding is
    applied to the right, top, etc. of the array indices.

    Parameters
    ----------
    v : array_like
      Array to be padded
    Nv : tuple
      Sizes to which each of initial indices should be padded

    Returns
    -------
    vp : array_like
      Padded array
    """

    vp = np.zeros(Nv + v.shape[len(Nv):])
    axnslc = tuple([slice(0, x) for x in v.shape])
    vp[axnslc] = v
    return vp



def bcrop(v, dsz):
    """Crop specified number of initial spatial dimensions of dictionary
    array to specified size. Parameter dsz must have one of the two
    forms (assuming two spatial dimensions):

    ::

      (flt_rows, filt_cols, num_filts)

    or

    ::

      (
       (flt_rows1, filt_cols1, num_filts1),
       (flt_rows2, filt_cols2, num_filts2),
       ...
      )

    The total number of dictionary filters, is either num_filts in the
    first form, or the sum of all num_filts in the second form. If the
    filters are not two-dimensional, then the dimensions above vary
    accordingly, i.e., there may be fewer or more filter spatial
    dimensions than flt_rows, filt_cols, e.g.

    ::

      (flt_rows, num_filts)

    for one-dimensional signals, or

    ::

      (flt_rows, filt_cols, filt_planes, num_filts)

    for three-dimensional signals.

    Parameters
    ----------
    v : array_like
      Dictionary array to be cropped
    dsz : tuple
      Filter support size(s)

    Returns
    -------
    vc : array_like
      Cropped dictionary array

    """

    if isinstance(dsz[0], tuple):
        # Multi-scale dictionary specification
        dimN = len(dsz[0]) - 1
        maxsz = np.amax(np.array(dsz)[:, 0:dimN], axis=0)  # Max. support size
        vc = np.zeros(tuple(maxsz) + v.shape[dimN:])  # Init. cropped array
        m0 = 0  # Initial index of current block of equi-sized filters
        # Iterate over distinct filter sizes
        for mb in range(0, len(dsz)):
            m1 = m0 + dsz[mb][-1]  # End index of current block of filters
            # Construct slice corresponding to cropped part of
            # current block of filters in output array and set from
            # input array
            mbslc = tuple([slice(0, x) for x in dsz[mb][0:-1]]) + \
                    (Ellipsis,) + (slice(m0, m1),)
            vc[mbslc] = v[mbslc]
            m0 = m1  # Update initial index for start of next block
        return vc
    else:
        # Single scale dictionary specification
        dimN = len(dsz) - 1
        axnslc = tuple([slice(0, x) for x in dsz[0:dimN]])
        return v[axnslc]
