# -*- coding: utf-8 -*-
# Copyright (C) 2017-2025 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Parallel consensus convolutional dictionary learning"""

from __future__ import print_function
from builtins import range

import platform
if platform.system() == 'Windows' or platform.system() == 'Darwin':
    raise RuntimeError('Module %s is not supported under Windows or '
                       'MacOS' % __name__)
import collections
import multiprocessing as mp
import numpy as np

import sporco.linalg as sl
import sporco.prox as sp
import sporco.fft
# Required due to pyFFTW bug #135 - see "Notes" section of SPORCO docs
sporco.fft.pyfftw_threads = 1
from sporco.dictlrn import cbpdndl
from sporco.dictlrn import cbpdndlmd
from sporco.fft import fftn_func, ifftn_func, fl2norm2_func
from sporco.array import transpose_ntpl_list
import sporco.cnvrep as cr
from sporco.util import u
from sporco import common



__all__ = ['ConvBPDNDictLearn_Consensus',
           'ConvBPDNMaskDcplDictLearn_Consensus']


# Initialise global variables required by multiprocessing mechanism
mp_cri = None    # A cnvrep.CSC_ConvRepIndexing object describing problem
                 # dimensions
mp_lmbda = None  # Regularisation parameter lambda
mp_dprox = None  # Projection operator of the dictionary update
mp_xrho = None   # Penalty parameter of the X (cbpdn) step
mp_drho = None   # Penalty parameter of the D (ccmod) step
mp_xrlx = None   # Relaxation parameter of the X (cbpdn) step
mp_drlx = None   # Relaxation parameter of the D (ccmod) step
mp_Sf = None     # Training data array in DFT domain
mp_Df = None     # Dictionary variable (in DFT domain) used by X step
mp_Zf = None     # Coefficient map variable (in DFT domain) used by D step
mp_DSf = None    # D^T S in DFT domain
mp_ZSf = None    # Z^T S in DFT domain
mp_Z_X = None    # Primary variable of X update
mp_Z_Y = None    # Auxiliary variable of X update
mp_Z_U = None    # Lagrange multiplier of X update
mp_D_X = None    # Primary variable of D update
mp_D_Y = None    # Auxiliary variable of D update
mp_D_U = None    # Lagrange multiplier of D update

# Global references to appropriate functions from sporco.fft
real_dtype = None
fftn = None
ifftn = None
fl2norm2 = None



def mpraw_as_np(shape, dtype):
    """Construct a numpy array of the specified shape and dtype for which the
    underlying storage is a multiprocessing RawArray in shared memory.

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



def swap_axis_to_0(x, axis):
    """Insert a new singleton axis at position 0 and swap it with the
    specified axis. The resulting array has an additional dimension,
    with ``axis`` + 1 (which was ``axis`` before the insertion of the
    new axis) of ``x`` at position 0, and a singleton axis at position
    ``axis`` + 1.

    Parameters
    ----------
    x : ndarray
      Input array
    axis : int
      Index of axis in ``x`` to swap to axis index 0.

    Returns
    -------
    arr : ndarray
      Output array
    """

    return np.ascontiguousarray(np.swapaxes(x[np.newaxis, ...], 0, axis+1))



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



def cbpdn_setdict():
    """Set the dictionary for the cbpdn stage. There are no parameters
    or return values because all inputs and outputs are from and to
    global variables.
    """

    global mp_DSf
    # Set working dictionary for cbpdn step and compute DFT of dictionary
    # D and of D^T S
    mp_Df[:] = fftn(mp_D_Y, mp_cri.Nv, mp_cri.axisN)
    if mp_cri.Cd == 1:
        mp_DSf[:] = np.conj(mp_Df) * mp_Sf
    else:
        mp_DSf[:] = sl.inner(np.conj(mp_Df[np.newaxis, ...]), mp_Sf,
                             axis=mp_cri.axisC+1)



def cbpdn_xstep(k):
    """Do the X step of the cbpdn stage. The only parameter is the slice
    index `k` and there are no return values; all inputs and outputs are
    from and to global variables.
    """

    YU = mp_Z_Y[k] - mp_Z_U[k]
    b = mp_DSf[k] + mp_xrho * fftn(YU, None, mp_cri.axisN)
    if mp_cri.Cd == 1:
        Xf = sl.solvedbi_sm(mp_Df, mp_xrho, b, axis=mp_cri.axisM)
    else:
        Xf = sl.solvemdbi_ism(mp_Df, mp_xrho, b, mp_cri.axisM, mp_cri.axisC)
    mp_Z_X[k] = ifftn(Xf, mp_cri.Nv, mp_cri.axisN)



def cbpdn_relax(k):
    """Do relaxation for the cbpdn stage. The only parameter is the slice
    index `k` and there are no return values; all inputs and outputs are
    from and to global variables.
    """

    mp_Z_X[k] = mp_xrlx * mp_Z_X[k] + (1 - mp_xrlx) * mp_Z_Y[k]



def cbpdn_ystep(k):
    """Do the Y step of the cbpdn stage. The only parameter is the slice
    index `k` and there are no return values; all inputs and outputs are
    from and to global variables.
    """

    AXU = mp_Z_X[k] + mp_Z_U[k]
    mp_Z_Y[k] = sp.prox_l1(AXU, (mp_lmbda/mp_xrho))



def cbpdn_ustep(k):
    """Do the U step of the cbpdn stage. The only parameter is the slice
    index `k` and there are no return values; all inputs and outputs are
    from and to global variables.
    """

    mp_Z_U[k] += mp_Z_X[k] - mp_Z_Y[k]



def ccmod_setcoef(k):
    """Set the coefficient maps for the ccmod stage. The only parameter
    is the slice index `k` and there are no return values; all inputs and
    outputs are from and to global variables.
    """

    # Set working coefficient maps for ccmod step and compute DFT of
    # coefficient maps Z and Z^T S
    mp_Zf[k] = fftn(mp_Z_Y[k], mp_cri.Nv, mp_cri.axisN)
    mp_ZSf[k] = np.conj(mp_Zf[k]) * mp_Sf[k]



def ccmod_xstep(k):
    """Do the X step of the ccmod stage. The only parameter is the slice
    index `k` and there are no return values; all inputs and outputs are
    from and to global variables.
    """

    YU = mp_D_Y - mp_D_U[k]
    b = mp_ZSf[k] + mp_drho * fftn(YU, None, mp_cri.axisN)
    Xf = sl.solvedbi_sm(mp_Zf[k], mp_drho, b, axis=mp_cri.axisM)
    mp_D_X[k] = ifftn(Xf, mp_cri.Nv, mp_cri.axisN)



def ccmod_relax(k):
    """Do relaxation for the ccmod stage. The only parameter is the slice
    index `k` and there are no return values; all inputs and outputs are
    from and to global variables.
    """

    mp_D_X[k] = mp_drlx * mp_D_X[k] + (1 - mp_drlx) * mp_D_Y



def ccmod_ystep():
    """Do the Y step of the ccmod stage. There are no parameters
    or return values because all inputs and outputs are from and to
    global variables.
    """

    mAXU = np.mean(mp_D_X + mp_D_U, axis=0)
    mp_D_Y[:] = mp_dprox(mAXU)



def ccmod_ustep():
    """Do the U step of the ccmod stage. There are no parameters
    or return values because all inputs and outputs are from and to
    global variables.
    """

    mp_D_U[:] += mp_D_X[:] - mp_D_Y



def step_group(k):
    """Do a single iteration over cbpdn and ccmod steps that can be
    performed independently for each slice `k` of the input data set.
    """

    cbpdn_xstep(k)
    if mp_xrlx != 1.0:
        cbpdn_relax(k)
    cbpdn_ystep(k)
    cbpdn_ustep(k)
    ccmod_setcoef(k)
    ccmod_xstep(k)
    if mp_drlx != 1.0:
        ccmod_relax(k)





class ConvBPDNDictLearn_Consensus(cbpdndl.ConvBPDNDictLearn):
    r"""
    Dictionary learning based on Convolutional BPDN
    :cite:`wohlberg-2014-efficient` and an ADMM Consensus solution of
    the constrained dictionary update problem :cite:`sorel-2016-fast`.

    |

    .. inheritance-diagram:: ConvBPDNDictLearn_Consensus
       :parts: 2

    |

    The dictionary learning algorithm itself is as described in
    :cite:`garcia-2018-convolutional1`. The sparse coding of each training
    image and the individual consensus problem components are computed in
    parallel, giving a substantial computational advantage, on a multi-core
    host, over :class:`.dictlrn.cbpdndl.ConvBPDNDictLearn` with the consensus
    solver (``dmethod`` = ``'cns'``) for the constrained dictionary update
    problem.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
       (1/2) \sum_k \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \right \|_2^2 + \lambda \sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1 \quad \text{such that}
       \quad \mathbf{d}_m \in C \;\; \forall m \;,

    where :math:`C` is the feasible set consisting of filters with
    unit norm and constrained support, via interleaved alternation
    between the ADMM steps of the sparse coding and dictionary update
    algorithms. Multi-channel signals with multi-channel dictionaries are
    supported. Unlike :class:`.dictlrn.cbpdndl.ConvBPDNDictLearn`,
    multi-channel signals with single-channel dictionaries are *not*
    supported; it is the responsibility of the user to convert the
    multi-channel input signal to a single-channel input with channels
    converted into additional single-channel signal instances.

    This class is derived from :class:`.dictlrn.cbpdndl.ConvBPDNDictLearn`
    so that the variable initialisation of its parent can be re-used. The
    entire :meth:`.solve` infrastructure is overidden in this class,
    without any use of inherited functionality. Variables initialised by
    the parent class that are non-singleton on axis ``axisK`` have this
    axis swapped with axis 0 for simpler and more computationally
    efficient indexing. Note that automatic penalty parameter selection
    (see option ``AutoRho`` in :class:`.admm.ADMM.Options`) is not
    supported, the option settings being silently ignored.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \sum_k \|
       \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} - \mathbf{s}_k \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1`

       ``Time`` : Cumulative run time
    """


    class Options(cbpdndl.ConvBPDNDictLearn.Options):
        """ConvBPDNDictLearn_Consensus algorithm options

        Options are the same as defined in
        :class:`cbpdndl.ConvBPDNDictLearn.Options`.
        """

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDNDictLearn_Consensus algorithm options
            """

            if opt is None:
                opt = {}
            cbpdndl.ConvBPDNDictLearn.Options.__init__(
                self, opt, xmethod='admm', dmethod='cns')




    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""


    def __init__(self, D0, S, lmbda=None, opt=None, nproc=None, dimK=1,
                 dimN=2):
        """
        Parameters
        ----------
        D0 : array_like
          Initial dictionary array
        S : array_like
          Signal array
        lmbda : float
          Regularisation parameter
        opt : :class:`.dictlrn.cbpdndl.ConvBPDNDictLearn.Options` object
          Algorithm options
        nproc : int
          Number of parallel processes to use
        dimK : int, optional (default 1)
          Number of signal dimensions. If there is only a single input
          signal (e.g. if `S` is a 2D array representing a single image)
          `dimK` must be set to 0.
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        if nproc is None:
            # Number of processes to run is the smaller of the number of
            # CPUs and K, the number of training signals
            self.nproc = min(mp.cpu_count(), S.shape[-1])
        else:
            self.nproc = nproc

        # Set flag indicating whether problem involves real or complex
        # values, and get appropriate versions of functions from fft
        # module
        global real_dtype, fftn, ifftn, fl2norm2
        real_dtype = np.isrealobj(D0) and np.isrealobj(S)
        fftn = fftn_func(real_dtype)
        ifftn = ifftn_func(real_dtype)
        fl2norm2 = fl2norm2_func(real_dtype)

        # Call parent constructor
        super(ConvBPDNDictLearn_Consensus, self).__init__(
            D0, S, lmbda, opt=opt, xmethod='admm', dmethod='cns',
            dimK=dimK, dimN=dimN)

        # Check for unsupported case of multi-channel signal with
        # single-channel dictionary
        if self.xstep.cri.Cd == 1 and self.xstep.cri.C > 1:
            raise ValueError("Learning of a single-channel dictionary from"
                             " multi-channel training data is not supported")

        # Set up iterations statistics
        itstat_fields = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Time']
        self.IterationStats = collections.namedtuple('IterationStats',
                                                     itstat_fields)
        self.itstat = []

        # Initialise iteration counter
        self.j = 0



        def init_mpraw_swap(mpv, npv):
            """Set a global variable as a multiprocessing RawArray in shared
            memory with a numpy array wrapper and initialise its value
            to the specified array after swapping axisK of that array
            to axis index 0.

            Parameters
            ----------
            mpv : string
              Name of global variable to set
            npv : ndarray
              Numpy array to use as initialiser for global variable value
            """

            v = swap_axis_to_0(npv, self.xstep.cri.axisK)
            init_mpraw(mpv, v)



        # Initialise global variables
        global mp_cri
        mp_cri = self.xstep.cri
        global mp_lmbda
        mp_lmbda = self.xstep.lmbda
        global mp_xrho
        mp_xrho = self.xstep.rho
        global mp_drho
        mp_drho = self.dstep.rho
        global mp_xrlx
        mp_xrlx = self.xstep.rlx
        global mp_drlx
        mp_drlx = self.dstep.rlx
        global mp_dprox
        mp_dprox = self.dstep.Pcn
        global mp_Sf
        init_mpraw_swap('mp_Sf', self.xstep.Sf)
        global mp_Df
        init_mpraw('mp_Df', self.xstep.Df)
        global mp_Zf
        shp = list(mp_Sf.shape)
        shp[-1] = self.xstep.cri.M
        mp_Zf = mpraw_as_np(shp, mp_Sf.dtype)
        global mp_DSf
        init_mpraw_swap('mp_DSf', self.xstep.DSf)
        global mp_ZSf
        mp_ZSf = mpraw_as_np(shp, mp_Sf.dtype)
        global mp_Z_Y
        init_mpraw_swap('mp_Z_Y', self.xstep.Y)
        global mp_Z_X
        mp_Z_X = mpraw_as_np(mp_Z_Y.shape, mp_Z_Y.dtype)
        global mp_Z_U
        init_mpraw_swap('mp_Z_U', self.xstep.U)
        global mp_D_X
        dxshp = list((self.dstep.cri.K,) + self.dstep.cri.shpD)
        mp_D_X = mpraw_as_np(dxshp, self.dstep.Y.dtype)
        global mp_D_Y
        init_mpraw('mp_D_Y', self.dstep.Y)
        global mp_D_U
        init_mpraw('mp_D_U', np.moveaxis(self.dstep.U, -1, 0))




    def step(self):
        """Do a single iteration over all cbpdn and ccmod steps. Those that
        are not coupled on the K axis are performed in parallel."""

        # If the nproc parameter of __init__ is zero, just iterate
        # over the K consensus instances instead of using
        # multiprocessing to do the computations in parallel. This is
        # useful for debugging and timing comparisons.
        if self.nproc == 0:
            for k in range(self.xstep.cri.K):
                step_group(k)
        else:
            self.pool.map(step_group, range(self.xstep.cri.K))

        ccmod_ystep()
        ccmod_ustep()
        cbpdn_setdict()



    def solve(self):
        """Start (or re-start) optimisation. This method implements the
        framework for the alternation between `X` and `D` updates in a
        dictionary learning algorithm.

        If option ``Verbose`` is ``True``, the progress of the
        optimisation is displayed at every iteration. At termination
        of this method, attribute :attr:`itstat` is a list of tuples
        representing statistics of each iteration.

        Attribute :attr:`timer` is an instance of :class:`.util.Timer`
        that provides the following labelled timers:

          ``init``: Time taken for object initialisation by
          :meth:`__init__`

          ``solve``: Total time taken by call(s) to :meth:`solve`

          ``solve_wo_func``: Total time taken by call(s) to
          :meth:`solve`, excluding time taken to compute functional
          value and related iteration statistics
        """

        # Construct tuple of status display column titles and set status
        # display strings
        hdrtxt = ['Itn', 'Fnc', 'DFid', u('Regℓ1')]
        hdrstr, fmtstr, nsep = common.solve_status_str(
            hdrtxt, fwdth0=type(self).fwiter, fprec=type(self).fpothr)

        # Print header and separator strings
        if self.opt['Verbose']:
            if self.opt['StatusHeader']:
                print(hdrstr)
                print("-" * nsep)

        # Reset timer
        self.timer.start(['solve', 'solve_wo_eval'])

        # Create process pool
        if self.nproc > 0:
            self.pool = mp.Pool(processes=self.nproc)

        for self.j in range(self.j, self.j + self.opt['MaxMainIter']):

            # Perform a set of update steps
            self.step()

            # Evaluate functional
            self.timer.stop('solve_wo_eval')
            fnev = self.evaluate()
            self.timer.start('solve_wo_eval')

            # Record iteration stats
            tk = self.timer.elapsed('solve')
            itst = self.IterationStats(*((self.j,) + fnev + (tk,)))
            self.itstat.append(itst)

            # Display iteration stats if Verbose option enabled
            if self.opt['Verbose']:
                print(fmtstr % itst[:-1])

            # Call callback function if defined
            if self.opt['Callback'] is not None:
                if self.opt['Callback'](self):
                    break

        # Clean up process pool
        if self.nproc > 0:
            self.pool.close()
            self.pool.join()

        # Increment iteration count
        self.j += 1

        # Record solve time
        self.timer.stop(['solve', 'solve_wo_eval'])

        # Print final separator string if Verbose option enabled
        if self.opt['Verbose'] and self.opt['StatusHeader']:
            print("-" * nsep)

        # Return final dictionary
        return self.getdict()



    def getdict(self, crop=True):
        """Get final dictionary. If ``crop`` is ``True``, apply
        :func:`.cnvrep.bcrop` to returned array.
        """

        global mp_D_Y
        D = mp_D_Y
        if crop:
            D = cr.bcrop(D, self.dstep.cri.dsz, self.dstep.cri.dimN)
        return D



    def getcoef(self):
        """Get final coefficient map array."""

        global mp_Z_Y
        return np.swapaxes(mp_Z_Y, 0, self.xstep.cri.axisK+1)[0]



    def evaluate(self):
        """Evaluate functional value of previous iteration."""

        X = mp_Z_Y
        Xf = mp_Zf
        Df = mp_Df
        Sf = mp_Sf
        Ef = sl.inner(Df[np.newaxis, ...], Xf,
                      axis=self.xstep.cri.axisM+1) - Sf
        Ef = np.swapaxes(Ef, 0, self.xstep.cri.axisK+1)[0]
        dfd = fl2norm2(Ef, self.xstep.S.shape, axis=self.xstep.cri.axisN)/2.0
        rl1 = np.sum(np.abs(X))
        obj = dfd + self.xstep.lmbda*rl1
        return (obj, dfd, rl1)



    def getitstat(self):
        """Get iteration stats as named tuple of arrays instead of array
        of named tuples.
        """

        return transpose_ntpl_list(self.itstat)







# Initialise global variables required by multiprocessing mechanism
mp_S = None       # Training data array
mp_W = None       # Mask array
mp_DX = None      # Product of D and X
mp_Z_Y0 = None    # Auxiliary variables of X update
mp_Z_Y1 = None
mp_Z_U0 = None    # Lagrange multipliers of X update
mp_Z_U1 = None
mp_D_Y0 = None    # Auxiliary variables of D update
mp_D_Y1 = None
mp_D_U0 = None    # Lagrange multipliers of D update
mp_D_U1 = None



def cbpdnmd_setdict():
    """Set the dictionary for the cbpdn stage. There are no parameters
    or return values because all inputs and outputs are from and to
    global variables.
    """

    # Set working dictionary for cbpdn step and compute DFT of dictionary D
    mp_Df[:] = fftn(mp_D_Y0, mp_cri.Nv, mp_cri.axisN)



def cbpdnmd_xstep(k):
    """Do the X step of the cbpdn stage. The only parameter is the slice
    index `k` and there are no return values; all inputs and outputs are
    from and to global variables.
    """

    YU0 = mp_Z_Y0[k] + mp_S[k] - mp_Z_U0[k]
    YU1 = mp_Z_Y1[k] - mp_Z_U1[k]
    if mp_cri.Cd == 1:
        b = np.conj(mp_Df) * fftn(YU0, None, mp_cri.axisN) + \
            fftn(YU1, None, mp_cri.axisN)
        Xf = sl.solvedbi_sm(mp_Df, 1.0, b, axis=mp_cri.axisM)
    else:
        b = sl.inner(np.conj(mp_Df), fftn(YU0, None, mp_cri.axisN),
                     axis=mp_cri.axisC) + fftn(YU1, None, mp_cri.axisN)
        Xf = sl.solvemdbi_ism(mp_Df, 1.0, b, mp_cri.axisM, mp_cri.axisC)
    mp_Z_X[k] = ifftn(Xf, mp_cri.Nv, mp_cri.axisN)
    mp_DX[k] = ifftn(sl.inner(mp_Df, Xf), mp_cri.Nv, mp_cri.axisN)



def cbpdnmd_relax(k):
    """Do relaxation for the cbpdn stage. The only parameter is the slice
    index `k` and there are no return values; all inputs and outputs are
    from and to global variables.
    """

    mp_Z_X[k] = mp_xrlx * mp_Z_X[k] + (1 - mp_xrlx) * mp_Z_Y1[k]
    mp_DX[k] = mp_xrlx * mp_DX[k] + (1 - mp_xrlx) * (mp_Z_Y0[k] + mp_S[k])



def cbpdnmd_ystep(k):
    """Do the Y step of the cbpdn stage. The only parameter is the slice
    index `k` and there are no return values; all inputs and outputs are
    from and to global variables.
    """

    if mp_W.shape[0] > 1:
        W = mp_W[k]
    else:
        W = mp_W
    AXU0 = mp_DX[k] - mp_S[k] + mp_Z_U0[k]
    AXU1 = mp_Z_X[k] + mp_Z_U1[k]
    mp_Z_Y0[k] = mp_xrho*AXU0 / (W**2 + mp_xrho)
    mp_Z_Y1[k] = sp.prox_l1(AXU1, (mp_lmbda/mp_xrho))



def cbpdnmd_ustep(k):
    """Do the U step of the cbpdn stage. The only parameter is the slice
    index `k` and there are no return values; all inputs and outputs are
    from and to global variables.
    """

    mp_Z_U0[k] += mp_DX[k] - mp_Z_Y0[k] - mp_S[k]
    mp_Z_U1[k] += mp_Z_X[k] - mp_Z_Y1[k]



def ccmodmd_setcoef(k):
    """Set the coefficient maps for the ccmod stage. The only parameter
    is the slice index `k` and there are no return values; all inputs and
    outputs are from and to global variables.
    """

    # Set working coefficient maps for ccmod step and compute DFT of
    # coefficient maps Z
    mp_Zf[k] = fftn(mp_Z_Y1[k], mp_cri.Nv, mp_cri.axisN)



def ccmodmd_xstep(k):
    """Do the X step of the ccmod stage. The only parameter is the slice
    index `k` and there are no return values; all inputs and outputs are
    from and to global variables.
    """

    YU0 = mp_D_Y0 - mp_D_U0[k]
    YU1 = mp_D_Y1[k] + mp_S[k] - mp_D_U1[k]
    b = fftn(YU0, None, mp_cri.axisN) + \
        np.conj(mp_Zf[k]) * fftn(YU1, None, mp_cri.axisN)
    Xf = sl.solvedbi_sm(mp_Zf[k], 1.0, b, axis=mp_cri.axisM)
    mp_D_X[k] = ifftn(Xf, mp_cri.Nv, mp_cri.axisN)
    mp_DX[k] = ifftn(sl.inner(Xf, mp_Zf[k]), mp_cri.Nv, mp_cri.axisN)



def ccmodmd_relax(k):
    """Do relaxation for the ccmod stage. The only parameter is the slice
    index `k` and there are no return values; all inputs and outputs are
    from and to global variables.
    """

    mp_D_X[k] = mp_drlx * mp_D_X[k] + (1 - mp_drlx) * mp_D_Y0
    mp_DX[k] = mp_drlx * mp_DX[k] + (1 - mp_drlx) * (mp_D_Y1[k] + mp_S[k])



def ccmodmd_ystep():
    """Do the Y step of the ccmod stage. There are no parameters
    or return values because all inputs and outputs are from and to
    global variables.
    """

    mAXU = np.mean(mp_D_X + mp_D_U0, axis=0)
    mp_D_Y0[:] = mp_dprox(mAXU)
    AXU1 = mp_DX - mp_S + mp_D_U1
    mp_D_Y1[:] = mp_drho*AXU1 / (mp_W**2 + mp_drho)



def ccmodmd_ustep():
    """Do the U step of the ccmod stage. There are no parameters
    or return values because all inputs and outputs are from and to
    global variables.
    """

    mp_D_U0[:] += mp_D_X - mp_D_Y0
    mp_D_U1[:] += mp_DX - mp_D_Y1 - mp_S



def md_step_group(k):
    """Do a single iteration over cbpdn and ccmod steps that can be
    performed independently for each slice `k` of the input data set.
    """

    cbpdnmd_xstep(k)
    if mp_xrlx != 1.0:
        cbpdnmd_relax(k)
    cbpdnmd_ystep(k)
    cbpdnmd_ustep(k)
    ccmodmd_setcoef(k)
    ccmodmd_xstep(k)
    if mp_drlx != 1.0:
        ccmodmd_relax(k)





class ConvBPDNMaskDcplDictLearn_Consensus(cbpdndlmd.ConvBPDNMaskDictLearn):
    r"""
    Dictionary learning based on Convolutional BPDN with Mask Decoupling
    :cite:`heide-2015-fast` and the hybrid Mask Decoupling/Consensus
    solution of the constrained dictionary update problem proposed in
    :cite:`garcia-2018-convolutional1`.

    |

    .. inheritance-diagram:: ConvBPDNMaskDcplDictLearn_Consensus
       :parts: 2

    |

    The dictionary learning algorithm itself is as described in
    :cite:`garcia-2018-convolutional1`. The sparse
    coding of each training image and the individual consensus problem
    components are computed in parallel, giving a substantial computational
    advantage, on a multi-core host, over
    :class:`.cbpdndlmd.ConvBPDNMaskDictLearn` with the consensus solver
    (``method`` = ``'cns'``) for the constrained dictionary update problem.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
       (1/2) \sum_k \left \| W ( \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k ) \right \|_2^2 + \lambda \sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1 \quad \text{such that}
       \quad \mathbf{d}_m \in C \;\; \forall m \;,

    where :math:`W` is a mask array and :math:`C` is the feasible set
    consisting of filters with unit norm and constrained support, via
    interleaved alternation between the ADMM steps of the sparse coding
    and dictionary update algorithms. Multi-channel signals with
    multi-channel dictionaries are supported. Unlike
    :class:`.dictlrn.cbpdndl.ConvBPDNDictLearn`, multi-channel signals
    with single-channel dictionaries are *not* supported; it is the
    responsibility of the user to convert the multi-channel input signal
    to a single-channel input with channels converted into additional
    single-channel signal instances.

    This class is derived from :class:`.cbpdndlmd.ConvBPDNMaskDictLearn`
    so that the variable initialisation of its parent can be re-used. The
    entire :meth:`.solve` infrastructure is overidden in this class, without
    any use of inherited functionality. Variables initialised by the parent
    class that are non-singleton on axis ``axisK`` have this axis swapped
    with axis 0 for simpler and more computationally efficient indexing.
    Note that automatic penalty parameter selection (see option ``AutoRho``
    in :class:`.admm.ADMM.Options`) is not supported, the option settings
    being silently ignored.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \sum_k \|
       W ( \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} - \mathbf{s}_k ) \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1`

       ``Time`` : Cumulative run time
    """


    class Options(cbpdndlmd.ConvBPDNMaskDictLearn.Options):
        """ConvBPDNMaskDcplDictLearn_Consensus algorithm options

        Options are the same as defined in
        :class:`.cbpdndlmd.ConvBPDNMaskDictLearn.Options`.
        """

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              ConvBPDNMaskDcplDictLearn_Consensus algorithm options
            """

            if opt is None:
                opt = {}
            cbpdndlmd.ConvBPDNMaskDictLearn.Options.__init__(
                self, opt, xmethod='admm', dmethod='cns')




    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""


    def __init__(self, D0, S, lmbda=None, W=None, opt=None, nproc=None,
                 dimK=1, dimN=2):
        """
        Parameters
        ----------
        D0 : array_like
          Initial dictionary array
        S : array_like
          Signal array
        lmbda : float
          Regularisation parameter
        W : array_like
          Mask array. The array shape must be such that the array is
          compatible for multiplication with the *internal* shape of
          input array S (see :class:`.cnvrep.CDU_ConvRepIndexing` for a
          discussion of the distinction between *external* and *internal*
          data layouts) after reshaping to the shape determined by
          :func:`.cnvrep.mskWshape`.
        opt : :class:`.cbpdndlmd.ConvBPDNMaskDictLearn.Options` object
          Algorithm options
        nproc : int
          Number of parallel processes to use
        dimK : int, optional (default 1)
          Number of signal dimensions. If there is only a single input
          signal (e.g. if `S` is a 2D array representing a single image)
          `dimK` must be set to 0.
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        if nproc is None:
            # Number of processes to run is the smaller of the number of
            # CPUs and K, the number of training signals
            self.nproc = min(mp.cpu_count(), S.shape[-1])
        else:
            self.nproc = nproc

        # Set flag indicating whether problem involves real or complex
        # values, and get appropriate versions of functions from fft
        # module
        global real_dtype, fftn, ifftn, fl2norm2
        real_dtype = np.isrealobj(D0) and np.isrealobj(S)
        fftn = fftn_func(real_dtype)
        ifftn = ifftn_func(real_dtype)
        fl2norm2 = fl2norm2_func(real_dtype)

        # Call parent constructor
        super(ConvBPDNMaskDcplDictLearn_Consensus, self).__init__(
            D0, S, lmbda, W, opt=opt, xmethod='admm', dmethod='cns',
            dimK=dimK, dimN=dimN)

        # Check for unsupported case of multi-channel signal with
        # single-channel dictionary
        if self.xstep.cri.Cd == 1 and self.xstep.cri.C > 1:
            raise ValueError("Learning of a single-channel dictionary from"
                             " multi-channel training data is not supported")

        # Set up iterations statistics
        itstat_fields = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Time']
        self.IterationStats = collections.namedtuple('IterationStats',
                                                     itstat_fields)
        self.itstat = []

        # Initialise iteration counter
        self.j = 0



        def init_mpraw_swap(mpv, npv):
            """Set a global variable as a multiprocessing RawArray in
            shared memory with a numpy array wrapper and initialise its
            value to the specified array after swapping axisK of that
            array to axis index 0.

            Parameters
            ----------
            mpv : string
              Name of global variable to set
            npv : ndarray
              Numpy array to use as initialiser for global variable value
            """

            v = swap_axis_to_0(npv, self.xstep.cri.axisK)
            init_mpraw(mpv, v)



        # Initialise global variables
        global mp_cri
        mp_cri = self.xstep.cri
        global mp_lmbda
        mp_lmbda = self.xstep.lmbda
        global mp_xrho
        mp_xrho = self.xstep.rho
        global mp_drho
        mp_drho = self.dstep.rho
        global mp_xrlx
        mp_xrlx = self.xstep.rlx
        global mp_drlx
        mp_drlx = self.dstep.rlx
        global mp_dprox
        mp_dprox = self.dstep.Pcn
        global mp_S
        init_mpraw_swap('mp_S', self.xstep.S)
        global mp_Df
        init_mpraw('mp_Df', self.xstep.Df)
        global mp_W
        init_mpraw_swap('mp_W', self.dstep.W)
        global mp_Zf
        shp = np.insert(np.roll(self.xstep.Xf.shape, 1), -1, 1)
        shp[[0, -1]] = shp[[-1, 0]]
        mp_Zf = mpraw_as_np(shp, self.xstep.Xf.dtype)
        global mp_Z_Y0
        init_mpraw_swap('mp_Z_Y0', self.xstep.block_sep0(self.xstep.Y))
        global mp_Z_Y1
        init_mpraw_swap('mp_Z_Y1', self.xstep.block_sep1(self.xstep.Y))
        global mp_Z_X
        mp_Z_X = mpraw_as_np(mp_Z_Y1.shape, self.xstep.Y.dtype)
        global mp_DX
        mp_DX = mpraw_as_np(mp_Z_Y0.shape, self.xstep.Y.dtype)
        mp_DX[:] = 0
        global mp_Z_U0
        init_mpraw_swap('mp_Z_U0', self.xstep.block_sep0(self.xstep.U))
        global mp_Z_U1
        init_mpraw_swap('mp_Z_U1', self.xstep.block_sep1(self.xstep.U))
        global mp_D_X
        dxshp = list((self.dstep.cri.K,) + self.dstep.cri.shpD)
        mp_D_X = mpraw_as_np(dxshp, self.dstep.Y.dtype)
        global mp_D_Y0
        init_mpraw('mp_D_Y0', self.dstep.Y)
        global mp_D_Y1
        init_mpraw('mp_D_Y1',
                   np.moveaxis(self.dstep.Y1, -2, 0)[..., np.newaxis])
        global mp_D_U0
        init_mpraw('mp_D_U0', np.moveaxis(self.dstep.U, -1, 0))
        global mp_D_U1
        init_mpraw('mp_D_U1',
                   np.moveaxis(self.dstep.U1, -2, 0)[..., np.newaxis])




    def step(self):
        """Do a single iteration over all cbpdn and ccmod steps. Those
        that are not coupled on the K axis are performed in parallel."""

        # If the nproc parameter of __init__ is zero, just iterate
        # over the K consensus instances instead of using
        # multiprocessing to do the computations in parallel. This is
        # useful for debugging and timing comparisons.
        if self.nproc == 0:
            for k in range(self.xstep.cri.K):
                md_step_group(k)
        else:
            self.pool.map(md_step_group, range(self.xstep.cri.K))

        ccmodmd_ystep()
        ccmodmd_ustep()
        cbpdnmd_setdict()



    def solve(self):
        """Start (or re-start) optimisation. This method implements the
        framework for the alternation between `X` and `D` updates in a
        dictionary learning algorithm.

        If option ``Verbose`` is ``True``, the progress of the
        optimisation is displayed at every iteration. At termination
        of this method, attribute :attr:`itstat` is a list of tuples
        representing statistics of each iteration.

        Attribute :attr:`timer` is an instance of :class:`.util.Timer`
        that provides the following labelled timers:

          ``init``: Time taken for object initialisation by
          :meth:`__init__`

          ``solve``: Total time taken by call(s) to :meth:`solve`

          ``solve_wo_func``: Total time taken by call(s) to
          :meth:`solve`, excluding time taken to compute functional
          value and related iteration statistics
        """

        # Construct tuple of status display column titles and set status
        # display strings
        hdrtxt = ['Itn', 'Fnc', 'DFid', u('Regℓ1')]
        hdrstr, fmtstr, nsep = common.solve_status_str(
            hdrtxt, fwdth0=type(self).fwiter, fprec=type(self).fpothr)

        # Print header and separator strings
        if self.opt['Verbose']:
            if self.opt['StatusHeader']:
                print(hdrstr)
                print("-" * nsep)

        # Reset timer
        self.timer.start(['solve', 'solve_wo_eval'])

        # Create process pool
        if self.nproc > 0:
            self.pool = mp.Pool(processes=self.nproc)

        for self.j in range(self.j, self.j + self.opt['MaxMainIter']):

            # Perform a set of update steps
            self.step()

            # Evaluate functional
            self.timer.stop('solve_wo_eval')
            fnev = self.evaluate()
            self.timer.start('solve_wo_eval')

            # Record iteration stats
            tk = self.timer.elapsed('solve')
            itst = self.IterationStats(*((self.j,) + fnev + (tk,)))
            self.itstat.append(itst)

            # Display iteration stats if Verbose option enabled
            if self.opt['Verbose']:
                print(fmtstr % itst[:-1])

            # Call callback function if defined
            if self.opt['Callback'] is not None:
                if self.opt['Callback'](self):
                    break

        # Clean up process pool
        if self.nproc > 0:
            self.pool.close()
            self.pool.join()

        # Increment iteration count
        self.j += 1

        # Record solve time
        self.timer.stop(['solve', 'solve_wo_eval'])

        # Print final separator string if Verbose option enabled
        if self.opt['Verbose'] and self.opt['StatusHeader']:
            print("-" * nsep)

        # Return final dictionary
        return self.getdict()



    def getdict(self, crop=True):
        """Get final dictionary. If ``crop`` is ``True``, apply
        :func:`.cnvrep.bcrop` to returned array.
        """

        global mp_D_Y0
        D = mp_D_Y0
        if crop:
            D = cr.bcrop(D, self.dstep.cri.dsz, self.dstep.cri.dimN)
        return D



    def getcoef(self):
        """Get final coefficient map array."""

        global mp_Z_Y1
        return np.swapaxes(mp_Z_Y1, 0, self.xstep.cri.axisK+1)[0]



    def evaluate(self):
        """Evaluate functional value of previous iteration."""

        if self.opt['AccurateDFid']:
            DX = self.reconstruct()
            W = self.dstep.W
            S = self.dstep.S
        else:
            W = mp_W
            S = mp_S
            Xf = mp_Zf
            Df = mp_Df
            DX = ifftn(sl.inner(Df[np.newaxis, ...], Xf,
                                axis=self.xstep.cri.axisM+1),
                       self.xstep.cri.Nv, np.array(self.xstep.cri.axisN) + 1)

        dfd = (np.linalg.norm(W * (DX - S))**2) / 2.0
        rl1 = np.sum(np.abs(self.getcoef()))
        obj = dfd + self.xstep.lmbda*rl1

        return (obj, dfd, rl1)



    def getitstat(self):
        """Get iteration stats as named tuple of arrays instead of array
        of named tuples.
        """

        return transpose_ntpl_list(self.itstat)
