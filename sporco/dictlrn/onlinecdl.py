# -*- coding: utf-8 -*-
# Copyright (C) 2018-2019 by Cristina Garcia-Cardona <cgarciac@lanl.gov>
#                            Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Online dictionary learning based on CBPDN sparse coding"""

from __future__ import print_function, absolute_import

import copy
import numpy as np

from sporco import common
from sporco.admm import cbpdn
from sporco import cuda
from sporco.util import u, Timer
from sporco.linalg import inner
from sporco.cnvrep import (DictionarySize, stdformD, Pcn, getPcn,
                           CDU_ConvRepIndexing, zpad, mskWshape)
from sporco.dictlrn import dictlrn
from sporco.array import transpose_ntpl_list
from sporco.fft import rfftn, irfftn, byte_aligned, empty_aligned


__author__ = """\n""".join(['Cristina Garcia-Cardona <cgarciac@lanl.gov>',
                            'Brendt Wohlberg <brendt@ieee.org>'])



class OnlineConvBPDNDictLearn(common.IterativeSolver):
    r"""
    Stochastic gradient descent (SGD) based online convolutional
    dictionary learning, as proposed in :cite:`liu-2018-first`.

    |

    .. inheritance-diagram:: OnlineConvBPDNDictLearn
       :parts: 2

    |
    """


    class Options(dictlrn.DictLearn.Options):
        r"""Online CBPDN dictionary learning algorithm options.

        Options:

          ``Verbose`` : Flag determining whether iteration status is
          displayed.

          ``StatusHeader`` : Flag determining whether status header and
          separator are displayed.

          ``IterTimer`` : Label of the timer to use for iteration times.

          ``DictSize`` : Dictionary size vector.

          ``DataType`` : Specify data type for solution variables,
          e.g. ``np.float32``.

          ``ZeroMean`` : Flag indicating whether the solution
          dictionary :math:`\{\mathbf{d}_m\}` should have zero-mean
          components.

          ``eta_a``, ``eta_b`` : Constants :math:`a` and :math:`b` used
          in setting the SGD step size, :math:`\eta`, which is set to
          :math:`a / (b + i)` where :math:`i` is the iteration index.
          See Sec. 3 (pg. 9) of :cite:`liu-2018-first`.

          ``CUDA_CBPDN`` : Flag indicating whether to use CUDA solver
          for CBPDN problem (see :ref:`cuda_package`)

          ``CBPDN`` : Options :class:`.admm.cbpdn.ConvBPDN.Options`.
        """

        defaults = {'Verbose': False, 'StatusHeader': True,
                    'IterTimer': 'solve', 'DictSize': None,
                    'DataType': None, 'ZeroMean': False, 'eta_a': 10.0,
                    'eta_b': 5.0, 'CUDA_CBPDN' : False,
                    'CBPDN': copy.deepcopy(cbpdn.ConvBPDN.Options.defaults)}


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              OnlineConvBPDNDictLearn algorithm options
            """

            dictlrn.DictLearn.Options.__init__(self, {
                'CBPDN': cbpdn.ConvBPDN.Options({
                    'MaxMainIter': 100,
                    'AutoRho': {'Period': 10, 'AutoScaling': False,
                                'RsdlRatio': 10.0, 'Scaling': 2.0,
                                'RsdlTarget': 1.0}})
                })

            if opt is None:
                opt = {}
            self.update(opt)


    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""

    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    """Fields in IterationStats associated with the objective function"""
    itstat_fields_alg = ('PrimalRsdl', 'DualRsdl', 'Rho', 'Cnstr',
                         'DeltaD', 'Eta')
    """Fields in IterationStats associated with the specific solver
    algorithm"""
    itstat_fields_extra = ()
    """Non-standard fields in IterationStats; see :meth:`itstat_extra`"""



    def __new__(cls, *args, **kwargs):
        """Create an OnlineConvBPDNDictLearn object and start its
        initialisation timer."""

        instance = super(OnlineConvBPDNDictLearn, cls).__new__(cls)
        instance.timer = Timer(['init', 'solve', 'solve_wo_eval'])
        instance.timer.start('init')
        return instance



    def __init__(self, D0, lmbda=None, opt=None, dimK=None, dimN=2):
        """
        Parameters
        ----------
        D0 : array_like
          Initial dictionary array
        lmbda : float
          Regularisation parameter
        opt : :class:`OnlineConvBPDNDictLearn.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of signal dimensions in signal array passed to
          :meth:`solve`. If there will only be a single input signal
          (e.g. if `S` is a 2D array representing a single image)
          `dimK` must be set to 0.
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        if opt is None:
            opt = OnlineConvBPDNDictLearn.Options()
        if not isinstance(opt, OnlineConvBPDNDictLearn.Options):
            raise TypeError('Parameter opt must be an instance of '
                            'OnlineConvBPDNDictLearn.Options')
        self.opt = opt

        if dimN != 2 and opt['CUDA_CBPDN']:
            raise ValueError('CUDA CBPDN solver can only be used when dimN=2')

        if opt['CUDA_CBPDN'] and cuda.device_count() == 0:
            raise ValueError('SPORCO-CUDA not installed or no GPU available')

        self.dimK = dimK
        self.dimN = dimN

        # DataType option overrides data type inferred from __init__
        # parameters of derived class
        self.set_dtype(opt, D0.dtype)

        # Initialise attributes representing algorithm parameter
        self.lmbda = lmbda
        self.eta_a = opt['eta_a']
        self.eta_b = opt['eta_b']
        self.set_attr('eta', opt['eta_a'] / opt['eta_b'],
                      dval=2.0, dtype=self.dtype)

        # Get dictionary size
        if self.opt['DictSize'] is None:
            self.dsz = D0.shape
        else:
            self.dsz = self.opt['DictSize']

        # Construct object representing problem dimensions
        self.cri = None

        # Normalise dictionary
        ds = DictionarySize(self.dsz, dimN)
        dimCd = ds.ndim - dimN - 1
        D0 = stdformD(D0, ds.nchn, ds.nflt, dimN).astype(self.dtype)
        self.D = Pcn(D0, self.dsz, (), dimN, dimCd, crp=True,
                        zm=opt['ZeroMean'])
        self.Dprv = self.D.copy()

        # Create constraint set projection function
        self.Pcn = getPcn(self.dsz, (), dimN, dimCd, crp=True,
                             zm=opt['ZeroMean'])

        # Initalise iterations stats list and iteration index
        self.itstat = []
        self.j = 0

        # Configure status display
        self.display_config()



    def solve(self, S, dimK=None):
        """Compute sparse coding and dictionary update for training
        data `S`."""

        # Use dimK specified in __init__ as default
        if dimK is None and self.dimK is not None:
            dimK = self.dimK

        # Start solve timer
        self.timer.start(['solve', 'solve_wo_eval'])

        # Solve CSC problem on S and do dictionary step
        self.init_vars(S, dimK)
        self.xstep(S, self.lmbda, dimK)
        self.dstep()

        # Stop solve timer
        self.timer.stop('solve_wo_eval')

        # Extract and record iteration stats
        self.manage_itstat()

        # Increment iteration count
        self.j += 1

        # Stop solve timer
        self.timer.stop('solve')

        # Return current dictionary
        return self.getdict()



    def init_vars(self, S, dimK):
        """Initalise variables required for sparse coding and dictionary
        update for training data `S`."""

        Nv = S.shape[0:self.dimN]
        if self.cri is None or Nv != self.cri.Nv:
            self.cri = CDU_ConvRepIndexing(self.dsz, S, dimK, self.dimN)
            if self.opt['CUDA_CBPDN']:
                if self.cri.Cd > 1 or self.cri.Cx > 1:
                    raise ValueError('CUDA CBPDN solver can only be used for '
                                     'single channel problems')
                if self.cri.K > 1:
                    raise ValueError('CUDA CBPDN solver can not be used with '
                                     'mini-batches')
            self.Df = byte_aligned(rfftn(self.D, self.cri.Nv,
                                         self.cri.axisN))
            self.Gf = empty_aligned(self.Df.shape, self.Df.dtype)
            self.Z = empty_aligned(self.cri.shpX, self.dtype)
        else:
            self.Df[:] = rfftn(self.D, self.cri.Nv, self.cri.axisN)



    def xstep(self, S, lmbda, dimK):
        """Solve CSC problem for training data `S`."""

        if self.opt['CUDA_CBPDN']:
            Z = cuda.cbpdn(self.D.squeeze(), S[..., 0], lmbda,
                           self.opt['CBPDN'])
            Z = Z.reshape(self.cri.Nv + (1, 1, self.cri.M,))
            self.Z[:] = np.asarray(Z, dtype=self.dtype)
            self.Zf = rfftn(self.Z, self.cri.Nv, self.cri.axisN)
            self.Sf = rfftn(S.reshape(self.cri.shpS), self.cri.Nv,
                               self.cri.axisN)
            self.xstep_itstat = None
        else:
            # Create X update object (external representation is expected!)
            xstep = cbpdn.ConvBPDN(self.D.squeeze(), S, lmbda,
                                   self.opt['CBPDN'], dimK=dimK,
                                   dimN=self.cri.dimN)
            xstep.solve()
            self.Sf = xstep.Sf
            self.setcoef(xstep.getcoef())
            self.xstep_itstat = xstep.itstat[-1] if xstep.itstat else None



    def setcoef(self, Z):
        """Set coefficient array."""

        # If the dictionary has a single channel but the input (and
        # therefore also the coefficient map array) has multiple
        # channels, the channel index and multiple image index have
        # the same behaviour in the dictionary update equation: the
        # simplest way to handle this is to just reshape so that the
        # channels also appear on the multiple image index.
        if self.cri.Cd == 1 and self.cri.C > 1:
            Z = Z.reshape(self.cri.Nv + (1,) + (self.cri.Cx*self.cri.K,) +
                          (self.cri.M,))
            if Z.shape != self.Z.shape:
                self.Z = empty_aligned(Z.shape, self.dtype)
        self.Z[:] = np.asarray(Z, dtype=self.dtype)
        self.Zf = rfftn(self.Z, self.cri.Nv, self.cri.axisN)



    def dstep(self):
        """Compute dictionary update for training data of preceding
        :meth:`xstep`.
        """

        # Compute X D - S
        Ryf = inner(self.Zf, self.Df, axis=self.cri.axisM) - self.Sf
        # Compute gradient
        gradf = inner(np.conj(self.Zf), Ryf, axis=self.cri.axisK)

        # If multiple channel signal, single channel dictionary
        if self.cri.C > 1 and self.cri.Cd == 1:
            gradf = np.sum(gradf, axis=self.cri.axisC, keepdims=True)

        # Update gradient step
        self.eta = self.eta_a / (self.j + self.eta_b)

        # Compute gradient descent
        self.Gf[:] = self.Df - self.eta * gradf
        self.G = irfftn(self.Gf, self.cri.Nv, self.cri.axisN)

        # Eval proximal operator
        self.Dprv[:] = self.D
        self.D[:] = self.Pcn(self.G)



    def manage_itstat(self):
        """Compute, record, and display iteration statistics."""

        # Extract and record iteration stats
        itst = self.iteration_stats()
        self.itstat.append(itst)
        self.display_status(self.fmtstr, itst)



    def getdict(self):
        """Get final dictionary."""

        return self.D



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return ()



    @classmethod
    def hdrtxt(cls):
        """Construct tuple of status display column title."""

        return ('Itn', 'X r', 'X s', u('X ρ'), 'D cnstr', 'D dlt', u('D η'))



    @classmethod
    def hdrval(cls):
        """Construct dictionary mapping display column title to
        IterationStats entries.
        """

        hdrmap = {'Itn': 'Iter', 'X r': 'PrimalRsdl', 'X s': 'DualRsdl',
                  u('X ρ'): 'Rho', 'D cnstr': 'Cnstr', 'D dlt': 'DeltaD',
                  u('D η'): 'Eta'}
        return hdrmap



    def iteration_stats(self):
        """Construct iteration stats record tuple."""

        tk = self.timer.elapsed(self.opt['IterTimer'])
        if self.xstep_itstat is None:
            objfn = (0.0,) * 3
            rsdl = (0.0,) * 2
            rho = (0.0,)
        else:
            objfn = (self.xstep_itstat.ObjFun, self.xstep_itstat.DFid,
                     self.xstep_itstat.RegL1)
            rsdl = (self.xstep_itstat.PrimalRsdl,
                    self.xstep_itstat.DualRsdl)
            rho = (self.xstep_itstat.Rho,)

        cnstr = np.linalg.norm(zpad(self.D, self.cri.Nv) - self.G)
        dltd = np.linalg.norm(self.D - self.Dprv)

        tpl = (self.j,) + objfn + rsdl + rho + (cnstr, dltd, self.eta) + \
              self.itstat_extra() + (tk,)
        return type(self).IterationStats(*tpl)



    def getitstat(self):
        """Get iteration stats as named tuple of arrays instead of
        array of named tuples.
        """

        return transpose_ntpl_list(self.itstat)



    def display_config(self):
        """Set up status display if option selected. NB: this method
        assumes that the first entry is the iteration count and the
        last is the rho value.
        """

        if self.opt['Verbose']:
            hdrtxt = type(self).hdrtxt()
            # Call utility function to construct status display formatting
            self.hdrstr, self.fmtstr, self.nsep = common.solve_status_str(
                hdrtxt, fwdth0=type(self).fwiter, fprec=type(self).fpothr)
        else:
            self.hdrstr, self.fmtstr, self.nsep = '', '', 0



    def display_start(self):
        """Start status display if option selected."""

        if self.opt['Verbose'] and self.opt['StatusHeader']:
            print(self.hdrstr)
            print("-" * self.nsep)



    def display_status(self, fmtstr, itst):
        """Display current iteration status as selection of fields from
        iteration stats tuple.
        """

        if self.opt['Verbose']:
            hdrtxt = type(self).hdrtxt()
            hdrval = type(self).hdrval()
            itdsp = tuple([getattr(itst, hdrval[col]) for col in hdrtxt])

            print(fmtstr % itdsp)



    def display_end(self):
        """Terminate status display if option selected."""

        if self.opt['Verbose'] and self.opt['StatusHeader']:
            print("-" * self.nsep)





class OnlineConvBPDNMaskDictLearn(OnlineConvBPDNDictLearn):
    r"""
    Stochastic gradient descent (SGD) based online convolutional
    dictionary learning with a spatial mask, as proposed in
    :cite:`liu-2018-first`.

    |

    .. inheritance-diagram:: OnlineConvBPDNMaskDictLearn
       :parts: 2

    |
    """

    class Options(OnlineConvBPDNDictLearn.Options):
        r"""Online masked CBPDN dictionary learning algorithm options.

        Options are the same as those of
        :class:`OnlineConvBPDNDictLearn.Options`, except for

          ``CBPDN`` : Options :class:`.admm.cbpdn.ConvBPDNMaskDcpl.Options`.
        """

        defaults = copy.deepcopy(OnlineConvBPDNDictLearn.Options.defaults)
        defaults.update({'CBPDN': copy.deepcopy(
            cbpdn.ConvBPDNMaskDcpl.Options.defaults)})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              OnlineConvBPDNMaskDictLearn algorithm options
            """

            OnlineConvBPDNDictLearn.Options.__init__(self, {
                'CBPDN': cbpdn.ConvBPDNMaskDcpl.Options({
                    'AutoRho': {'Period': 10, 'AutoScaling': False,
                                'RsdlRatio': 10.0, 'Scaling': 2.0,
                                'RsdlTarget': 1.0}})
                })

            if opt is None:
                opt = {}
            self.update(opt)



    def solve(self, S, W=None, dimK=None):
        """Compute sparse coding and dictionary update for training
        data `S`."""

        # Use dimK specified in __init__ as default
        if dimK is None and self.dimK is not None:
            dimK = self.dimK

        # Start solve timer
        self.timer.start(['solve', 'solve_wo_eval'])

        # Solve CSC problem on S and do dictionary step
        self.init_vars(S, dimK)
        if W is None:
            W = np.array([1.0], dtype=self.dtype)
        W = np.asarray(W.reshape(mskWshape(W, self.cri)), dtype=self.dtype)
        self.xstep(S, W, self.lmbda, dimK)
        self.dstep(W)

        # Stop solve timer
        self.timer.stop('solve_wo_eval')

        # Extract and record iteration stats
        self.manage_itstat()

        # Increment iteration count
        self.j += 1

        # Stop solve timer
        self.timer.stop('solve')

        # Return current dictionary
        return self.getdict()



    def xstep(self, S, W, lmbda, dimK):
        """Solve CSC problem for training data `S`."""

        if self.opt['CUDA_CBPDN']:
            Z = cuda.cbpdnmsk(self.D.squeeze(), S[..., 0], W.squeeze(), lmbda,
                              self.opt['CBPDN'])
            Z = Z.reshape(self.cri.Nv + (1, 1, self.cri.M,))
            self.Z[:] = np.asarray(Z, dtype=self.dtype)
            self.Zf = rfftn(self.Z, self.cri.Nv, self.cri.axisN)
            self.Sf = rfftn(S.reshape(self.cri.shpS), self.cri.Nv,
                               self.cri.axisN)
            self.xstep_itstat = None
        else:
            # Create X update object (external representation is expected!)
            xstep = cbpdn.ConvBPDNMaskDcpl(self.D.squeeze(), S, lmbda, W,
                                           self.opt['CBPDN'], dimK=dimK,
                                           dimN=self.cri.dimN)
            xstep.solve()
            self.Sf = rfftn(S.reshape(self.cri.shpS), self.cri.Nv,
                               self.cri.axisN)
            self.setcoef(xstep.getcoef())
            self.xstep_itstat = xstep.itstat[-1] if xstep.itstat else None



    def dstep(self, W):
        """Compute dictionary update for training data of preceding
        :meth:`xstep`.
        """

        # Compute residual X D - S in frequency domain
        Ryf = inner(self.Zf, self.Df, axis=self.cri.axisM) - self.Sf
        # Transform to spatial domain, apply mask, and transform back to
        # frequency domain
        Ryf[:] = rfftn(W * irfftn(Ryf, self.cri.Nv, self.cri.axisN),
                          None, self.cri.axisN)
        # Compute gradient
        gradf = inner(np.conj(self.Zf), Ryf, axis=self.cri.axisK)

        # If multiple channel signal, single channel dictionary
        if self.cri.C > 1 and self.cri.Cd == 1:
            gradf = np.sum(gradf, axis=self.cri.axisC, keepdims=True)

        # Update gradient step
        self.eta = self.eta_a / (self.j + self.eta_b)

        # Compute gradient descent
        self.Gf[:] = self.Df - self.eta * gradf
        self.G = irfftn(self.Gf, self.cri.Nv, self.cri.axisN)

        # Eval proximal operator
        self.Dprv[:] = self.D
        self.D[:] = self.Pcn(self.G)
