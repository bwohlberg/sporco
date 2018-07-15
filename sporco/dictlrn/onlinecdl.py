# -*- coding: utf-8 -*-
# Copyright (C) 2018 by Cristina Garcia-Cardona <cgarciac@lanl.gov>
#                       Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Online dictionary learning based on CBPDN sparse coding"""

from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import object

import copy
import collections
import numpy as np
from scipy import linalg

from sporco import util
from sporco import common
from sporco.util import u
import sporco.linalg as sl
import sporco.cnvrep as cr
from sporco.admm import cbpdn
from sporco.dictlrn import dictlrn
from sporco.common import _fix_nested_class_lookup


__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""



class OnlineConvBPDNDictLearn(common.BasicIterativeSolver):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: OnlineConvBPDNDictLearn
       :parts: 2

    |

    Stochastic gradient descent based online convolutional dictionary
    learning, as proposed in :cite:`liu-2018-first`.
    """


    class Options(dictlrn.DictLearn.Options):
        """Online CBPDN dictionary learning algorithm options.

        Options include all of those defined in
        :class:`.dictlrn.DictLearn.Options`, together with additional
        options:

          ``AccurateDFid`` : Flag determining whether data fidelity term is
          estimated from the value computed in the X update (``False``) or
          is computed after every outer iteration over an X update and a D
          update (``True``), which is slower but more accurate.

          ``DictSize`` : Dictionary size vector.

          ``CBPDN`` : Options :class:`.admm.cbpdn.ConvBPDN.Options`
        """

        defaults = copy.deepcopy(dictlrn.DictLearn.Options.defaults)
        del defaults['Callback']
        defaults.update({'DictSize': None, 'AccurateDFid': False,
                         'DataType': None,
            'CBPDN': copy.deepcopy(cbpdn.ConvBPDN.Options.defaults),
            'OCDL' : {'ZeroMean': False, 'eta_a': 10.0, 'eta_b': 5.0,
                      'DataType': None}})


        def __init__(self, opt=None):
            """Initialise OnlineConvBPDN dictionary learning algorithm
            options.
            """

            dictlrn.DictLearn.Options.__init__(self, {
                'CBPDN': cbpdn.ConvBPDN.Options({
                    'AutoRho': {'Period': 10, 'AutoScaling': False,
                    'RsdlRatio': 10.0, 'Scaling': 2.0, 'RsdlTarget': 1.0}})
                })

            if opt is None:
                opt = {}
            self.update(opt)


    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""

    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'Cnstr')
    """Fields in IterationStats associated with the objective function;
    see :meth:`eval_objfn`"""
    itstat_fields_extra = ()
    """Non-standard fields in IterationStats; see :meth:`itstat_extra`"""

    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), 'Cnstr')
    """Display column headers associated with the objective function;
    see :meth:`eval_objfn`"""
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1',
                     'Cnstr' : 'Cnstr'}
    """Dictionary mapping display column headers in :attr:`hdrtxt_objfn`
    to IterationStats entries"""



    def __new__(cls, *args, **kwargs):
        """Create an OnlineConvBPDNDictLearn object and start its
        initialisation timer."""

        # Initialise named tuple type for recording FISTA iteration statistics
        cls.IterationStats = collections.namedtuple('IterationStats',
                                                    cls.itstat_fields())
        # Apply _fix_nested_class_lookup function to class after creation
        _fix_nested_class_lookup(cls, nstnm='Options')

        instance = super(OnlineConvBPDNDictLearn, cls).__new__(cls)
        instance.timer = util.Timer(['init', 'solve', 'solve_wo_eval'])
        instance.timer.start('init')
        return instance



    def __init__(self, D0, S0, lmbda=None, opt=None, dimK=1, dimN=2):
        """
        Initialise an OnlineConvBPDNDictLearn object with problem size and
        options.

        Parameters
        ----------
        D0 : array_like
          Initial dictionary array
        S0 : array_like
          Signal array (dummy)
        lmbda : float
          Regularisation parameter
        opt : :class:`OnlineConvBPDNDictLearn.Options` object
          Algorithm options
        dimK : int, optional (default 1)
          Number of signal dimensions. If there is only a single input
          signal (e.g. if `S` is a 2D array representing a single image)
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

        # DataType option overrides data type inferred from __init__
        # parameters of derived class
        self.set_dtype(opt, S0.dtype)

        # Initialise attributes representing step parameter
        self.set_attr('eta', opt['OCDL', 'eta_a'] / opt['OCDL', 'eta_b'],
                      dval=2.0, dtype=self.dtype)
        self.set_attr('rho', opt['CBPDN', 'rho'], dval=1.0, dtype=self.dtype)

        # Get dictionary size
        if self.opt['DictSize'] is None:
            dsz = D0.shape
        else:
            dsz = self.opt['DictSize']

        # Construct object representing problem dimensions
        self.cri = cr.CDU_ConvRepIndexing(dsz, S0, dimK, dimN)

        # Normalise dictionary
        D0 = cr.Pcn(D0, dsz, self.cri.Nv, dimN, self.cri.dimCd, crp=True,
                    zm=opt['OCDL', 'ZeroMean'])

        # Create byte aligned arrays for FFT calls
        self.D = sl.pyfftw_empty_aligned(self.cri.shpD, dtype=self.dtype)
        # Initialize dictionary state
        D0_stdformD = cr.zpad(cr.stdformD(D0, self.cri.Cd, self.cri.M, dimN),
                              self.cri.Nv)
        self.D[:] = D0_stdformD.astype(self.dtype, copy=True)
        self.Df = sl.rfftn(self.D, None, self.cri.axisN)

        # Create constraint set projection function
        self.Pcn = cr.getPcn(dsz, self.cri.Nv, self.cri.dimN, self.cri.dimCd,
                             zm=opt['OCDL', 'ZeroMean'])

        # Create byte aligned arrays for FFT calls
        self.Gf = sl.pyfftw_rfftn_empty_aligned(self.D.shape, self.cri.axisN,
                                                self.dtype)

        # Configure algorithm parameters
        self.eta_a = opt['OCDL', 'eta_a']
        self.eta_b = opt['OCDL', 'eta_b']

        # For function evaluation
        self.Sf_last = None
        self.S_last = None
        self.lmbda = lmbda

        self.itstat = []
        self.j = 0

        # Open status display
        self.fmtstr, self.nsep = self.display_start()


    def update_shape(self, dsz, S, D_crop):
        """Update memory allocation and cropping
        functionality when the image size changes."""

        self.cri = cr.CDU_ConvRepIndexing(dsz, S)#, 1, dimN)
        # Update dictionary state
        D_stdformD = cr.zpad(cr.stdformD(D_crop, self.cri.Cd, self.cri.M, self.cri.dimN),
                              self.cri.Nv)
        # Create byte aligned arrays for FFT calls
        self.D = sl.pyfftw_empty_aligned(self.cri.shpD, dtype=self.dtype)
        self.D[:] = D_stdformD.astype(self.dtype, copy=True)
        self.Df = sl.rfftn(self.D, None, self.cri.axisN)

        # Create constraint set projection function
        self.Pcn = cr.getPcn(dsz, self.cri.Nv, self.cri.dimN, self.cri.dimCd,
                             zm=self.opt['OCDL', 'ZeroMean'])

        # Create byte aligned arrays for FFT calls
        self.Gf = sl.pyfftw_rfftn_empty_aligned(self.D.shape, self.cri.axisN,
                                                self.dtype)


    def solve(self, s_k):
        """Solve and compute statistics per iteration."""

        # Start solve timer
        self.timer.start('solve')

        self.step(s_k, self.lmbda)

        self.cnstr = linalg.norm((self.D - self.G))

        # Extract and record iteration stats
        itst = self.iteration_stats()
        self.itstat.append(itst)
        self.display_status(self.fmtstr, itst)

        # Increment iteration count
        self.j += 1

        # Stop solve timer
        self.timer.stop('solve')

        # Return current dictionary
        return self.getdict()



    def step(self, s_k, lmbda):
        """Do a single iteration over one image."""

        # Start solve timer
        self.timer.start('solve_wo_eval')

        D_crop = cr.bcrop(self.D, self.cri.dsz, self.cri.dimN)

        # Check if image size changed
        Nv = s_k.shape[0:self.cri.dimN]
        if Nv != self.cri.Nv:
            dsz = self.cri.dsz
            self.update_shape(dsz, s_k, D_crop)
            D_crop = cr.bcrop(self.D, self.cri.dsz, self.cri.dimN)

        # Create X update object (external representation is expected!)
        xstep = cbpdn.ConvBPDN(D_crop.squeeze(), s_k, lmbda,
                               self.opt['CBPDN'], dimK=self.cri.dimK,
                               dimN=self.cri.dimN)
        xstep.solve()

        self.setcoef(xstep.getcoef())
        self.rho = xstep.rho
        self.xstep_itstat = xstep.itstat[-1] if len(xstep.itstat) > 0 \
                                             else None

        # Compute X D - S
        Ryf = sl.inner(self.Zf, self.Df, axis=self.cri.axisM) - xstep.Sf
        # Compute gradient
        gradf = sl.inner(np.conj(self.Zf), Ryf, axis=self.cri.axisK)

        # If Multiple channel signal, single channel dictionary
        if self.cri.C > 1 and self.cri.Cd == 1:
            gradf = np.sum(gradf, axis=self.cri.axisC, keepdims=True)

        # Update gradient step
        self.eta = self.eta_a / (self.j + self.eta_b)

        # Compute gradient descent
        self.Gf[:] = self.Df - self.eta * gradf
        self.G = sl.irfftn(self.Gf, self.cri.Nv, self.cri.axisN)

        # Eval proximal operator
        self.D[:] = self.Pcn(self.G)
        self.Df[:] = sl.rfftn(self.D, None, self.cri.axisN)

        # Stop solve timer
        self.timer.start('solve_wo_eval')

        # For evaluation
        self.Sf_last = xstep.Sf
        self.S_last = s_k



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
        self.Z = np.asarray(Z, dtype=self.dtype)

        self.Zf = sl.rfftn(self.Z, self.cri.Nv, self.cri.axisN)



    def getdict(self, crop=True):
        """Get final dictionary. If ``crop`` is ``True``, apply
        :func:`.cnvrep.bcrop` to returned array.
        """

        D = self.D
        if crop:
            D = cr.bcrop(D, self.cri.dsz, self.cri.dimN)
        return D



    def reconstruct(self, D=None, X=None):
        """Reconstruct representation."""

        if D is None:
            D = self.getdict(crop=False)
        if X is None:
            X = self.Z
        Df = sl.rfftn(D, self.cri.Nv, self.cri.axisN)
        Xf = sl.rfftn(X, self.cri.Nv, self.cri.axisN)
        DXf = sl.inner(Df, Xf, axis=self.cri.axisM)
        return sl.irfftn(DXf, self.cri.Nv, self.cri.axisN)



    def evaluate(self):
        """Evaluate functional value of previous iteration."""

        D = self.getdict(crop=False)
        X = self.Z
        Df = sl.rfftn(D, self.cri.Nv, self.cri.axisN)
        Xf = sl.rfftn(X, self.cri.Nv, self.cri.axisN)
        Sf = self.Sf_last
        Ef = sl.inner(Df, Xf, axis=self.cri.axisM) - Sf
        dfd = sl.rfl2norm2(Ef, self.S_last.shape, axis=self.cri.axisN)/2.0
        rl1 = np.sum(np.abs(X))
        return dict(DFid=dfd, RegL1=rl1, ObjFun=dfd+self.lmbda*rl1)



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return ()



    @classmethod
    def itstat_fields(cls):
        """Construct tuple of field names used to initialise IterationStats
        named tuple.
        """

        return ('Iter',) + cls.itstat_fields_objfn + \
          ('Rho', 'Eta') + cls.itstat_fields_extra + ('Time',)



    @classmethod
    def hdrtxt(cls):
        """Construct tuple of status display column title."""

        return ('Itn',) + cls.hdrtxt_objfn + (u('ρ'), u('η'))



    @classmethod
    def hdrval(cls):
        """Construct dictionary mapping display column title to
        IterationStats entries.
        """

        dict = {'Itn': 'Iter'}
        dict.update(cls.hdrval_objfun)
        dict.update({u('ρ'): 'Rho', u('η') : 'Eta'})

        return dict



    def iteration_stats(self):
        """Construct iteration stats record tuple."""

        tk = self.timer.elapsed(self.opt['IterTimer'])
        if self.opt['AccurateDFid']:
            evl = self.evaluate()
            objfn = (evl['ObjFun'], evl['DFid'], evl['RegL1'])
        else:
            if self.xstep_itstat is None:
                objfn = (0.0,) * 3
            else:
                objfn = (self.xstep_itstat.ObjFun, self.xstep_itstat.DFid,
                         self.xstep_itstat.RegL1)

        tpl = (self.j,) + objfn + (self.cnstr, self.rho, self.eta) + \
              self.itstat_extra() + (tk,)
        return type(self).IterationStats(*tpl)



    def getitstat(self):
        """Get iteration stats as named tuple of arrays instead of array of
        named tuples.
        """

        return util.transpose_ntpl_list(self.itstat)



    def display_start(self):
        """Set up status display if option selected. NB: this method assumes
        that the first entry is the iteration count and the last is
        the rho value.
        """

        if self.opt['Verbose']:
            hdrtxt = type(self).hdrtxt()
            # Call utility function to construct status display formatting
            hdrstr, fmtstr, nsep = common.solve_status_str(
                hdrtxt, fwdth0=type(self).fwiter, fprec=type(self).fpothr)
            # Print header and separator strings
            if self.opt['StatusHeader']:
                print(hdrstr)
                print("-" * nsep)
        else:
            fmtstr, nsep = '', 0

        return fmtstr, nsep



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
