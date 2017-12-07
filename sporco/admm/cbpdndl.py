# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Dictionary learning based on CBPDN sparse coding"""

from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import object

import copy
import numpy as np

from sporco.util import u
import sporco.linalg as sl
import sporco.cnvrep as cr
from sporco.admm import cbpdn
from sporco.admm import ccmod
from sporco.admm import ccmodmd
from sporco.admm import dictlrn

__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class ConvBPDNDictLearn(dictlrn.DictLearn):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ConvBPDNDictLearn
       :parts: 2

    |

    Dictionary learning by alternating between sparse coding and
    dictionary update stages, using :class:`.admm.cbpdn.ConvBPDN` and
    :func:`.ConvCnstrMOD` respectively, with the coupling between
    stages as in :cite:`garcia-2017-subproblem`
    :cite:`garcia-2017-convolutional`. The sparse coding algorithm is
    as in :cite:`wohlberg-2014-efficient`, and the dictionary update
    supports algorithms from :cite:`wohlberg-2016-efficient` and
    :cite:`sorel-2016-fast`.


    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
       (1/2) \sum_k \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \right \|_2^2 + \lambda \sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1 \quad \text{such that}
       \quad \mathbf{d}_m \in C  \;\; \forall m \;,

    where :math:`C` is the feasible set consisting of filters with
    unit norm and constrained support, via interleaved alternation
    between the ADMM steps of the :class:`.admm.cbpdn.ConvBPDN` and
    :func:`.ConvCnstrMOD` problems. The multi-channel variants
    :cite:`wohlberg-2016-convolutional` supported by
    :class:`.admm.cbpdn.ConvBPDN` and :func:`.ConvCnstrMOD` are also
    supported.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \sum_k \|
       \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} - \mathbf{s}_k \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1`

       ``Cnstr`` : Constraint violation measure

       ``XPrRsdl`` : Norm of X primal residual

       ``XDlRsdl`` : Norm of X dual residual

       ``XRho`` : X penalty parameter

       ``DPrRsdl`` : Norm of D primal residual

       ``DDlRsdl`` : Norm of D dual residual

       ``DRho`` : D penalty parameter

       ``Time`` : Cumulative run time
    """


    class Options(dictlrn.DictLearn.Options):
        """CBPDN dictionary learning algorithm options.

        Options include all of those defined in
        :class:`.dictlrn.DictLearn.Options`, together with additional
        options:

          ``AccurateDFid`` : Flag determining whether data fidelity term is
          estimated from the value computed in the X update (``False``) or
          is computed after every outer iteration over an X update and a D
          update (``True``), which is slower but more accurate.

          ``DictSize`` : Dictionary size vector.

          ``CBPDN`` : Options :class:`.admm.cbpdn.ConvBPDN.Options`

          ``CCMOD`` : Options :func:`.admm.ccmod.ConvCnstrMODOptions`
        """

        defaults = copy.deepcopy(dictlrn.DictLearn.Options.defaults)
        defaults.update({'DictSize': None, 'AccurateDFid': False,
            'CBPDN': copy.deepcopy(cbpdn.ConvBPDN.Options.defaults)})


        def __init__(self, opt=None, method='cns'):
            """Initialise ConvBPDN dictionary learning algorithm options.

            Valid values for parameter ``method`` are documented in function
            :func:`.ConvCnstrMOD`.
            """

            self.defaults.update({'CCMOD': copy.deepcopy(
                ccmod.ConvCnstrMODOptions(method=method).defaults)})

            dictlrn.DictLearn.Options.__init__(self, {
                'CBPDN': cbpdn.ConvBPDN.Options({'MaxMainIter': 1,
                    'AutoRho': {'Period': 10, 'AutoScaling': False,
                    'RsdlRatio': 10.0, 'Scaling': 2.0, 'RsdlTarget': 1.0}}),
                'CCMOD': ccmod.ConvCnstrMODOptions({'MaxMainIter': 1,
                    'AutoRho': {'Period': 10, 'AutoScaling': False,
                    'RsdlRatio': 10.0, 'Scaling': 2.0, 'RsdlTarget': 1.0}},
                    method=method)
                })

            if opt is None:
                opt = {}
            self.update(opt)



    def __init__(self, D0, S, lmbda=None, opt=None, method='cns',
                 dimK=1, dimN=2):
        """
        Initialise a ConvBPDNDictLearn object with problem size and options.

        |

        **Call graph**

        .. image:: _static/jonga/cbpdndl_init.svg
           :width: 20%
           :target: _static/jonga/cbpdndl_init.svg

        |


        Parameters
        ----------
        D0 : array_like
          Initial dictionary array
        S : array_like
          Signal array
        lmbda : float
          Regularisation parameter
        opt : :class:`ConvBPDNDictLearn.Options` object
          Algorithm options
        method : string, optional (default 'cns')
          String selecting dictionary update solver. Valid values are
          documented in function :func:`.ConvCnstrMOD`.
        dimK : int, optional (default 1)
          Number of signal dimensions. If there is only a single input
          signal (e.g. if `S` is a 2D array representing a single image)
          `dimK` must be set to 0.
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        if opt is None:
            opt = ConvBPDNDictLearn.Options(method=method)
        self.opt = opt

        # Get dictionary size
        if self.opt['DictSize'] is None:
            dsz = D0.shape
        else:
            dsz = self.opt['DictSize']

        # Construct object representing problem dimensions
        cri = cr.CDU_ConvRepIndexing(dsz, S, dimK, dimN)

        # Normalise dictionary
        D0 = cr.Pcn(D0, dsz, cri.Nv, dimN, cri.dimCd, crp=True,
                    zm=opt['CCMOD', 'ZeroMean'])

        # Modify D update options to include initial values for Y and U
        opt['CCMOD'].update({'Y0': cr.zpad(
            cr.stdformD(D0, cri.C, cri.M, dimN), cri.Nv)})

        # Create X update object
        xstep = cbpdn.ConvBPDN(D0, S, lmbda, opt['CBPDN'], dimK=dimK,
                               dimN=dimN)

        # Create D update object
        dstep = ccmod.ConvCnstrMOD(None, S, dsz, opt['CCMOD'], method=method,
                                   dimK=dimK, dimN=dimN)

        # Configure iteration statistics reporting
        if self.opt['AccurateDFid']:
            isxmap = {'XPrRsdl': 'PrimalRsdl', 'XDlRsdl': 'DualRsdl',
                      'XRho': 'Rho'}
            evlmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1'}
        else:
            isxmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1',
                      'XPrRsdl': 'PrimalRsdl', 'XDlRsdl': 'DualRsdl',
                      'XRho': 'Rho'}
            evlmap = {}
        isc = dictlrn.IterStatsConfig(
            isfld=['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr', 'XPrRsdl',
                   'XDlRsdl', 'XRho', 'DPrRsdl', 'DDlRsdl', 'DRho', 'Time'],
            isxmap=isxmap,
            isdmap={'Cnstr':  'Cnstr', 'DPrRsdl': 'PrimalRsdl',
                    'DDlRsdl': 'DualRsdl', 'DRho': 'Rho'},
            evlmap=evlmap,
            hdrtxt=['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr', 'r_X', 's_X',
                    u('ρ_X'), 'r_D', 's_D', u('ρ_D')],
            hdrmap={'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
                    u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr', 'r_X': 'XPrRsdl',
                    's_X': 'XDlRsdl', u('ρ_X'): 'XRho', 'r_D': 'DPrRsdl',
                    's_D': 'DDlRsdl', u('ρ_D'): 'DRho'}
            )

        # Call parent constructor
        super(ConvBPDNDictLearn, self).__init__(xstep, dstep, opt, isc)



    def getdict(self, crop=True):
        """Get final dictionary. If ``crop`` is ``True``, apply
        :func:`.cnvrep.bcrop` to returned array.
        """

        return self.dstep.getdict(crop=crop)



    def reconstruct(self, D=None, X=None):
        """Reconstruct representation."""

        if D is None:
            D = self.getdict(crop=False)
        if X is None:
            X = self.getcoef()
        Df = sl.rfftn(D, self.xstep.cri.Nv, self.xstep.cri.axisN)
        Xf = sl.rfftn(X, self.xstep.cri.Nv, self.xstep.cri.axisN)
        DXf = sl.inner(Df, Xf, axis=self.xstep.cri.axisM)
        return sl.irfftn(DXf, self.xstep.cri.Nv, self.xstep.cri.axisN)



    def evaluate(self):
        """Evaluate functional value of previous iteration."""

        if self.opt['AccurateDFid']:
            D = self.dstep.var_y()
            X = self.xstep.var_y()
            Df = sl.rfftn(D, self.xstep.cri.Nv, self.xstep.cri.axisN)
            Xf = sl.rfftn(X, self.xstep.cri.Nv, self.xstep.cri.axisN)
            Sf = self.xstep.Sf
            Ef = sl.inner(Df, Xf, axis=self.xstep.cri.axisM) - Sf
            dfd = sl.rfl2norm2(Ef, self.xstep.S.shape,
                               axis=self.xstep.cri.axisN)/2.0
            rl1 = np.sum(np.abs(X))
            return dict(DFid=dfd, RegL1=rl1, ObjFun=dfd+self.xstep.lmbda*rl1)
        else:
            return None






class ConvBPDNMaskDcplDictLearn(dictlrn.DictLearn):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ConvBPDNMaskDcplDictLearn
       :parts: 2

    |

    Dictionary learning by alternating between sparse coding and
    dictionary update stages, using :class:`.ConvBPDNMaskDcpl` and
    :func:`.ConvCnstrMODMaskDcpl` respectively, with the coupling between
    stages as in :cite:`garcia-2017-subproblem`
    :cite:`garcia-2017-convolutional`. The sparse coding algorithm is
    as in :cite:`heide-2015-fast`, and the dictionary update
    supports algorithms from :cite:`garcia-2017-convolutional`.


    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
       (1/2) \sum_k \left \|  W (\sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k ) \right \|_2^2 + \lambda \sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1 \quad \text{such that}
       \quad \mathbf{d}_m \in C \;\; \forall m \;,

    where :math:`C` is the feasible set consisting of filters with
    unit norm and constrained support, via interleaved alternation
    between the ADMM steps of the :class:`.ConvBPDNMaskDcpl` and
    :func:`.ConvCnstrMODMaskDcpl` problems. The multi-channel variants
    :cite:`wohlberg-2016-convolutional` supported by
    :class:`.ConvBPDNMaskDcpl` and :func:`.ConvCnstrMODMaskDcpl` are
    also supported.

    After termination of the :meth:`solve` method, attribute :attr:`itstat`
    is a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \sum_k \|
       W (\sum_m \mathbf{d}_m * \mathbf{x}_{k,m} - \mathbf{s}_k) \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1`

       ``Cnstr`` : Constraint violation measure

       ``XPrRsdl`` : Norm of X primal residual

       ``XDlRsdl`` : Norm of X dual residual

       ``XRho`` : X penalty parameter

       ``DPrRsdl`` : Norm of D primal residual

       ``DDlRsdl`` : Norm of D dual residual

       ``DRho`` : D penalty parameter

       ``Time`` : Cumulative run time
    """


    class Options(dictlrn.DictLearn.Options):
        """CBPDN dictionary learning algorithm options.

        Options include all of those defined in
        :class:`.dictlrn.DictLearn.Options`, together with additional
        options:

          ``AccurateDFid`` : Flag determining whether data fidelity term is
          estimated from the value computed in the X update (``False``) or
          is computed after every outer iteration over an X update and a D
          update (``True``), which is slower but more accurate.

          ``DictSize`` : Dictionary size vector.

          ``CBPDN`` : Options :class:`.ConvBPDNMaskDcpl.Options`

          ``CCMOD`` : Options :func:`.ConvCnstrMODMaskDcplOptions`
        """

        defaults = copy.deepcopy(dictlrn.DictLearn.Options.defaults)
        defaults.update({'DictSize': None, 'AccurateDFid': False,
            'CBPDN': copy.deepcopy(cbpdn.ConvBPDNMaskDcpl.Options.defaults)})


        def __init__(self, opt=None, method='cns'):
            """Initialise ConvBPDNMaskDcpl dictionary learning algorithm
            options.

            Valid values for parameter ``method`` are documented in function
            :func:`.ConvCnstrMODMaskDcpl`.
            """

            self.defaults.update({'CCMOD': copy.deepcopy(
                ccmodmd.ConvCnstrMODMaskDcplOptions(method=method).defaults)})

            dictlrn.DictLearn.Options.__init__(self, {
                'CBPDN': cbpdn.ConvBPDNMaskDcpl.Options({'MaxMainIter': 1,
                    'AutoRho': {'Period': 10, 'AutoScaling': False,
                    'RsdlRatio': 10.0, 'Scaling': 2.0, 'RsdlTarget': 1.0}}),
                'CCMOD': ccmodmd.ConvCnstrMODMaskDcplOptions(
                    {'MaxMainIter': 1, 'AutoRho': {'Period': 10,
                    'AutoScaling': False, 'RsdlRatio': 10.0,
                    'Scaling': 2.0, 'RsdlTarget': 1.0}}, method=method)
                })

            if opt is None:
                opt = {}
            self.update(opt)



    def __init__(self, D0, S, lmbda, W, opt=None, method='cns',
                 dimK=1, dimN=2):
        """
        Initialise a ConvBPDNMaskDcplDictLearn object with problem size and
        options.

        |

        **Call graph**

        .. image:: _static/jonga/cbpdnmddl_init.svg
           :width: 20%
           :target: _static/jonga/cbpdnmddl_init.svg

        |


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
          data layouts).
        opt : :class:`ConvBPDNMaskDcplDictLearn.Options` object
          Algorithm options
        method : string, optional (default 'cns')
          String selecting dictionary update solver. Valid values are
          documented in function :func:`.ConvCnstrMODMaskDcpl`.
        dimK : int, optional (default 1)
          Number of signal dimensions. If there is only a single input
          signal (e.g. if `S` is a 2D array representing a single image)
          `dimK` must be set to 0.
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        if opt is None:
            opt = ConvBPDNMaskDcplDictLearn.Options(method=method)
        self.opt = opt

        # Get dictionary size
        if self.opt['DictSize'] is None:
            dsz = D0.shape
        else:
            dsz = self.opt['DictSize']

        # Construct object representing problem dimensions
        cri = cr.CDU_ConvRepIndexing(dsz, S, dimK, dimN)

        # Normalise dictionary
        D0 = cr.Pcn(D0, dsz, cri.Nv, dimN, cri.dimCd, crp=True,
                    zm=opt['CCMOD', 'ZeroMean'])

        # Modify D update options to include initial values for Y and U
        if cri.C == cri.Cd:
            Y0b0 = np.zeros(cri.Nv + (cri.C, 1, cri.K))
        else:
            Y0b0 = np.zeros(cri.Nv + (1, 1, cri.C * cri.K))
        Y0b1 = cr.zpad(cr.stdformD(D0, cri.Cd, cri.M, dimN), cri.Nv)
        if method == 'cns':
            Y0 = Y0b1
        else:
            Y0 = np.concatenate((Y0b0, Y0b1), axis=cri.axisM)
        opt['CCMOD'].update({'Y0': Y0})

        # Create X update object
        xstep = cbpdn.ConvBPDNMaskDcpl(D0, S, lmbda, W, opt['CBPDN'],
                                       dimK=dimK, dimN=dimN)

        # Create D update object
        dstep = ccmodmd.ConvCnstrMODMaskDcpl(None, S, W, dsz, opt['CCMOD'],
                                    method=method, dimK=dimK, dimN=dimN)

        # Configure iteration statistics reporting
        if self.opt['AccurateDFid']:
            isxmap = {'XPrRsdl': 'PrimalRsdl', 'XDlRsdl': 'DualRsdl',
                      'XRho': 'Rho'}
            evlmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1'}
        else:
            isxmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1',
                      'XPrRsdl': 'PrimalRsdl', 'XDlRsdl': 'DualRsdl',
                      'XRho': 'Rho'}
            evlmap = {}
        isc = dictlrn.IterStatsConfig(
            isfld=['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr', 'XPrRsdl',
                   'XDlRsdl', 'XRho', 'DPrRsdl', 'DDlRsdl', 'DRho', 'Time'],
            isxmap=isxmap,
            isdmap={'Cnstr':  'Cnstr', 'DPrRsdl': 'PrimalRsdl',
                    'DDlRsdl': 'DualRsdl', 'DRho': 'Rho'},
            evlmap=evlmap,
            hdrtxt=['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr', 'r_X', 's_X',
                    u('ρ_X'), 'r_D', 's_D', u('ρ_D')],
            hdrmap={'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
                    u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr', 'r_X': 'XPrRsdl',
                    's_X': 'XDlRsdl', u('ρ_X'): 'XRho', 'r_D': 'DPrRsdl',
                    's_D': 'DDlRsdl', u('ρ_D'): 'DRho'}
            )

        # Call parent constructor
        super(ConvBPDNMaskDcplDictLearn, self).__init__(xstep, dstep,
                                                        opt, isc)



    def getdict(self, crop=True):
        """Get final dictionary. If ``crop`` is ``True``, apply
        :func:`.cnvrep.bcrop` to returned array.
        """

        return self.dstep.getdict(crop=crop)



    def reconstruct(self, D=None, X=None):
        """Reconstruct representation."""

        if D is None:
            D = self.getdict(crop=False)
        if X is None:
            X = self.getcoef()
        Df = sl.rfftn(D, self.xstep.cri.Nv, self.xstep.cri.axisN)
        Xf = sl.rfftn(X, self.xstep.cri.Nv, self.xstep.cri.axisN)
        DXf = sl.inner(Df, Xf, axis=self.xstep.cri.axisM)
        return sl.irfftn(DXf, self.xstep.cri.Nv, self.xstep.cri.axisN)



    def evaluate(self):
        """Evaluate functional value of previous iteration"""

        if self.opt['AccurateDFid']:
            DX = self.reconstruct()
            S = self.xstep.S
            dfd = (np.linalg.norm(self.xstep.W * (DX - S))**2) / 2.0
            X = self.xstep.var_y1()
            rl1 = np.sum(np.abs(X))
            return dict(DFid=dfd, RegL1=rl1, ObjFun=dfd+self.xstep.lmbda*rl1)
        else:
            return None
