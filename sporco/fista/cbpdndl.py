# -*- coding: utf-8 -*-
# Copyright (C) 2016-2017 by Brendt Wohlberg <brendt@ieee.org>
#                            Cristina Garcia-Cardona <cgarciac@lanl.gov>
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
from sporco.fista import ccmod
from sporco.admm import dictlrn
import sporco.admm.cbpdn as Acbpdn
from sporco.fista import cbpdn as Fcbpdn

__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""


class ConvBPDNDictLearn(dictlrn.DictLearn):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ConvBPDNDictLearn
       :parts: 2

    |

    Dictionary learning based on ConvBPDN (FISTA) and ConvCnstrMOD
    (FISTA) :cite:`garcia-2017-convolutional`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
       (1/2) \sum_k \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \right \|_2^2 + \lambda \sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1 \quad \text{such that}
       \quad \mathbf{d}_m \in C \;\;,

    where :math:`C` is the feasible set consisting of filters with
    unit norm and constrained support, via interleaved alternation
    between FISTA steps of the :class:`.fista.cbpdn.ConvBPDN` and
    :class:`.ConvCnstrMOD` problems. The multi-channel variants
    supported by :class:`.ConvCnstrMOD` are also supported.

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

       ``X_F_Btrack`` : Value of objective function for CSC problem

       ``X_Q_Btrack`` : Value of Quadratic approximation for CSC problem

       ``X_ItBt`` : Number of iterations in bactracking for CSC problem

       ``X_L`` : Inverse of gradient step parameter for CSC problem

       ``D_F_Btrack`` : Value of objective function for CDU problem

       ``D_Q_Btrack`` : Value of Quadratic approximation for CDU problem

       ``D_ItBt`` : Number of iterations in bactracking for CDU problem

       ``D_L`` : Inverse of gradient step parameter for CDU problem

       ``Time`` : Cumulative run time
    """


    class Options(dictlrn.DictLearn.Options):
        """ConvBPDNDictLearn dictionary learning algorithm options.

        Options include all of those defined in
        :class:`.dictlrn.DictLearn.Options`, together with
        additional options:

          ``AccurateDFid`` : Flag determining whether data fidelity term is
          estimated from the value computed in the X update (``False``) or is
          computed after every outer iteration over an X update and a D
          update (``True``), which is slower but more accurate.

          ``DictSize`` : Dictionary size vector.

          ``CBPDN`` : Options :class:`.cbpdn.GenericConvBPDN.Options`

          ``CCMOD`` : Options :func:`.ccmod.ConvCnstrMOD.Options`
        """

        defaults = copy.deepcopy(dictlrn.DictLearn.Options.defaults)
        defaults.update({'DictSize' : None, 'AccurateDFid' : False,
                         'CBPDN' : copy.deepcopy(
                             Fcbpdn.ConvBPDN.Options.defaults)})


        def __init__(self, opt=None):
            """Initialise ConvDictLearnFista dictionary learning algorithm
            options.
            """

            self.defaults.update({'CCMOD' : copy.deepcopy(
                ccmod.ConvCnstrMOD.Options.defaults)})

            dictlrn.DictLearn.Options.__init__(self, {
                'CBPDN': Fcbpdn.ConvBPDN.Options(
                    {'MaxMainIter': 1,
                     'BackTrack': {'Eta': 1.2, 'MaxIter': 50}}),
                'CCMOD': ccmod.ConvCnstrMOD.Options({'MaxMainIter': 1,
                    'BackTrack': {'Eta': 1.2, 'MaxIter': 50}})
                })

            if opt is None:
                opt = {}
            self.update(opt)



    def __init__(self, D0, S, lmbda=None, opt=None, dimK=1, dimN=2):
        """
        Initialise a ConvBPDNDictLearn object with problem size and options.


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
        dimK : int, optional (default 1)
          Number of signal dimensions. If there is only a single input
          signal (e.g. if `S` is a 2D array representing a single image)
          `dimK` must be set to 0.
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        if opt is None:
            opt = ConvBPDNDictLearn.Options()
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

        # Modify D update options to include initial values for X
        opt['CCMOD'].update({'X0' : cr.zpad(
            cr.stdformD(D0, cri.C, cri.M, dimN), cri.Nv)})

        # Create X update object
        xstep = Fcbpdn.ConvBPDN(D0, S, lmbda, opt['CBPDN'],
                                dimK=dimK, dimN=dimN)

        # Create D update object
        dstep = ccmod.ConvCnstrMOD(None, S, dsz, opt['CCMOD'],
                                   dimK=dimK, dimN=dimN)

        print("L xstep in cbpdndl: ", xstep.L)
        print("L dstep in cbpdndl: ", dstep.L)

        # Configure iteration statistics reporting
        isfld = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr']
        hdrtxt = ['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr']
        hdrmap = {'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
                  u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr'}

        if self.opt['AccurateDFid']:
            isxmap = {'X_F_Btrack': 'F_Btrack', 'X_Q_Btrack': 'Q_Btrack',
                      'X_ItBt': 'IterBTrack', 'X_L': 'L',
                      'X_Rsdl': 'Rsdl'}
            evlmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1'}
        else:
            isxmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1',
                      'X_F_Btrack': 'F_Btrack', 'X_Q_Btrack': 'Q_Btrack',
                      'X_ItBt': 'IterBTrack', 'X_L': 'L',
                      'X_Rsdl': 'Rsdl'}
            evlmap = {}

        # If Backtracking enabled in xstep display the BT variables also
        if xstep.opt['BackTrack', 'Enabled']:
            isfld.extend(['X_F_Btrack', 'X_Q_Btrack', 'X_ItBt', 'X_L',
                          'X_Rsdl'])
            hdrtxt.extend(['F_X', 'Q_X', 'It_X', 'L_X'])
            hdrmap.update({'F_X': 'X_F_Btrack','Q_X': 'X_Q_Btrack',
                           'It_X': 'X_ItBt', 'L_X': 'X_L'})
        else: # Add just L value to xstep display
            isfld.extend(['X_L', 'X_Rsdl'])
            hdrtxt.append('L_X')
            hdrmap.update({'L_X': 'X_L'})

        isdmap = {'Cnstr': 'Cnstr', 'D_F_Btrack': 'F_Btrack',
                  'D_Q_Btrack': 'Q_Btrack', 'D_ItBt': 'IterBTrack',
                  'D_L': 'L', 'D_Rsdl': 'Rsdl'}

        # If Backtracking enabled in dstep display the BT variables also
        if dstep.opt['BackTrack', 'Enabled']:
            isfld.extend(['D_F_Btrack', 'D_Q_Btrack', 'D_ItBt', 'D_L',
                          'D_Rsdl', 'Time'])
            hdrtxt.extend(['F_D', 'Q_D', 'It_D', 'L_D'])
            hdrmap.update({'F_D': 'D_F_Btrack', 'Q_D': 'D_Q_Btrack',
                           'It_D': 'D_ItBt', 'L_D': 'D_L'})
        else: # Add just L value to dstep display
            isfld.extend(['D_L', 'D_Rsdl', 'Time'])
            hdrtxt.append('L_D')
            hdrmap.update({'L_D': 'D_L'})

        isc = dictlrn.IterStatsConfig(isfld=isfld, isxmap=isxmap,
                                      isdmap=isdmap, evlmap=evlmap,
                                      hdrtxt=hdrtxt, hdrmap=hdrmap)

        # Call parent constructor
        super(ConvBPDNDictLearn, self).__init__(xstep, dstep, opt, isc)


    def getdict(self, crop=True):
        """Get final dictionary. If ``crop`` is ``True``, apply
        :func:`.cnvrep.bcrop` to returned array.
        """

        return self.dstep.getdict(crop=crop)



    def evaluate(self):
        """Evaluate functional value of previous iteration"""

        if self.opt['AccurateDFid']:
            D = self.getdict(crop=False)
            X = self.getcoef()
            Df = sl.rfftn(D, self.xstep.cri.Nv, self.xstep.cri.axisN)
            Xf = sl.rfftn(X, self.xstep.cri.Nv, self.xstep.cri.axisN)
            Sf = self.xstep.Sf
            Ef = np.sum(Df * Xf, axis=self.xstep.cri.axisM, keepdims=True) - Sf
            dfd = sl.rfl2norm2(Ef, self.xstep.S.shape,
                               axis=self.xstep.cri.axisN)/2.0
            rl1 = np.sum(np.abs(X))
            return dict(DFid=dfd, RegL1=rl1, ObjFun=dfd+self.xstep.lmbda*rl1)
        else:
            return None



class MixConvBPDNDictLearn(dictlrn.DictLearn):

    r"""**Class inheritance structure**

    .. inheritance-diagram:: MixConvBPDNDictLearn
       :parts: 2

    |

    Dictionary learning based on ConvBPDN (ADMM) and ConvCnstrMOD (FISTA)
    :cite:`garcia-2017-convolutional`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
       (1/2) \sum_k \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \right \|_2^2 + \lambda \sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1 \quad \text{such that}
       \quad \mathbf{d}_m \in C \;\;,

    where :math:`C` is the feasible set consisting of filters with
    unit norm and constrained support, via interleaved alternation
    between ADMM steps of the :class:`.admm.cbpdn.ConvBPDN` and FISTA steps
    :class:`.ConvCnstrMOD` problems. The multi-channel variants
    supported by :class:`.ConvCnstrMOD` are also supported.

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

       ``D_F_Btrack`` : Value of objective function for CDU problem

       ``D_Q_Btrack`` : Value of Quadratic approximation for CDU problem

       ``D_ItBt`` : Number of iterations in bactracking for CDU problem

       ``D_L`` : Inverse of gradient step parameter for CDU problem

       ``Time`` : Cumulative run time
    """


    class Options(dictlrn.DictLearn.Options):
        """MixConvBPDNDictLearn dictionary learning algorithm options.

        Options include all of those defined in
        :class:`.dictlrn.DictLearn.Options`, together with
        additional options:

          ``AccurateDFid`` : Flag determining whether data fidelity term is
          estimated from the value computed in the X update (``False``) or is
          computed after every outer iteration over an X update and a D
          update (``True``), which is slower but more accurate.

          ``DictSize`` : Dictionary size vector.

          ``CBPDN`` : Options :class:`.admm.cbpdn.ConvBPDN.Options`

          ``CCMOD`` : Options :func:`.ccmod.ConvCnstrMOD.Options`
        """

        defaults = copy.deepcopy(dictlrn.DictLearn.Options.defaults)
        defaults.update({'DictSize' : None, 'AccurateDFid' : False,
                        'CBPDN' : copy.deepcopy(
                            Acbpdn.ConvBPDN.Options.defaults)})


        def __init__(self, opt=None):
            """Initialise ConvBPDNAdmm_DictLearnFista dictionary learning
            algorithm options.
            """

            self.defaults.update({'CCMOD' : copy.deepcopy(
                ccmod.ConvCnstrMOD.Options.defaults)})

            dictlrn.DictLearn.Options.__init__(self, {
                    'CBPDN': Acbpdn.ConvBPDN.Options({'MaxMainIter': 1,
                        'AutoRho': {'Period': 10, 'AutoScaling': False,
                        'RsdlRatio': 10.0, 'Scaling': 2.0,
                        'RsdlTarget': 1.0}}),
                    'CCMOD': ccmod.ConvCnstrMOD.Options(
                        {'MaxMainIter': 1,
                        'BackTrack': {'Eta': 1.2, 'MaxIter': 50}})
                })

            if opt is None:
                opt = {}
            self.update(opt)



    def __init__(self, D0, S, lmbda=None, opt=None, dimK=1, dimN=2):
        """
        Initialise a MixConvBPDNDictLearn object with problem size
        and options.


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
        dimK : int, optional (default 1)
          Number of signal dimensions. If there is only a single input
          signal (e.g. if `S` is a 2D array representing a single image)
          `dimK` must be set to 0.
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        if opt is None:
            opt = MixConvBPDNDictLearn.Options()
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
        opt['CCMOD'].update({'X0' : cr.zpad(
            cr.stdformD(D0, cri.C, cri.M, dimN), cri.Nv)})

        # Create X update object
        xstep = Acbpdn.ConvBPDN(D0, S, lmbda, opt['CBPDN'], dimK=dimK,
                                dimN=dimN)

        # Create D update object
        dstep = ccmod.ConvCnstrMOD(None, S, dsz, opt['CCMOD'],
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

        if dstep.opt['BackTrack', 'Enabled']:
            isfld = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr', 'XPrRsdl',
                     'XDlRsdl', 'XRho', 'D_F_Btrack', 'D_Q_Btrack', 'D_ItBt',
                     'D_L', 'Time']
            isdmap = {'Cnstr':  'Cnstr', 'D_F_Btrack': 'F_Btrack',
                      'D_Q_Btrack': 'Q_Btrack', 'D_ItBt': 'IterBTrack',
                      'D_L': 'L'}
            hdrtxt = ['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr', 'r_X', 's_X',
                      u('ρ_X'), 'F_D', 'Q_D', 'It_D', 'L_D']
            hdrmap={'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
                    u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr', 'r_X': 'XPrRsdl',
                    's_X': 'XDlRsdl', u('ρ_X'): 'XRho', 'F_D': 'D_F_Btrack',
                    'Q_D': 'D_Q_Btrack', 'It_D': 'D_ItBt', 'L_D': 'D_L'}

        else:
            isfld = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr', 'XPrRsdl',
                     'XDlRsdl', 'XRho', 'D_L', 'Time']
            isdmap={'Cnstr':  'Cnstr', 'D_L': 'L'}
            hdrtxt=['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr', 'r_X', 's_X',
                    u('ρ_X'), 'L_D']
            hdrmap={'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
                    u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr', 'r_X': 'XPrRsdl',
                    's_X': 'XDlRsdl', u('ρ_X'): 'XRho', 'L_D': 'D_L'}

        isc = dictlrn.IterStatsConfig(isfld=isfld, isxmap=isxmap,
                                      isdmap=isdmap, evlmap=evlmap,
                                      hdrtxt=hdrtxt, hdrmap=hdrmap)

        # Call parent constructor
        super(MixConvBPDNDictLearn, self).__init__(xstep, dstep, opt,
                                                          isc)

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
            D = self.getdict(crop=False)
            X = self.getcoef()
            Df = sl.rfftn(D, self.xstep.cri.Nv, self.xstep.cri.axisN)
            Xf = sl.rfftn(X, self.xstep.cri.Nv, self.xstep.cri.axisN)
            Sf = self.xstep.Sf
            Ef = np.sum(Df * Xf, axis=self.xstep.cri.axisM, keepdims=True) - Sf
            dfd = sl.rfl2norm2(Ef, self.xstep.S.shape,
                               axis=self.xstep.cri.axisN)/2.0
            rl1 = np.sum(np.abs(X))
            return dict(DFid=dfd, RegL1=rl1, ObjFun=dfd+self.xstep.lmbda*rl1)
        else:
            return None



class MixConvBPDNMaskDcplDictLearn(dictlrn.DictLearn):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: MixConvBPDNMaskDcplDictLearn
       :parts: 2

    |

    Dictionary learning based on ConvBPDNMaskDcpl (ADMM) and
    ConvCnstrMODMaskDcpl (FISTA) :cite:`garcia-2017-convolutional`.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
       (1/2) \sum_k \left \|  W (\sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k ) \right \|_2^2 + \lambda \sum_k \sum_m
       \| \mathbf{x}_{k,m} \|_1 \quad \text{such that}
       \quad \mathbf{d}_m \in C \;\; \forall m \;,

    where :math:`C` is the feasible set consisting of filters with
    unit norm and constrained support, via interleaved alternation
    between ADMM steps of the :class:`.admm.cbpdn.ConvBPDNMaskDcpl` and
    FISTA steps :class:`.ConvCnstrMODMaskDcpl` problems. The
    multi-channel variants supported by :class:`.admm.cbpdn.ConvBPDNMaskDcpl`
    and :func:`.ConvCnstrMODMaskDcpl` are also supported.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
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

       ``D_F_Btrack`` : Value of objective function for CDU problem

       ``D_Q_Btrack`` : Value of Quadratic approximation for CDU problem

       ``D_ItBt`` : Number of iterations in bactracking for CDU problem

       ``D_L`` : Inverse of gradient step parameter for CDU problem

       ``Time`` : Cumulative run time
    """

    class Options(dictlrn.DictLearn.Options):
        """MixConvBPDNMaskDcplDictLearn dictionary learning algorithm
        options.

        Options include all of those defined in
        :class:`.dictlrn.DictLearn.Options`, together with
        additional options:

          ``AccurateDFid`` : Flag determining whether data fidelity term is
          estimated from the value computed in the X update (``False``) or is
          computed after every outer iteration over an X update and a D
          update (``True``), which is slower but more accurate.

          ``DictSize`` : Dictionary size vector.

          ``CBPDN`` : Options :class:`.admm.cbpdn.ConvBPDNMaskDcpl.Options`

          ``CCMOD`` : Options :func:`.ccmod.ConvCnstrMODMaskDcpl.Options`
        """

        defaults = copy.deepcopy(dictlrn.DictLearn.Options.defaults)
        defaults.update({'DictSize' : None, 'AccurateDFid' : False,
                        'CBPDN' : copy.deepcopy(
                            Acbpdn.ConvBPDNMaskDcpl.Options.defaults)})


        def __init__(self, opt=None):
            """Initialise ConvBPDNMaskDcplDictLearn dictionary
            learning algorithm options.
            """

            self.defaults.update({'CCMOD' : copy.deepcopy(
                ccmod.ConvCnstrMODMaskDcpl.Options.defaults)})

            dictlrn.DictLearn.Options.__init__(self, {
                'CBPDN': Acbpdn.ConvBPDNMaskDcpl.Options({'MaxMainIter': 1,
                'AutoRho': {'Period': 10, 'AutoScaling': False,
                'RsdlRatio': 10.0, 'Scaling': 2.0, 'RsdlTarget': 1.0}}),
                'CCMOD': ccmod.ConvCnstrMODMaskDcpl.Options(
                    {'MaxMainIter': 1,
                'BackTrack': {'Eta': 1.2, 'MaxIter': 50}})
            })

            if opt is None:
                opt = {}
            self.update(opt)



    def __init__(self, D0, S, lmbda, W, opt=None, dimK=1, dimN=2):
        """
        Initialise a MixConvBPDNMaskDcplDictLearn object with problem
        size and options.


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
        opt : :class:`MixConvBPDNMaskDcplDictLearn.Options` object
          Algorithm options
        dimK : int, optional (default 1)
          Number of signal dimensions. If there is only a single input
          signal (e.g. if `S` is a 2D array representing a single image)
          `dimK` must be set to 0.
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        if opt is None:
            opt = MixConvBPDNMaskDcplDictLearn.Options()
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

        # Modify D update options to include initial values for X
        X0 = cr.zpad(cr.stdformD(D0, cri.Cd, cri.M, dimN), cri.Nv)
        opt['CCMOD'].update({'X0' : X0})


        # Create X update object
        xstep = Acbpdn.ConvBPDNMaskDcpl(D0, S, lmbda, W, opt['CBPDN'],
                                        dimK=dimK, dimN=dimN)

        # Create D update object
        dstep = ccmod.ConvCnstrMODMaskDcpl(None, S, W, dsz, opt['CCMOD'],
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

        if dstep.opt['BackTrack', 'Enabled']:
            isfld = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr', 'XPrRsdl',
                     'XDlRsdl', 'XRho', 'D_F_Btrack', 'D_Q_Btrack', 'D_ItBt',
                     'D_L', 'Time']
            isdmap = {'Cnstr':  'Cnstr', 'D_F_Btrack': 'F_Btrack',
                      'D_Q_Btrack': 'Q_Btrack', 'D_ItBt': 'IterBTrack',
                      'D_L': 'L'}
            hdrtxt = ['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr', 'r_X', 's_X',
                      u('ρ_X'), 'F_D', 'Q_D', 'It_D', 'L_D']
            hdrmap = {'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
                      u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr', 'r_X': 'XPrRsdl',
                      's_X': 'XDlRsdl', u('ρ_X'): 'XRho', 'F_D': 'D_F_Btrack',
                      'Q_D': 'D_Q_Btrack', 'It_D': 'D_ItBt', 'L_D': 'D_L'}

        else:
            isfld = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr', 'XPrRsdl',
                     'XDlRsdl', 'XRho', 'D_L', 'Time']
            isdmap = {'Cnstr':  'Cnstr', 'D_L': 'L'}
            hdrtxt = ['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr', 'r_X', 's_X',
                      u('ρ_X'), 'L_D']
            hdrmap = {'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
                      u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr', 'r_X': 'XPrRsdl',
                      's_X': 'XDlRsdl', u('ρ_X'): 'XRho', 'L_D': 'D_L'}

        isc = dictlrn.IterStatsConfig(isfld=isfld, isxmap=isxmap,
                                      isdmap=isdmap, evlmap=evlmap,
                                      hdrtxt=hdrtxt, hdrmap=hdrmap)


        # Call parent constructor
        super(MixConvBPDNMaskDcplDictLearn, self).__init__(xstep, dstep,
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
            X = self.getcoef()
            rl1 = np.sum(np.abs(X))
            return dict(DFid=dfd, RegL1=rl1, ObjFun=dfd+self.xstep.lmbda*rl1)

        else:
            return None
