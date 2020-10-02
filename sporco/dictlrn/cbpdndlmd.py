# -*- coding: utf-8 -*-
# Copyright (C) 2015-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Dictionary learning based on CBPDN sparse coding with a spatial mask in
the data fidelity term
"""

from __future__ import print_function, absolute_import

import copy
import numpy as np

import sporco.cnvrep as cr
import sporco.admm.cbpdn as admm_cbpdn
import sporco.admm.ccmodmd as admm_ccmod
import sporco.pgm.cbpdn as pgm_cbpdn
import sporco.pgm.ccmod as pgm_ccmod
from sporco.dictlrn import dictlrn
import sporco.dictlrn.common as dc
from sporco.common import _fix_dynamic_class_lookup
from sporco.fft import rfftn, irfftn
from sporco.linalg import inner


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def cbpdnmsk_class_label_lookup(label):
    """Get a ConvBPDNMask class from a label string."""

    clsmod = {'admm': admm_cbpdn.ConvBPDNMaskDcpl,
              'pgm': pgm_cbpdn.ConvBPDNMask}
    if label in clsmod:
        return clsmod[label]
    else:
        raise ValueError('Unknown ConvBPDNMask solver method %s' % label)



def ConvBPDNMaskOptionsDefaults(method='admm'):
    """Get defaults dict for the ConvBPDNMask class specified by the
    ``method`` parameter.
    """

    dflt = copy.deepcopy(cbpdnmsk_class_label_lookup(method).Options.defaults)
    if method == 'admm':
        dflt.update({'MaxMainIter': 1, 'AutoRho':
                     {'Period': 10, 'AutoScaling': False,
                      'RsdlRatio': 10.0, 'Scaling': 2.0,
                      'RsdlTarget': 1.0}})
    else:
        dflt.update({'MaxMainIter': 1})
    return dflt



def ConvBPDNMaskOptions(opt=None, method='admm'):
    """A wrapper function that dynamically defines a class derived from
    the Options class associated with one of the implementations of
    the Convolutional BPDN problem, and returns an object
    instantiated with the provided parameters. The wrapper is designed
    to allow the appropriate object to be created by calling this
    function using the same syntax as would be used if it were a
    class. The specific implementation is selected by use of an
    additional keyword argument 'method'. Valid values are as
    specified in the documentation for :func:`ConvBPDN`.
    """

    # Assign base class depending on method selection argument
    base = cbpdnmsk_class_label_lookup(method).Options

    # Nested class with dynamically determined inheritance
    class ConvBPDNMaskOptions(base):
        def __init__(self, opt):
            super(ConvBPDNMaskOptions, self).__init__(opt)

    # Allow pickling of objects of type ConvBPDNOptions
    _fix_dynamic_class_lookup(ConvBPDNMaskOptions, method)

    # Return object of the nested class type
    return ConvBPDNMaskOptions(opt)



def ConvBPDNMask(*args, **kwargs):
    """A wrapper function that dynamically defines a class derived from
    one of the implementations of the Convolutional Constrained MOD
    problems, and returns an object instantiated with the provided
    parameters. The wrapper is designed to allow the appropriate
    object to be created by calling this function using the same
    syntax as would be used if it were a class. The specific
    implementation is selected by use of an additional keyword
    argument 'method'. Valid values are:

    - ``'admm'`` :
      Use the implementation defined in :class:`.admm.cbpdn.ConvBPDNMaskDcpl`.
    - ``'pgm'`` :
      Use the implementation defined in :class:`.pgm.cbpdn.ConvBPDNMask`.

    The default value is ``'admm'``.
    """

    # Extract method selection argument or set default
    method = kwargs.pop('method', 'admm')

    # Assign base class depending on method selection argument
    base = cbpdnmsk_class_label_lookup(method)

    # Nested class with dynamically determined inheritance
    class ConvBPDNMask(base):
        def __init__(self, *args, **kwargs):
            super(ConvBPDNMask, self).__init__(*args, **kwargs)

    # Allow pickling of objects of type ConvBPDNMask
    _fix_dynamic_class_lookup(ConvBPDNMask, method)

    # Return object of the nested class type
    return ConvBPDNMask(*args, **kwargs)



def ccmodmsk_class_label_lookup(label):
    """Get a ConvCnstrMODMask class from a label string."""

    clsmod = {'ism': admm_ccmod.ConvCnstrMODMaskDcpl_IterSM,
              'cg': admm_ccmod.ConvCnstrMODMaskDcpl_CG,
              'cns': admm_ccmod.ConvCnstrMODMaskDcpl_Consensus,
              'pgm': pgm_ccmod.ConvCnstrMODMask}
    if label in clsmod:
        return clsmod[label]
    else:
        raise ValueError('Unknown ConvCnstrMODMask solver method %s' % label)



def ConvCnstrMODMaskOptionsDefaults(method='pgm'):
    """Get defaults dict for the ConvCnstrMODMask class specified by the
    ``method`` parameter.
    """

    dflt = copy.deepcopy(ccmodmsk_class_label_lookup(method).Options.defaults)
    if method == 'pgm':
        dflt.update({'MaxMainIter': 1})
    else:
        dflt.update({'MaxMainIter': 1, 'AutoRho':
                     {'Period': 10, 'AutoScaling': False,
                      'RsdlRatio': 10.0, 'Scaling': 2.0,
                      'RsdlTarget': 1.0}})
    return dflt



def ConvCnstrMODMaskOptions(opt=None, method='pgm'):
    """A wrapper function that dynamically defines a class derived from
    the Options class associated with one of the implementations of
    the Convolutional Constrained MOD problem, and returns an object
    instantiated with the provided parameters. The wrapper is designed
    to allow the appropriate object to be created by calling this
    function using the same syntax as would be used if it were a
    class. The specific implementation is selected by use of an
    additional keyword argument 'method'. Valid values are as
    specified in the documentation for :func:`ConvCnstrMODMask`.
    """

    # Assign base class depending on method selection argument
    base = ccmodmsk_class_label_lookup(method).Options

    # Nested class with dynamically determined inheritance
    class ConvCnstrMODMaskOptions(base):
        def __init__(self, opt):
            super(ConvCnstrMODMaskOptions, self).__init__(opt)

    # Allow pickling of objects of type ConvCnstrMODMaskOptions
    _fix_dynamic_class_lookup(ConvCnstrMODMaskOptions, method)

    # Return object of the nested class type
    return ConvCnstrMODMaskOptions(opt)



def ConvCnstrMODMask(*args, **kwargs):
    """A wrapper function that dynamically defines a class derived from
    one of the implementations of the Convolutional Constrained MOD
    problems, and returns an object instantiated with the provided
    parameters. The wrapper is designed to allow the appropriate
    object to be created by calling this function using the same
    syntax as would be used if it were a class. The specific
    implementation is selected by use of an additional keyword
    argument 'method'. Valid values are:

    - ``'ism'`` :
      Use the implementation defined in :class:`.ConvCnstrMODMaskDcpl_IterSM`.
      This method works well for a small number of training images, but is
      very slow for larger training sets.
    - ``'cg'`` :
      Use the implementation defined in :class:`.ConvCnstrMODMaskDcpl_CG`.
      This method is slower than ``'ism'`` for small training sets, but has
      better run time scaling as the training set grows.
    - ``'cns'`` :
      Use the implementation defined in
      :class:`.ConvCnstrMODMaskDcpl_Consensus`.
      This method is a good choice for large training sets.
    - ``'pgm'`` :
      Use the implementation defined in :class:`.pgm.ccmod.ConvCnstrMODMask`.
      This method is the best choice for large training sets.

    The default value is ``'pgm'``.
    """

    # Extract method selection argument or set default
    method = kwargs.pop('method', 'pgm')

    # Assign base class depending on method selection argument
    base = ccmodmsk_class_label_lookup(method)

    # Nested class with dynamically determined inheritance
    class ConvCnstrMODMask(base):
        def __init__(self, *args, **kwargs):
            super(ConvCnstrMODMask, self).__init__(*args, **kwargs)

    # Allow pickling of objects of type ConvCnstrMODMask
    _fix_dynamic_class_lookup(ConvCnstrMODMask, method)

    # Return object of the nested class type
    return ConvCnstrMODMask(*args, **kwargs)





class ConvBPDNMaskDictLearn(dictlrn.DictLearn):
    r"""
    Dictionary learning by alternating between sparse coding and dictionary
    update stages.

    |

    .. inheritance-diagram:: ConvBPDNMaskDictLearn
       :parts: 2

    |

    The sparse coding is performed using
    :class:`.admm.cbpdn.ConvBPDNMaskDcpl` (see :cite:`heide-2015-fast`) or
    :class:`.pgm.cbpdn.ConvBPDNMask` (see :cite:`chalasani-2013-fast` and
    :cite:`wohlberg-2016-efficient`), and the dictionary update is computed
    using :class:`.pgm.ccmod.ConvCnstrMODMask` (see
    :cite:`garcia-2018-convolutional1`) or one of the solver classes in
    :mod:`.admm.ccmodmd` (see :cite:`wohlberg-2016-efficient` and
    :cite:`garcia-2018-convolutional1`). The coupling between sparse coding
    and dictionary update stages is as in :cite:`garcia-2017-subproblem`.

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

       *If the ADMM solver is selected for sparse coding:*

           ``XPrRsdl`` : Norm of X primal residual

           ``XDlRsdl`` : Norm of X dual residual

           ``XRho`` : X penalty parameter

       *If the PGM solver is selected for sparse coding:*

           ``X_F_Btrack`` : Value of objective function for CSC problem

           ``X_Q_Btrack`` : Value of quadratic approximation for CSC problem

           ``X_ItBt`` : Number of iterations in backtracking for CSC problem

           ``X_L`` : Inverse of gradient step parameter for CSC problem

       *If an ADMM solver is selected for the dictionary update:*

           ``DPrRsdl`` : Norm of D primal residual

           ``DDlRsdl`` : Norm of D dual residual

           ``DRho`` : D penalty parameter

       *If the PGM solver is selected for the dictionary update:*

           ``D_F_Btrack`` : Value of objective function for CDU problem

           ``D_Q_Btrack`` : Value of wuadratic approximation for CDU problem

           ``D_ItBt`` : Number of iterations in backtracking for CDU problem

           ``D_L`` : Inverse of gradient step parameter for CDU problem

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

          ``CBPDN`` : An options class appropriate for the selected
          sparse coding solver class

          ``CCMOD`` : An options class appropriate for the selected
          dictionary update solver class
        """

        defaults = copy.deepcopy(dictlrn.DictLearn.Options.defaults)
        defaults.update({'DictSize': None, 'AccurateDFid': False})


        def __init__(self, opt=None, xmethod=None, dmethod=None):
            """
            Valid values for parameters ``xmethod`` and ``dmethod`` are
            documented in functions :func:`.ConvBPDNMask` and
            :func:`.ConvCnstrMODMask` respectively.
            """

            if xmethod is None:
                xmethod = 'admm'
            if dmethod is None:
                dmethod = 'pgm'

            self.xmethod = xmethod
            self.dmethod = dmethod

            self.defaults.update(
                {'CBPDN': ConvBPDNMaskOptionsDefaults(xmethod),
                 'CCMOD': ConvCnstrMODMaskOptionsDefaults(dmethod)})

            # Initialisation of CBPDN and CCMOD keys here is required to
            # ensure that the corresponding options have types appropriate
            # for classes in the cbpdn and ccmod modules, and are not just
            # standard entries in the parent option tree
            dictlrn.DictLearn.Options.__init__(self, {
                'CBPDN': ConvBPDNMaskOptions(self.defaults['CBPDN'],
                                             method=xmethod),
                'CCMOD': ConvCnstrMODMaskOptions(self.defaults['CCMOD'],
                                                 method=dmethod)})

            if opt is None:
                opt = {}
            self.update(opt)



    def __init__(self, D0, S, lmbda, W, opt=None, xmethod=None,
                 dmethod=None, dimK=1, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdnmddl_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdnmddl_init.svg

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
          data layouts) after reshaping to the shape determined by
          :func:`.cnvrep.mskWshape`.
        opt : :class:`ConvBPDNMaskDictLearn.Options` object
          Algorithm options
        xmethod : string, optional (default 'admm')
          String selecting sparse coding solver. Valid values are
          documented in function :func:`.ConvBPDNMask`.
        dmethod : string, optional (default 'pgm')
          String selecting dictionary update solver. Valid values are
          documented in function :func:`.ConvCnstrMODMask`.
        dimK : int, optional (default 1)
          Number of signal dimensions. If there is only a single input
          signal (e.g. if `S` is a 2D array representing a single image)
          `dimK` must be set to 0.
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        if opt is None:
            opt = ConvBPDNMaskDictLearn.Options(xmethod=xmethod,
                                                dmethod=dmethod)
        if xmethod is None:
            xmethod = opt.xmethod
        if dmethod is None:
            dmethod = opt.dmethod
        if opt.xmethod != xmethod or opt.dmethod != dmethod:
            raise ValueError('Parameters xmethod and dmethod must have the '
                             'same values used to initialise the Options '
                             'object')
        self.opt = opt
        self.xmethod = xmethod
        self.dmethod = dmethod

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

        # Modify D update options to include initial values for Y
        if cri.C == cri.Cd:
            Y0b0 = np.zeros(cri.Nv + (cri.C, 1, cri.K))
        else:
            Y0b0 = np.zeros(cri.Nv + (1, 1, cri.C * cri.K))
        Y0b1 = cr.zpad(cr.stdformD(D0, cri.Cd, cri.M, dimN), cri.Nv)
        if dmethod == 'pgm':
            opt['CCMOD'].update({'X0': Y0b1})
        else:
            if dmethod == 'cns':
                Y0 = Y0b1
            else:
                Y0 = np.concatenate((Y0b0, Y0b1), axis=cri.axisM)
            opt['CCMOD'].update({'Y0': Y0})

        # Create X update object
        xstep = ConvBPDNMask(D0, S, lmbda, W, opt['CBPDN'], method=xmethod,
                             dimK=dimK, dimN=dimN)

        # Create D update object
        dstep = ConvCnstrMODMask(None, S, W, dsz, opt['CCMOD'],
                                 method=dmethod, dimK=dimK, dimN=dimN)

        # Configure iteration statistics reporting
        isc = dictlrn.IterStatsConfig(
            isfld=dc.isfld(xmethod, dmethod, opt),
            isxmap=dc.isxmap(xmethod, opt), isdmap=dc.isdmap(dmethod),
            evlmap=dc.evlmap(opt['AccurateDFid']),
            hdrtxt=dc.hdrtxt(xmethod, dmethod, opt),
            hdrmap=dc.hdrmap(xmethod, dmethod, opt),
            fmtmap={'It_X': '%4d', 'It_D': '%4d'})

        # Call parent constructor
        super(ConvBPDNMaskDictLearn, self).__init__(xstep, dstep, opt, isc)



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
        Df = rfftn(D, self.xstep.cri.Nv, self.xstep.cri.axisN)
        Xf = rfftn(X, self.xstep.cri.Nv, self.xstep.cri.axisN)
        DXf = inner(Df, Xf, axis=self.xstep.cri.axisM)
        return irfftn(DXf, self.xstep.cri.Nv, self.xstep.cri.axisN)



    def evaluate(self):
        """Evaluate functional value of previous iteration."""

        if self.opt['AccurateDFid']:
            DX = self.reconstruct()
            S = self.xstep.S
            dfd = (np.linalg.norm(self.xstep.W * (DX - S))**2) / 2.0
            if self.xmethod == 'pgm':
                X = self.xstep.getcoef()
            else:
                X = self.xstep.var_y1()
            rl1 = np.sum(np.abs(X))
            return dict(DFid=dfd, RegL1=rl1,
                        ObjFun=dfd + self.xstep.lmbda * rl1)
        else:
            return None
