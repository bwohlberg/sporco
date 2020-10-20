# -*- coding: utf-8 -*-
# Copyright (C) 2015-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Dictionary learning based on CBPDN sparse coding"""

from __future__ import print_function, absolute_import

import copy
import numpy as np

import sporco.cnvrep as cr
import sporco.admm.cbpdn as admm_cbpdn
import sporco.admm.ccmod as admm_ccmod
import sporco.pgm.cbpdn as pgm_cbpdn
import sporco.pgm.ccmod as pgm_ccmod
from sporco.dictlrn import dictlrn
import sporco.dictlrn.common as dc
from sporco.common import _fix_dynamic_class_lookup
from sporco.linalg import inner
from sporco.fft import (rfftn, irfftn, rfl2norm2)


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def cbpdn_class_label_lookup(label):
    """Get a CBPDN class from a label string."""

    clsmod = {'admm': admm_cbpdn.ConvBPDN,
              'pgm': pgm_cbpdn.ConvBPDN}
    if label in clsmod:
        return clsmod[label]
    else:
        raise ValueError('Unknown ConvBPDN solver method %s' % label)



def ConvBPDNOptionsDefaults(method='admm'):
    """Get defaults dict for the ConvBPDN class specified by the ``method``
    parameter.
    """

    dflt = copy.deepcopy(cbpdn_class_label_lookup(method).Options.defaults)
    if method == 'admm':
        dflt.update({'MaxMainIter': 1, 'AutoRho':
                     {'Period': 10, 'AutoScaling': False,
                      'RsdlRatio': 10.0, 'Scaling': 2.0,
                      'RsdlTarget': 1.0}})
    else:
        dflt.update({'MaxMainIter': 1})
    return dflt



def ConvBPDNOptions(opt=None, method='admm'):
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
    base = cbpdn_class_label_lookup(method).Options

    # Nested class with dynamically determined inheritance
    class ConvBPDNOptions(base):
        def __init__(self, opt):
            super(ConvBPDNOptions, self).__init__(opt)

    # Allow pickling of objects of type ConvBPDNOptions
    _fix_dynamic_class_lookup(ConvBPDNOptions, method)

    # Return object of the nested class type
    return ConvBPDNOptions(opt)



def ConvBPDN(*args, **kwargs):
    """A wrapper function that dynamically defines a class derived from
    one of the implementations of the Convolutional Constrained MOD
    problems, and returns an object instantiated with the provided
    parameters. The wrapper is designed to allow the appropriate
    object to be created by calling this function using the same
    syntax as would be used if it were a class. The specific
    implementation is selected by use of an additional keyword
    argument 'method'. Valid values are:

    - ``'admm'`` :
      Use the implementation defined in :class:`.admm.cbpdn.ConvBPDN`.
    - ``'pgm'`` :
      Use the implementation defined in :class:`.pgm.cbpdn.ConvBPDN`.

    The default value is ``'admm'``.
    """

    # Extract method selection argument or set default
    method = kwargs.pop('method', 'admm')

    # Assign base class depending on method selection argument
    base = cbpdn_class_label_lookup(method)

    # Nested class with dynamically determined inheritance
    class ConvBPDN(base):
        def __init__(self, *args, **kwargs):
            super(ConvBPDN, self).__init__(*args, **kwargs)

    # Allow pickling of objects of type ConvBPDN
    _fix_dynamic_class_lookup(ConvBPDN, method)

    # Return object of the nested class type
    return ConvBPDN(*args, **kwargs)



def ccmod_class_label_lookup(label):
    """Get a CCMOD class from a label string."""

    clsmod = {'ism': admm_ccmod.ConvCnstrMOD_IterSM,
              'cg': admm_ccmod.ConvCnstrMOD_CG,
              'cns': admm_ccmod.ConvCnstrMOD_Consensus,
              'pgm': pgm_ccmod.ConvCnstrMOD}
    if label in clsmod:
        return clsmod[label]
    else:
        raise ValueError('Unknown ConvCnstrMOD solver method %s' % label)



def ConvCnstrMODOptionsDefaults(method='pgm'):
    """Get defaults dict for the ConvCnstrMOD class specified by the
    ``method`` parameter.
    """

    dflt = copy.deepcopy(ccmod_class_label_lookup(method).Options.defaults)
    if method == 'pgm':
        dflt.update({'MaxMainIter': 1})
    else:
        dflt.update({'MaxMainIter': 1, 'AutoRho':
                     {'Period': 10, 'AutoScaling': False,
                      'RsdlRatio': 10.0, 'Scaling': 2.0,
                      'RsdlTarget': 1.0}})
    return dflt



def ConvCnstrMODOptions(opt=None, method='pgm'):
    """A wrapper function that dynamically defines a class derived from
    the Options class associated with one of the implementations of
    the Convolutional Constrained MOD problem, and returns an object
    instantiated with the provided parameters. The wrapper is designed
    to allow the appropriate object to be created by calling this
    function using the same syntax as would be used if it were a
    class. The specific implementation is selected by use of an
    additional keyword argument 'method'. Valid values are as
    specified in the documentation for :func:`ConvCnstrMOD`.
    """

    # Assign base class depending on method selection argument
    base = ccmod_class_label_lookup(method).Options

    # Nested class with dynamically determined inheritance
    class ConvCnstrMODOptions(base):
        def __init__(self, opt):
            super(ConvCnstrMODOptions, self).__init__(opt)

    # Allow pickling of objects of type ConvCnstrMODOptions
    _fix_dynamic_class_lookup(ConvCnstrMODOptions, method)

    # Return object of the nested class type
    return ConvCnstrMODOptions(opt)



def ConvCnstrMOD(*args, **kwargs):
    """A wrapper function that dynamically defines a class derived from
    one of the implementations of the Convolutional Constrained MOD
    problems, and returns an object instantiated with the provided
    parameters. The wrapper is designed to allow the appropriate
    object to be created by calling this function using the same
    syntax as would be used if it were a class. The specific
    implementation is selected by use of an additional keyword
    argument 'method'. Valid values are:

    - ``'ism'`` :
      Use the implementation defined in :class:`.ConvCnstrMOD_IterSM`. This
      method works well for a small number of training images, but is very
      slow for larger training sets.
    - ``'cg'`` :
      Use the implementation defined in :class:`.ConvCnstrMOD_CG`. This
      method is slower than ``'ism'`` for small training sets, but has better
      run time scaling as the training set grows.
    - ``'cns'`` :
      Use the implementation defined in :class:`.ConvCnstrMOD_Consensus`.
      This method is a good choice for large training sets.
    - ``'pgm'`` :
      Use the implementation defined in :class:`.pgm.ccmod.ConvCnstrMOD`.
      This method is the best choice for large training sets.

    The default value is ``'pgm'``.
    """

    # Extract method selection argument or set default
    method = kwargs.pop('method', 'pgm')

    # Assign base class depending on method selection argument
    base = ccmod_class_label_lookup(method)

    # Nested class with dynamically determined inheritance
    class ConvCnstrMOD(base):
        def __init__(self, *args, **kwargs):
            super(ConvCnstrMOD, self).__init__(*args, **kwargs)

    # Allow pickling of objects of type ConvCnstrMOD
    _fix_dynamic_class_lookup(ConvCnstrMOD, method)

    # Return object of the nested class type
    return ConvCnstrMOD(*args, **kwargs)



class ConvBPDNDictLearn(dictlrn.DictLearn):
    r"""
    Dictionary learning by alternating between sparse coding and dictionary
    update stages.

    |

    .. inheritance-diagram:: ConvBPDNDictLearn
       :parts: 2

    |

    The sparse coding is performed using
    :class:`.admm.cbpdn.ConvBPDN` (see :cite:`wohlberg-2014-efficient`) or
    :class:`.pgm.cbpdn.ConvBPDN` (see :cite:`chalasani-2013-fast` and
    :cite:`wohlberg-2016-efficient`), and the dictionary update is computed
    using :class:`.pgm.ccmod.ConvCnstrMOD` (see
    :cite:`garcia-2018-convolutional1`) or one of the solver classes in
    :mod:`.admm.ccmod` (see :cite:`wohlberg-2016-efficient` and
    :cite:`sorel-2016-fast`). The coupling between sparse coding and
    dictionary update stages is as in :cite:`garcia-2017-subproblem`.


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
    :func:`.ConvCnstrMOD` problems. Multi-channel variants
    :cite:`wohlberg-2016-convolutional` are also supported.

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
            documented in functions :func:`.ConvBPDN` and
            :func:`.ConvCnstrMOD` respectively.
            """

            if xmethod is None:
                xmethod = 'admm'
            if dmethod is None:
                dmethod = 'pgm'

            self.xmethod = xmethod
            self.dmethod = dmethod

            self.defaults.update(
                {'CBPDN': ConvBPDNOptionsDefaults(xmethod),
                 'CCMOD': ConvCnstrMODOptionsDefaults(dmethod)})

            # Initialisation of CBPDN and CCMOD keys here is required to
            # ensure that the corresponding options have types appropriate
            # for classes in the cbpdn and ccmod modules, and are not just
            # standard entries in the parent option tree
            dictlrn.DictLearn.Options.__init__(self, {
                'CBPDN': ConvBPDNOptions(self.defaults['CBPDN'],
                                         method=xmethod),
                'CCMOD': ConvCnstrMODOptions(self.defaults['CCMOD'],
                                             method=dmethod)})

            if opt is None:
                opt = {}
            self.update(opt)



    def __init__(self, D0, S, lmbda=None, opt=None, xmethod=None,
                 dmethod=None, dimK=1, dimN=2):
        """

        |

        **Call graph**

        .. image:: ../_static/jonga/cbpdndl_init.svg
           :width: 20%
           :target: ../_static/jonga/cbpdndl_init.svg

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
        xmethod : string, optional (default 'admm')
          String selecting sparse coding solver. Valid values are
          documented in function :func:`.ConvBPDN`.
        dmethod : string, optional (default 'pgm')
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
            opt = ConvBPDNDictLearn.Options(xmethod=xmethod, dmethod=dmethod)
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

        # Modify D update options to include initial value for Y
        optname = 'X0' if dmethod == 'pgm' else 'Y0'
        opt['CCMOD'].update({optname: cr.zpad(
            cr.stdformD(D0, cri.Cd, cri.M, dimN), cri.Nv)})

        # Create X update object
        xstep = ConvBPDN(D0, S, lmbda, opt['CBPDN'], method=xmethod,
                         dimK=dimK, dimN=dimN)

        # Create D update object
        dstep = ConvCnstrMOD(None, S, dsz, opt['CCMOD'], method=dmethod,
                             dimK=dimK, dimN=dimN)

        # Configure iteration statistics reporting
        isc = dictlrn.IterStatsConfig(
            isfld=dc.isfld(xmethod, dmethod, opt),
            isxmap=dc.isxmap(xmethod, opt), isdmap=dc.isdmap(dmethod),
            evlmap=dc.evlmap(opt['AccurateDFid']),
            hdrtxt=dc.hdrtxt(xmethod, dmethod, opt),
            hdrmap=dc.hdrmap(xmethod, dmethod, opt),
            fmtmap={'It_X': '%4d', 'It_D': '%4d'})

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
        Df = rfftn(D, self.xstep.cri.Nv, self.xstep.cri.axisN)
        Xf = rfftn(X, self.xstep.cri.Nv, self.xstep.cri.axisN)
        DXf = inner(Df, Xf, axis=self.xstep.cri.axisM)
        return irfftn(DXf, self.xstep.cri.Nv, self.xstep.cri.axisN)



    def evaluate(self):
        """Evaluate functional value of previous iteration."""

        if self.opt['AccurateDFid']:
            if self.dmethod == 'pgm':
                D = self.dstep.getdict(crop=False)
            else:
                D = self.dstep.var_y()
            if self.xmethod == 'pgm':
                X = self.xstep.getcoef()
            else:
                X = self.xstep.var_y()
            Df = rfftn(D, self.xstep.cri.Nv, self.xstep.cri.axisN)
            Xf = rfftn(X, self.xstep.cri.Nv, self.xstep.cri.axisN)
            Sf = self.xstep.Sf
            Ef = inner(Df, Xf, axis=self.xstep.cri.axisM) - Sf
            dfd = rfl2norm2(Ef, self.xstep.S.shape,
                            axis=self.xstep.cri.axisN) / 2.0
            rl1 = np.sum(np.abs(X))
            return dict(DFid=dfd, RegL1=rl1,
                        ObjFun=dfd + self.xstep.lmbda * rl1)
        else:
            return None
