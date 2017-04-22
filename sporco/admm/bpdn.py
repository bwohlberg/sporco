# -*- coding: utf-8 -*-
# Copyright (C) 2015-2017 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithm for the BPDN problem"""

from __future__ import division
from __future__ import absolute_import

import copy
import numpy as np
from scipy import linalg

from sporco.admm import admm
import sporco.linalg as sl
from sporco.util import u


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class GenericBPDN(admm.ADMMEqual):
    r"""
    **Class inheritance structure**

    .. inheritance-diagram:: GenericBPDN
       :parts: 2

    |


    Base class for ADMM algorithm for solving variants of the
    Basis Pursuit DeNoising (BPDN) :cite:`chen-1998-atomic` problem.

    The generic problem form is

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + f(\mathbf{x}) \;\;,

    where :math:`f(\cdot)` is a penalty term or the indicator function of
    a constraint, and is solved via the ADMM problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + f(\mathbf{y})
       \quad \text{such that} \quad \mathbf{x} = \mathbf{y} \;\;.

    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term
       :math:`(1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2`

       ``Reg`` : Value of regularisation term

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """


    class Options(admm.ADMMEqual.Options):
        """GenericBPDN algorithm options

        Options include all of those defined in
        :class:`sporco.admm.admm.ADMMEqual.Options`, together with
        additional options:

          ``AuxVarObj`` : Flag indicating whether the objective
          function should be evaluated using variable X (``False``) or
          Y (``True``) as its argument. Setting this flag to ``True``
          often gives a better estimate of the objective function.

          ``NonNegCoef`` : If ``True``, force solution to be non-negative.
        """

        defaults = copy.deepcopy(admm.ADMMEqual.Options.defaults)
        defaults.update({'AuxVarObj' : True, 'ReturnX' : False,
                        'RelaxParam' : 1.8, 'NonNegCoef' : False})
        defaults['AutoRho'].update({'Enabled' : True, 'Period' : 10,
                                    'AutoScaling' : True, 'Scaling' : 1000.0,
                                    'RsdlRatio' : 1.2})

        def __init__(self, opt=None):
            """Initialise GenericBPDN algorithm options object."""

            if opt is None:
                opt = {}
            admm.ADMMEqual.Options.__init__(self, opt)



        def __setitem__(self, key, value):
            """Set options 'fEvalX' and 'gEvalY' appropriately when option
            'AuxVarObj' is set.
            """

            admm.ADMMEqual.Options.__setitem__(self, key, value)

            if key == 'AuxVarObj':
                if value is True:
                    self['fEvalX'] = False
                    self['gEvalY'] = True
                else:
                    self['fEvalX'] = True
                    self['gEvalY'] = False



    itstat_fields_objfn = ('ObjFun', 'DFid', 'Reg')
    hdrtxt_objfn = ('Fnc', 'DFid', 'Reg')
    hdrval_objfun = {'Fnc' : 'ObjFun', 'DFid' : 'DFid', 'Reg' : 'Reg'}



    def __init__(self, D, S, opt=None):
        """
        Initialise a GenericBPDN object with problem parameters.

        Parameters
        ----------
        D : array_like, shape (N, M)
          Dictionary matrix
        S : array_like, shape (N, K)
          Signal vector or matrix
        opt : :class:`BPDN.Options` object
          Algorithm options
        """

        Nc = D.shape[1]
        Nm = S.shape[1]
        if opt is None:
            opt = GenericBPDN.Options()
        super(GenericBPDN, self).__init__((Nc, Nm), S.dtype, opt)

        self.S = np.asarray(S, dtype=self.dtype)
        self.setdict(D)



    def setdict(self, D):
        """Set dictionary array."""

        self.D = np.asarray(D, dtype=self.dtype)
        self.DTS = self.D.T.dot(self.S)
        # Factorise dictionary for efficient solves
        self.lu, self.piv = sl.lu_factor(self.D, self.rho)
        self.lu = np.asarray(self.lu, dtype=self.dtype)



    def getcoef(self):
        """Get final coefficient array."""

        return self.Y



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{x}`."""

        self.X = np.asarray(sl.lu_solve_ATAI(self.D, self.rho, self.DTS +
                        self.rho*(self.Y - self.U), self.lu, self.piv),
                        dtype=self.dtype)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{y}`.
        If this method is not overridden, the problem is solved without
        any regularisation other than the option enforcement of
        non-negativity of the solution. When it is overridden, it
        should be explicitly called at the end of the overriding
        method.
        """

        if self.opt['NonNegCoef']:
            self.Y[self.Y < 0.0] = 0.0



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_dfd()
        reg = self.obfn_reg()
        obj = dfd + reg[0]
        return (obj, dfd) + reg[1:]



    def obfn_dfd(self):
        r"""Compute data fidelity term :math:`(1/2) \| D \mathbf{x} -
        \mathbf{s} \|_2^2`.
        """

        return 0.5*linalg.norm((self.D.dot(self.obfn_fvar()) - self.S))**2



    def obfn_reg(self):
        """Compute regularisation term(s) and contribution to objective
        function.
        """

        raise NotImplementedError()



    def rhochange(self):
        """Re-factorise matrix when rho changes."""

        self.lu, self.piv = sl.lu_factor(self.D, self.rho)
        self.lu = np.asarray(self.lu, dtype=self.dtype)





class BPDN(GenericBPDN):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: BPDN
       :parts: 2

    |

    ADMM algorithm for the Basis Pursuit DeNoising (BPDN)
    :cite:`chen-1998-atomic` problem.

    Solve the Single Measurement Vector (SMV) BPDN problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1

    via the ADMM problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{y} \|_1
       \quad \text{such that} \quad \mathbf{x} = \mathbf{y} \;\;.


    The Multiple Measurement Vector (MMV) BPDN problem

    .. math::
       \mathrm{argmin}_X \;
       (1/2) \| D X - S \|_F^2 + \lambda \| X \|_1

    is also supported.


    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| D
       \mathbf{x} - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\| \mathbf{x}
       \|_1`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """


    class Options(GenericBPDN.Options):
        r"""BPDN algorithm options

        Options include all of those defined in
        :class:`.GenericBPDN.Options`, together with additional
        options:

          ``L1Weight`` : An array of weights for the :math:`\ell_1`
          norm. The array shape must be such that the array is
          compatible for multiplication with the X/Y variables. If this
          option is defined, the regularization term is :math:`\lambda \|
          \mathbf{w} \odot \mathbf{x} \|_1` where :math:`\mathbf{w}`
          denotes the weighting array.
        """

        defaults = copy.deepcopy(GenericBPDN.Options.defaults)
        defaults.update({'L1Weight' : 1.0})


        def __init__(self, opt=None):
            """Initialise BPDN algorithm options object."""

            if opt is None:
                opt = {}
            GenericBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc' : 'ObjFun', 'DFid' : 'DFid', u('Regℓ1') : 'RegL1'}



    def __init__(self, D, S, lmbda=None, opt=None):
        """
        Initialise a BPDN object with problem parameters.

        Parameters
        ----------
        D : array_like, shape (N, M)
          Dictionary matrix
        S : array_like, shape (N, K)
          Signal vector or matrix
        lmbda : float
          Regularisation parameter
        opt : :class:`BPDN.Options` object
          Algorithm options
        """

        # Set default options if necessary
        if opt is None:
            opt = BPDN.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        # Set default lambda value if not specified
        if lmbda is None:
            DTS = D.T.dot(S)
            lmbda = 0.1*abs(DTS).max()

        # Set l1 term scaling and weight array
        self.lmbda = self.dtype.type(lmbda)
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(50.0*self.lmbda + 1.0),
                      dtype=self.dtype)

        # Set rho_xi attribute (see Sec. VI.C of wohlberg-2015-adaptive)
        if self.lmbda != 0.0:
            rho_xi = (1.0 + (18.3)**(np.log10(self.lmbda) + 1.0))
        else:
            rho_xi = 1.0
        self.set_attr('rho_xi', opt['AutoRho', 'RsdlTarget'], dval=rho_xi,
                      dtype=self.dtype)

        super(BPDN, self).__init__(D, S, opt)



    def uinit(self, ushape):
        """Return initialiser for working variable U"""

        if  self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            return (self.lmbda/self.rho)*np.sign(self.Y)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{y}`."""

        self.Y = np.asarray(sl.shrink1(self.AX + self.U,
                            (self.lmbda/self.rho)*self.wl1),
                            dtype=self.dtype)
        super(BPDN, self).ystep()



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        return (self.lmbda*rl1, rl1)




class BPDNJoint(BPDN):
    r"""
    **Class inheritance structure**

    .. inheritance-diagram:: BPDNJoint
       :parts: 2

    |


    ADMM algorithm for BPDN with joint sparsity via an :math:`\ell_{2,1}`
    norm term.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_X \; (1/2) \| D X - S \|_2^2 + \lambda \| X \|_1
       + \mu \| X \|_{2,1}

    via the ADMM problem

    .. math::
       \mathrm{argmin}_X \; (1/2) \| D X - S \|_2^2 +
       \lambda \| Y \|_1 + \mu \| Y \|_{2,1} \quad \text{such that} \quad
       X = Y \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term :math:`(1/2) \| D X - S \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\| X \|_1`

       ``RegL21`` : Value of regularisation term :math:`\| X \|_{2,1}`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual Residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegL21')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), u('Regℓ2,1'))
    hdrval_objfun = {'Fnc' : 'ObjFun', 'DFid' : 'DFid',
                     u('Regℓ1') : 'RegL1', u('Regℓ2,1') : 'RegL21'}



    def __init__(self, D, S, lmbda=None, mu=0.0, opt=None):
        """
        Initialise a BPDNJoint object with problem parameters.

        Parameters
        ----------
        D : array_like, shape (N, M)
          Dictionary matrix
        S : array_like, shape (M, K)
          Signal vector or matrix
        lmbda : float
          Regularisation parameter (l1)
        mu : float
          Regularisation parameter (l2,1)
        opt : :class:`BPDN.Options` object
          Algorithm options
        """

        if opt is None:
            opt = BPDN.Options()
        super(BPDNJoint, self).__init__(D, S, lmbda, opt)
        self.mu = self.dtype.type(mu)


    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to :math:`\mathbf{y}`."""

        self.Y = np.asarray(sl.shrink12(self.AX + self.U,
                            (self.lmbda/self.rho)*self.wl1, self.mu/self.rho),
                            dtype=self.dtype)
        GenericBPDN.ystep(self)



    def obfn_reg(self):
        r"""Compute regularisation terms and contribution to objective
        function. Regularisation terms are :math:`\| Y \|_1` and
        :math:`\| Y \|_{2,1}`.
        """

        rl1 = linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rl21 = np.sum(np.sqrt(np.sum(self.obfn_gvar()**2, axis=1)))
        return (self.lmbda*rl1 + self.mu*rl21, rl1, rl21)





class ElasticNet(BPDN):
    r"""**Class inheritance structure**

    .. inheritance-diagram:: ElasticNet
       :parts: 2

    |

    ADMM algorithm for the elastic net :cite:`zou-2005-regularization`
    problem.

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1
       + (\mu/2) \| \mathbf{x} \|_2^2

    via the ADMM problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{y} \|_1
       + (\mu/2) \| \mathbf{x} \|_2^2 \quad \text{such that} \quad
       \mathbf{x} = \mathbf{y} \;\;.

    After termination of the :meth:`solve` method, attribute :attr:`itstat` is
    a list of tuples representing statistics of each iteration. The
    fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` : Value of data fidelity term :math:`(1/2) \| D
       \mathbf{x} - \mathbf{s} \|_2^2`

       ``RegL1`` : Value of regularisation term :math:`\| \mathbf{x}
       \|_1`

       ``RegL2`` : Value of regularisation term :math:`(1/2) \|
       \mathbf{x} \|_2^2`

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual Residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1', 'RegL2')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), u('Regℓ2'))
    hdrval_objfun = {'Fnc' : 'ObjFun', 'DFid' : 'DFid',
                     u('Regℓ1') : 'RegL1', u('Regℓ2') : 'RegL2'}



    def __init__(self, D, S, lmbda=None, mu=0.0, opt=None):
        """
        Initialise an ElasticNet object with problem parameters.

        Parameters
        ----------
        D : array_like, shape (N, M)
          Dictionary matrix
        S : array_like, shape (M, K)
          Signal vector or matrix
        lmbda : float
          Regularisation parameter (l1)
        mu : float
          Regularisation parameter (l2)
        opt : :class:`BPDN.Options` object
          Algorithm options
        """

        if opt is None:
            opt = BPDN.Options()

        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)

        self.mu = self.dtype.type(mu)

        super(ElasticNet, self).__init__(D, S, lmbda, opt)



    def setdict(self, D):
        """Set dictionary array."""

        self.D = np.asarray(D)
        self.DTS = self.D.T.dot(self.S)
        # Factorise dictionary for efficient solves
        self.lu, self.piv = sl.lu_factor(self.D, self.mu + self.rho)
        self.lu = np.asarray(self.lu, dtype=self.dtype)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.X = np.asarray(sl.lu_solve_ATAI(self.D, self.mu + self.rho,
                    self.DTS + self.rho*(self.Y - self.U), self.lu, self.piv),
                    dtype=self.dtype)



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rl2 = 0.5*linalg.norm(self.obfn_gvar())**2
        return (self.lmbda*rl1 + self.mu*rl2, rl1, rl2)



    def rhochange(self):
        """Re-factorise matrix when rho changes."""

        self.lu, self.piv = sl.lu_factor(self.D, self.mu + self.rho)
        self.lu = np.asarray(self.lu, dtype=self.dtype)
