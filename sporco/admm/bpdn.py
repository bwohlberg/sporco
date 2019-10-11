# -*- coding: utf-8 -*-
# Copyright (C) 2015-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM algorithm for the BPDN problem"""

from __future__ import division, absolute_import

import copy
import numpy as np

from sporco.admm import admm
import sporco.linalg as sl
import sporco.prox as sp
from sporco.util import u


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class GenericBPDN(admm.ADMMEqual):
    r"""
    Base class for ADMM algorithm for solving variants of the
    Basis Pursuit DeNoising (BPDN) :cite:`chen-1998-atomic` problem.

    |

    .. inheritance-diagram:: GenericBPDN
       :parts: 2

    |

    The generic problem form is

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + g(\mathbf{x}) \;\;,

    where :math:`g(\cdot)` is a penalty term or the indicator function
    of a constraint, and is solved via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + g(\mathbf{y})
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
        :class:`.admm.ADMMEqual.Options`, together with
        additional options:

          ``AuxVarObj`` : Flag indicating whether the objective
          function should be evaluated using variable X (``False``) or
          Y (``True``) as its argument. Setting this flag to ``True``
          often gives a better estimate of the objective function.

          ``LinSolveCheck`` : Flag indicating whether to compute
          relative residual of X step solver.

          ``NonNegCoef`` : If ``True``, force solution to be non-negative.
        """

        defaults = copy.deepcopy(admm.ADMMEqual.Options.defaults)
        # Warning: although __setitem__ below takes care of setting
        # 'fEvalX' and 'gEvalY' from the value of 'AuxVarObj', this
        # cannot be relied upon for initialisation since the order of
        # initialisation of the dictionary keys is not deterministic;
        # if 'AuxVarObj' is initialised first, the other two keys are
        # correctly set, but this setting is overwritten when 'fEvalX'
        # and 'gEvalY' are themselves initialised
        defaults.update({'AuxVarObj': True, 'fEvalX': False,
                         'gEvalY': True, 'ReturnX': False,
                         'LinSolveCheck': False, 'RelaxParam': 1.8,
                         'NonNegCoef': False})
        defaults['AutoRho'].update({'Enabled': True, 'Period': 10,
                                    'AutoScaling': True, 'Scaling': 1000.0,
                                    'RsdlRatio': 1.2})

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              GenericBPDN algorithm options
            """

            if opt is None:
                opt = {}
            admm.ADMMEqual.Options.__init__(self, opt)



        def __setitem__(self, key, value):
            """Set options 'fEvalX' and 'gEvalY' appropriately when
            option 'AuxVarObj' is set.
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
    itstat_fields_extra = ('XSlvRelRes',)
    hdrtxt_objfn = ('Fnc', 'DFid', 'Reg')
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', 'Reg': 'Reg'}



    def __init__(self, D, S, opt=None):
        """
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
        self.lu, self.piv = sl.cho_factor(self.D, self.rho)
        self.lu = np.asarray(self.lu, dtype=self.dtype)



    def getcoef(self):
        """Get final coefficient array."""

        return self.Y



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.X = np.asarray(sl.cho_solve_ATAI(
            self.D, self.rho, self.DTS + self.rho * (self.Y - self.U),
            self.lu, self.piv), dtype=self.dtype)

        if self.opt['LinSolveCheck']:
            b = self.DTS + self.rho * (self.Y - self.U)
            ax = self.D.T.dot(self.D.dot(self.X)) + self.rho*self.X
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.  If this method is not overridden, the
        problem is solved without any regularisation other than the
        option enforcement of non-negativity of the solution.  When it
        is overridden, it should be explicitly called at the end of
        the overriding method.
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

        return 0.5*np.linalg.norm((self.D.dot(self.obfn_fvar()) - self.S))**2



    def obfn_reg(self):
        """Compute regularisation term(s) and contribution to objective
        function.
        """

        raise NotImplementedError()



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return (self.xrrs,)



    def rhochange(self):
        """Re-factorise matrix when rho changes."""

        self.lu, self.piv = sl.cho_factor(self.D, self.rho)
        self.lu = np.asarray(self.lu, dtype=self.dtype)





class BPDN(GenericBPDN):
    r"""
    ADMM algorithm for the Basis Pursuit DeNoising (BPDN)
    :cite:`chen-1998-atomic` problem.

    |

    .. inheritance-diagram:: BPDN
       :parts: 2

    |


    Solve the Single Measurement Vector (SMV) BPDN problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x}
       \|_1

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{y}
       \|_1 \quad \text{such that} \quad \mathbf{x} = \mathbf{y} \;\;.


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
          option is defined, the regularization term is :math:`\lambda
          \| \mathbf{w} \odot \mathbf{x} \|_1` where :math:`\mathbf{w}`
          denotes the weighting array.
        """

        defaults = copy.deepcopy(GenericBPDN.Options.defaults)
        defaults.update({'L1Weight': 1.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              BPDN algorithm options
            """

            if opt is None:
                opt = {}
            GenericBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1'}



    def __init__(self, D, S, lmbda=None, opt=None):
        """
        |

        **Call graph**

        .. image:: ../_static/jonga/bpdn_init.svg
           :width: 20%
           :target: ../_static/jonga/bpdn_init.svg

        |


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
            lmbda = 0.1 * abs(DTS).max()

        # Set l1 term scaling and weight array
        self.lmbda = self.dtype.type(lmbda)
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)

        # Set penalty parameter
        self.set_attr('rho', opt['rho'], dval=(50.0*self.lmbda + 1.0),
                      dtype=self.dtype)

        # Set rho_xi attribute (see Sec. VI.C of wohlberg-2015-adaptive)
        if self.lmbda != 0.0:
            rho_xi = float((1.0 + (18.3)**(np.log10(self.lmbda) + 1.0)))
        else:
            rho_xi = 1.0
        self.set_attr('rho_xi', opt['AutoRho', 'RsdlTarget'], dval=rho_xi,
                      dtype=self.dtype)

        super(BPDN, self).__init__(D, S, opt)



    def uinit(self, ushape):
        """Return initialiser for working variable U"""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            return (self.lmbda / self.rho) * np.sign(self.Y)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`."""

        self.Y = np.asarray(sp.prox_l1(self.AX + self.U,
                                       (self.lmbda / self.rho) * self.wl1),
                            dtype=self.dtype)
        super(BPDN, self).ystep()



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = np.linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        return (self.lmbda*rl1, rl1)




class BPDNJoint(BPDN):
    r"""
    ADMM algorithm for BPDN with joint sparsity via an :math:`\ell_{2,1}`
    norm term.

    |

    .. inheritance-diagram:: BPDNJoint
       :parts: 2

    |


    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_X \; (1/2) \| D X - S \|_2^2 + \lambda \| X \|_1
       + \mu \| X \|_{2,1}

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{X, Y} \; (1/2) \| D X - S \|_2^2 +
       \lambda \| Y \|_1 + \mu \| Y \|_{2,1} \quad \text{such that} \quad
       X = Y \;\;.

    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value

       ``DFid`` :  Value of data fidelity term :math:`(1/2) \| D X - S
       \|_2^2`

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
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid',
                     u('Regℓ1'): 'RegL1', u('Regℓ2,1'): 'RegL21'}



    def __init__(self, D, S, lmbda=None, mu=0.0, opt=None):
        """
        |

        **Call graph**

        .. image:: ../_static/jonga/bpdnjnt_init.svg
           :width: 20%
           :target: ../_static/jonga/bpdnjnt_init.svg

        |


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
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`."""

        self.Y = np.asarray(sp.prox_sl1l2(
            self.AX + self.U, (self.lmbda / self.rho) * self.wl1,
            self.mu / self.rho, axis=-1), dtype=self.dtype)
        GenericBPDN.ystep(self)



    def obfn_reg(self):
        r"""Compute regularisation terms and contribution to objective
        function. Regularisation terms are :math:`\| Y \|_1` and
        :math:`\| Y \|_{2,1}`.
        """

        rl1 = np.linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rl21 = np.sum(np.sqrt(np.sum(self.obfn_gvar()**2, axis=1)))
        return (self.lmbda*rl1 + self.mu*rl21, rl1, rl21)





class ElasticNet(BPDN):
    r"""
    ADMM algorithm for the elastic net :cite:`zou-2005-regularization`
    problem.

    |

    .. inheritance-diagram:: ElasticNet
       :parts: 2

    |

    Solve the optimisation problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x}
       \|_1 + (\mu/2) \| \mathbf{x} \|_2^2

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{y}
       \|_1 + (\mu/2) \| \mathbf{x} \|_2^2 \quad \text{such that} \quad
       \mathbf{x} = \mathbf{y} \;\;.

    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

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
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid',
                     u('Regℓ1'): 'RegL1', u('Regℓ2'): 'RegL2'}



    def __init__(self, D, S, lmbda=None, mu=0.0, opt=None):
        """
        |

        **Call graph**

        .. image:: ../_static/jonga/elnet_init.svg
           :width: 20%
           :target: ../_static/jonga/elnet_init.svg

        |


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
        self.lu, self.piv = sl.cho_factor(self.D, self.mu + self.rho)
        self.lu = np.asarray(self.lu, dtype=self.dtype)



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.X = np.asarray(sl.cho_solve_ATAI(
            self.D, self.mu + self.rho, self.DTS +
            self.rho * (self.Y - self.U),
            self.lu, self.piv), dtype=self.dtype)

        if self.opt['LinSolveCheck']:
            b = self.DTS + self.rho * (self.Y - self.U)
            ax = self.D.T.dot(self.D.dot(self.X)) + (self.mu+self.rho)*self.X
            self.xrrs = sl.rrs(ax, b)
        else:
            self.xrrs = None



    def obfn_reg(self):
        """Compute regularisation term and contribution to objective
        function.
        """

        rl1 = np.linalg.norm((self.wl1 * self.obfn_gvar()).ravel(), 1)
        rl2 = 0.5 * np.linalg.norm(self.obfn_gvar())**2
        return (self.lmbda*rl1 + self.mu*rl2, rl1, rl2)



    def rhochange(self):
        """Re-factorise matrix when rho changes."""

        self.lu, self.piv = sl.cho_factor(self.D, self.mu + self.rho)
        self.lu = np.asarray(self.lu, dtype=self.dtype)





class BPDNProjL1(GenericBPDN):
    r"""
    ADMM algorithm for a BPDN variant with projection onto the
    :math:`\ell_1` ball instead of an :math:`\ell_1` penalty.

    |

    .. inheritance-diagram:: BPDNProjL1
       :parts: 2

    |

    This variant of the BPDN problem was originally referred to as the
    lasso :cite:`tibshirani-1996-regression`, but that name is now also
    frequently applied to the penalised form that is referred to here as
    the BPDN problem.

    Solve the problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 \; \text{such that} \;
       \| \mathbf{x} \|_1 \leq \gamma

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x}, \mathbf{y}} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \iota_{C(\gamma)}
       (\mathbf{y}) \quad \text{such that} \quad \mathbf{x} = \mathbf{y}
       \;\;,

    where :math:`\iota_{C(\gamma)}(\cdot)` is the indicator function
    of the :math:`\ell_1` ball of radius :math:`\gamma` about the origin.
    The algorithm is very similar to that for the BPDN problem (see
    :class:`BPDN`), the only difference being in the replacement in the
    :math:`\mathbf{y}` step of the proximal operator of the :math:`\ell_1`
    norm with the projection operator of the :math:`\ell_1` norm.

    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value :math:`(1/2) \| D
       \mathbf{x} - \mathbf{s} \|_2^2`

       ``Cnstr`` : Constraint violation measure

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
        """BPDNProjL1 algorithm options

        Options are the same as those defined in
        :class:`.GenericBPDN.Options`.
        """

        defaults = copy.deepcopy(GenericBPDN.Options.defaults)
        defaults['AutoRho'].update({'RsdlTarget': 1.0})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              BPDNProjL1 algorithm options
            """

            if opt is None:
                opt = {}
            GenericBPDN.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'Cnstr')
    hdrtxt_objfn = ('Fnc', 'Cnstr')
    hdrval_objfun = {'Fnc': 'ObjFun', 'Cnstr': 'Cnstr'}



    def __init__(self, D, S, gamma, opt=None):
        """
        |

        **Call graph**

        .. image:: ../_static/jonga/bpdnprjl1_init.svg
           :width: 20%
           :target: ../_static/jonga/bpdnprjl1_init.svg

        |


        Parameters
        ----------
        D : array_like, shape (N, M)
          Dictionary matrix
        S : array_like, shape (N, K)
          Signal vector or matrix
        gamma : float
          Constraint parameter
        opt : :class:`BPDNProjL1.Options` object
          Algorithm options
        """

        # Set default options if necessary
        if opt is None:
            opt = BPDNProjL1.Options()

        super(BPDNProjL1, self).__init__(D, S, opt)
        self.gamma = self.dtype.type(gamma)



    def uinit(self, ushape):
        """Return initialiser for working variable U."""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            # NB: still needs to be worked out.
            return np.zeros(ushape, dtype=self.dtype)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y = np.asarray(sp.proj_l1(self.AX + self.U, self.gamma, axis=0),
                            dtype=self.dtype)
        super(BPDNProjL1, self).ystep()



    def eval_objfn(self):
        """Compute components of regularisation function as well as total
        contribution to objective function.
        """

        dfd = self.obfn_dfd()
        prj = sp.proj_l1(self.obfn_gvar(), self.gamma, axis=0)
        cns = np.linalg.norm(prj - self.obfn_gvar())
        return (dfd, cns)





class MinL1InL2Ball(admm.ADMMTwoBlockCnstrnt):
    r"""
    ADMM algorithm for the problem with an :math:`\ell_1` objective and
    an :math:`\ell_2` constraint.

    |

    .. inheritance-diagram:: MinL1InL2Ball
       :parts: 2

    |

    The solution is computed following the approach proposed in
    :cite:`afonso-2011-augmented`.

    Solve the Single Measurement Vector (SMV) problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \| \mathbf{x} \|_1 \; \text{such that}
       \; \| D \mathbf{x} - \mathbf{s} \|_2 \leq \epsilon

    via the ADMM problem

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
       \| \mathbf{y}_0 \|_1 + \iota_{C(\mathbf{s}, \epsilon)}
       (\mathbf{y}_1) \;\text{such that}\;
       \left( \begin{array}{c} I \\ D \end{array} \right) \mathbf{x}
       - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1
       \end{array} \right) = \left( \begin{array}{c} \mathbf{0} \\
       \mathbf{0} \end{array} \right) \;\;,

    where :math:`\iota_{C(\mathbf{s}, \epsilon)}(\cdot)` is the indicator
    function of the :math:`\ell_2` ball of radius :math:`\epsilon` about
    :math:`\mathbf{s}`. The Multiple Measurement Vector (MMV) problem

    .. math::
       \mathrm{argmin}_X \| X \|_1 \; \text{such that} \;
       \| [D X - S]_k \|_2 \leq \epsilon \;\;\; \forall k \;\;,

    where :math:`[X]_k` denotes column :math:`k` of matrix :math:`X`,
    is also supported.

    After termination of the :meth:`solve` method, attribute
    :attr:`itstat` is a list of tuples representing statistics of each
    iteration. The fields of the named tuple ``IterationStats`` are:

       ``Iter`` : Iteration number

       ``ObjFun`` : Objective function value :math:`\| \mathbf{x} \|_1`

       ``Cnstr`` : Constraint violation measure

       ``PrimalRsdl`` : Norm of primal residual

       ``DualRsdl`` : Norm of dual residual

       ``EpsPrimal`` : Primal residual stopping tolerance
       :math:`\epsilon_{\mathrm{pri}}`

       ``EpsDual`` : Dual residual stopping tolerance
       :math:`\epsilon_{\mathrm{dua}}`

       ``Rho`` : Penalty parameter

       ``Time`` : Cumulative run time
    """


    class Options(admm.ADMMTwoBlockCnstrnt.Options):
        r"""MinL1InL2Ball algorithm options

        Options include all of those defined in
        :class:`.admm.ADMMTwoBlockCnstrnt.Options`, together
        with additional options:

          ``L1Weight`` : An array of weights for the :math:`\ell_1`
          norm. The array shape must be such that the array is
          compatible for multiplication with the X/Y variables. If this
          option is defined, the objective function is :math:`\lambda \|
          \mathbf{w} \odot \mathbf{x} \|_1` where :math:`\mathbf{w}`
          denotes the weighting array.

          ``NonNegCoef`` : If ``True``, force solution to be non-negative.
        """

        defaults = copy.deepcopy(admm.ADMMTwoBlockCnstrnt.Options.defaults)
        defaults.update({'AuxVarObj': False, 'fEvalX': True,
                         'gEvalY': False, 'RelaxParam': 1.8,
                         'L1Weight': 1.0, 'NonNegCoef': False,
                         'ReturnVar': 'X'})
        defaults['AutoRho'].update({'Enabled': True, 'Period': 10,
                                    'AutoScaling': True, 'Scaling': 1000.0,
                                    'RsdlRatio': 1.2, 'RsdlTarget': 1.0})

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              MinL1InL2Ball algorithm options
            """

            if opt is None:
                opt = {}
            admm.ADMMTwoBlockCnstrnt.Options.__init__(self, opt)



    itstat_fields_objfn = ('ObjFun', 'Cnstr')
    hdrtxt_objfn = ('Fnc', 'Cnstr')
    hdrval_objfun = {'Fnc': 'ObjFun', 'Cnstr': 'Cnstr'}



    def __init__(self, D, S, epsilon, opt=None):
        r"""
        |

        **Call graph**

        .. image:: ../_static/jonga/bpdnml1l2_init.svg
           :width: 20%
           :target: ../_static/jonga/bpdnml1l2_init.svg

        |


        Parameters
        ----------
        D : array_like, shape (N, M)
          Dictionary matrix
        S : array_like, shape (N, K)
          Signal vector or matrix
        epsilon : float
          :math:`\ell_2` ball radius
        opt : :class:`MinL1InL2Ball.Options` object
          Algorithm options
        """

        Nr, Nc = D.shape
        Nm = S.shape[1]
        if opt is None:
            opt = MinL1InL2Ball.Options()
        super(MinL1InL2Ball, self).__init__(Nc * Nm, (Nc + Nr, Nm), 0, Nc,
                                            S.dtype, opt)

        # Record epsilon value and l1 weight array
        self.epsilon = self.dtype.type(epsilon)
        self.wl1 = np.asarray(opt['L1Weight'], dtype=self.dtype)

        self.S = np.asarray(S, dtype=self.dtype)
        self.setdict(D)



    def uinit(self, ushape):
        """Return initialiser for working variable U."""

        if self.opt['Y0'] is None:
            return np.zeros(ushape, dtype=self.dtype)
        else:
            # If initial Y is non-zero, initial U is chosen so that
            # the relevant dual optimality criterion (see (3.10) in
            # boyd-2010-distributed) is satisfied.
            U0 = np.sign(self.block_sep0(self.Y)) / self.rho
            U1 = self.block_sep1(self.Y) - self.S
            return self.block_cat(U0, U1)



    def setdict(self, D):
        """Set dictionary array."""

        self.D = np.asarray(D, dtype=self.dtype)
        # Factorise dictionary for efficient solves
        self.lu, self.piv = sl.cho_factor(self.D, 1.0)
        self.lu = np.asarray(self.lu, dtype=self.dtype)



    def getcoef(self):
        """Get final coefficient array."""

        return self.X



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        YU = self.Y - self.U
        self.X = np.asarray(sl.cho_solve_ATAI(
            self.D, 1.0, self.block_sep0(YU) +
            self.D.T.dot(self.block_sep1(YU)), self.lu, self.piv),
            dtype=self.dtype)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        AXU = self.AX + self.U
        Y0 = np.asarray(sp.prox_l1(self.block_sep0(AXU), self.wl1 / self.rho),
                        dtype=self.dtype)
        if self.opt['NonNegCoef']:
            Y0[Y0 < 0.0] = 0.0
        Y1 = self.S + sp.proj_l2(self.block_sep1(AXU) - self.S, self.epsilon,
                                 axis=0)
        self.Y = self.block_cat(Y0, Y1)



    def cnst_A1(self, X):
        r"""Compute :math:`A_1 \mathbf{x}` component of ADMM problem
        constraint."""

        return self.D.dot(X)



    def cnst_A1T(self, X):
        r"""Compute :math:`A_1^T \mathbf{x}` where :math:`A_1 \mathbf{x}`
        is a component of ADMM problem constraint."""

        return self.D.T.dot(X)



    def eval_objfn(self):
        r"""Compute components of objective function as well as total
        contribution to objective function.  The objective function is
        :math:`\| \mathbf{x} \|_1` and the constraint violation
        measure is :math:`P(\mathbf{x}) - \mathbf{x}` where
        :math:`P(\mathbf{x})` is the projection into the constraint
        set.
        """

        obj = np.linalg.norm((self.wl1 * self.obfn_g0var()).ravel(), 1)
        cns = np.linalg.norm(self.S + sp.proj_l2(self.obfn_g1var() - self.S,
                                                 self.epsilon, axis=0) -
                             self.obfn_g1var())
        return (obj, cns)



    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        return self.rho * np.linalg.norm(self.cnst_AT(self.U))



    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho * np.linalg.norm(U)
