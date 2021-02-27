# -*- coding: utf-8 -*-
# Copyright (C) 2019-2020 by Brendt Wohlberg <brendt@ieee.org>
#                            Ulugbek Kamilov <kamilov@wustl.edu>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for ADMM variant of the Plug and Play Priors (PPP) algorithm."""

from __future__ import division, absolute_import, print_function

import numpy as np

from sporco.admm import admm


__author__ = """\n""".join(['Brendt Wohlberg <brendt@ieee.org>',
                            'Ulugbek Kamilov <kamilov@wustl.edu>'])


class GenericPPP(admm.ADMMEqual):
    """Base class for Plug and Play Priors (PPP) ADMM solvers
    :cite:`venkatakrishnan-2013-plugandplay2` :cite:`sreehari-2016-plug`."""

    def __init__(self, xshape, opt=None):
        """
        Parameters
        ----------
        xshape : tuple of ints
          Shape of working variable X
        opt : :class:`GenericPPP.Options` object
          Algorithm options
        """

        if opt is None:
            opt = GenericPPP.Options()

        # Set dtype attribute, default is np.float32
        self.set_dtype(opt, np.dtype(np.float32))

        super(GenericPPP, self).__init__(xshape, self.dtype, opt)



    itstat_fields_objfn = ('FVal',)
    hdrtxt_objfn = ('FVal',)
    hdrval_objfun = {'FVal': 'FVal'}



    def xstep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{x}`.
        """

        self.X = self.proxf(self.Y - self.U, self.rho)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y = self.proxg(self.AX + self.U, self.rho)



    def eval_objfn(self):
        r"""Compute components of objective function.

        In this case the regularisation term is implicit so we can only
        evaluate the data fidelity term represented by the
        :math:`f(\cdot)` component of the functional to be minimised.
        """

        return (self.f(self.X),)



    def proxf(self, X, rho):
        r"""Compute the proximal operator of :math:`\rho^{-1} f(\cdot)`.

        Overriding this method is required. Note that this method
        should compute the proximal operator of
        :math:`\rho^{-1} f(\cdot)`, *not* the proximal operator
        of :math:`\rho f(\cdot)`.
        """

        raise NotImplementedError()



    def proxg(self, X, rho):
        r"""Compute the proximal operator of :math:`\rho^{-1} g(\cdot)`.

        Overriding this method is required. Note that this method
        should compute the proximal operator of
        :math:`\rho^{-1} g(\cdot)`, *not* the proximal operator
        of :math:`\rho g(\cdot)`.
        """

        raise NotImplementedError()



    def f(self, X):
        r"""Evauate the data fidelity term :math:`f(\mathbf{x})`.

        Overriding this method is required.
        """

        raise NotImplementedError()





class PPP(GenericPPP):
    """Plug and Play Priors (PPP) solver
    :cite:`venkatakrishnan-2013-plugandplay2` :cite:`sreehari-2016-plug`
    that can be used without the need to derive a new class."""

    def __init__(self, xshape, f, proxf, proxg, opt=None):
        """
        Parameters
        ----------
        xshape : tuple of ints
          Shape of working variable X
        f : function
          Function evaluating the data fidelity term
        proxf : function
          Function computing the proximal operator of the data fidelity
          term
        proxg : function
          Function computing the proximal operator of the regularisation
          term
        opt : :class:`PPP.Options` object
          Algorithm options
        """

        if opt is None:
            opt = PPP.Options()

        super(PPP, self).__init__(xshape, opt)

        self.f = f
        self.proxf = proxf
        self.proxg = proxg





class GenericPPPConsensus(admm.WeightedADMMConsensus):
    """Base class for Plug and Play Priors (PPP) ADMM Consensus solvers

    This class solves the Multi-Agent Consensus Equilibrium problem
    :cite:`buzzard-2018-plug` via ADMM Consensus instead of the
    Douglas-Rachford algorithm used in :cite:`buzzard-2018-plug`. It can
    also be viewed as a variant of the Plug and Play Priors (PPP)
    approach :cite:`venkatakrishnan-2013-plugandplay2`
    :cite:`sreehari-2016-plug` based on a weighted version of the ADMM
    Consensus (see Ch. 7 of :cite:`boyd-2010-distributed`) problem
    instead of the simpler ADMM problem of the original PPP approach.
    """

    def __init__(self, xshape, Nb, mu=None, opt=None):
        """
        Parameters
        ----------
        xshape : tuple of ints
          Shape of working variable X
        Nb : int
          Number of consensus blocks
        mu : array_like
          Array of scalar weights
        opt : :class:`GenericPPPConsensus.Options` object
          Algorithm options
        """

        if mu is None:
            mu = np.ones((Nb,))
        if opt is None:
            opt = GenericPPPConsensus.Options()

        # Set dtype attribute, default is np.float32
        self.set_dtype(opt, np.dtype(np.float32))

        mu = np.array(mu, dtype=self.dtype)

        super(GenericPPPConsensus, self).__init__(Nb, mu, xshape,
                                                   self.dtype, opt)

        # Initialise working variable
        self.X = np.zeros(self.xshape)



    itstat_fields_objfn = ()
    hdrtxt_objfn = ()
    hdrval_objfun = {}



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.

        In this case we assume that there are no components that can be
        computed.
        """

        return ()



    def xistep(self, i):
        r"""Minimise Augmented Lagrangian with respect to
        ADMM Consensus component :math:`\mathbf{x}_i`.
        """

        self.X[..., i] = self.prox_fi(self.Y - self.U[..., i], self.rho, i)



    def prox_fi(self, X, rho):
        r"""Compute the proximal operator of
        :math:`\rho^{-1} f_i(\cdot)`.

        Overriding this method is required. Note that this method
        should compute the proximal operator of
        :math:`\rho^{-1} f_i(\cdot)`, *not* the proximal operator
        of :math:`\rho f_i(\cdot)`.
        """

        raise NotImplementedError()



    def prox_g(self, X, rho):
        r"""Compute the proximal operator of :math:`\rho^{-1} g(\cdot)`.

        If this method is not overridden, the default is to assume
        :math:`g(\mathbf{y}) = 0`, so that the corresponding proximal
        operator is the identity mapping. Note that this method
        should compute the proximal operator of
        :math:`\rho^{-1} g(\cdot)`, *not* the proximal operator
        of :math:`\rho g(\cdot)`.
        """

        return X




class PPPConsensus(GenericPPPConsensus):
    """Plug and Play Priors (PPP) ADMM Consensus solver that can be used
    without the need to derive a new class.

    This class solves the Multi-Agent Consensus Equilibrium problem
    :cite:`buzzard-2018-plug` via ADMM Consensus instead of the
    Douglas-Rachford algorithm used in :cite:`buzzard-2018-plug`.
    """

    def __init__(self, xshape, proxfi, proxg=None, mu=None, opt=None):
        """
        Parameters
        ----------
        xshape : tuple of ints
          Shape of working variable X
        proxfi : tuple of functions
          Tuple of functions that compute the proximal operators for
          each consensus block
        proxg : function
          Function computing the proximal operator of the regularisation
          term
        opt : :class:`PPPConsensus.Options` object
          Algorithm options
        """

        if opt is None:
            opt = PPPConsensus.Options()

        self._proxfi = proxfi
        if proxg is not None:
            self.prox_g = proxg

        super(PPPConsensus, self).__init__(xshape, len(proxfi), mu, opt)



    def prox_fi(self, Xi, rho, i):
        r"""Compute the proximal operator of
        :math:`\rho^{-1} f_i(\cdot)`.
        """

        return self._proxfi[i](Xi, rho)
