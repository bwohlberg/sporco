# -*- coding: utf-8 -*-
# Copyright (C) 2019-2020 by Brendt Wohlberg <brendt@ieee.org>
#                            Ulugbek Kamilov <kamilov@wustl.edu>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Classes for PGM variant of the Plug and Play Priors (PPP) algorithm."""

from __future__ import division, absolute_import, print_function

import numpy as np

from sporco.pgm import pgm


__author__ = """\n""".join(['Brendt Wohlberg <brendt@ieee.org>',
                            'Ulugbek Kamilov <kamilov@wustl.edu>'])



class GenericPPP(pgm.PGM):
    """Base class for Plug and Play Priors (PPP) PGM solvers
    :cite:`kamilov-2017-plugandplay`."""

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

        self.Y = self.X.copy()
        self.Yprv = np.zeros(self.Y.shape)



    itstat_fields_objfn = ('FVal',)
    hdrtxt_objfn = ('FVal',)
    hdrval_objfun = {'FVal': 'FVal'}



    def grad_f(self):
        """Compute the gradient of :math:`f`."""

        return self.gradf(self.Y)



    def prox_g(self, V):
        """Compute proximal operator of :math:`g`."""

        return self.proxg(V, self.L)



    def rsdl(self):
        """Compute fixed point residual."""

        return np.linalg.norm((self.X - self.Yprv).ravel())



    def eval_objfn(self):
        r"""Compute components of objective function.

        In this case the regularisation term is implicit so we can only
        evaluate the data fidelity term represented by the
        :math:`f(\cdot)` component of the functional to be minimised.
        """

        return (self.f(self.X),)



    def gradf(self, X):
        r"""Compute the gradient of :math:`f(\cdot)`.

        Overriding this method is required.
        """

        raise NotImplementedError()



    def proxg(self, X, L):
        r"""Compute the proximal operator of :math:`L^{-1} g(\cdot)`.

        Overriding this method is required. Note that this method
        should compute the proximal operator of
        :math:`L^{-1} g(\cdot)`, *not* the proximal operator
        of :math:`L g(\cdot)`.
        """

        raise NotImplementedError()



    def f(self, X):
        r"""Evauate the data fidelity term :math:`f(\mathbf{x})`.

        Overriding this method is required.
        """

        raise NotImplementedError()





class PPP(GenericPPP):
    """Plug and Play Priors (PPP) solver :cite:`kamilov-2017-plugandplay`
    that can be used without the need to derive a new class."""

    def __init__(self, xshape, f, gradf, proxg, opt=None):
        """
        Parameters
        ----------
        xshape : tuple of ints
          Shape of working variable X
        f : function
          Function evaluating the data fidelity term
        gradf : function
          Function computing the gradient of the data fidelity term
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
        self.gradf = gradf
        self.proxg = proxg
