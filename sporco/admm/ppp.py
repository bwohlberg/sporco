# -*- coding: utf-8 -*-
# Copyright (C) 2019 by Brendt Wohlberg <brendt@ieee.org>
#                       Ulugbek Kamilov <kamilov@wustl.edu>
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
    """Base class for PPP ADMM solvers
    :cite:`venkatakrishnan-2013-plugandplay2` :cite:`sreehari-2016-plug`."""

    def __init__(self, xshape, opt=None):
        """
        Parameters
        ----------
        xshape : tuple of ints
          Shape of working variable X
        opt : :class:`PPP.Options` object
          Algorithm options
        """

        if opt is None:
            opt = GenericPPP.Options()

        # Set dtype attribute, default is np.float32
        self.set_dtype(opt, np.float32)

        super(GenericPPP, self).__init__(xshape, self.dtype, opt)



    itstat_fields_objfn = ('FVal',)
    hdrtxt_objfn = ('FVal',)
    hdrval_objfun = {'FVal': 'FVal'}



    def xstep(self):
        self.X = self.proxf(self.Y - self.U, self.rho)



    def ystep(self):
        self.Y = self.proxg(self.AX + self.U, self.rho)



    def eval_objfn(self):
        r"""Compute components of objective function. In this case the
        regularisation term is implicit so we can only evaluate the
        data fidelity term represented by the :math:`f(\cdot)`
        component of the functional to be minimised.
        """

        return (self.f(self.X),)




class PPP(GenericPPP):
    """PPP solver that can be used without the need to derive a new
    class."""

    def __init__(self, xshape, f, proxf, proxg, opt=None):
        """
        Parameters
        ----------
        xshape : tuple of ints
          Shape of working variable X
        f : function
          Evaluate the data fidelity term
        proxf : function
          Compute the proximal operator of the data fidelity term
        proxg : function
          Compute the proximal operator of the regularisation term
        opt : :class:`PPP.Options` object
          Algorithm options
        """

        if opt is None:
            opt = PPP.Options()

        super(PPP, self).__init__(xshape, opt)

        self.f = f
        self.proxf = proxf
        self.proxg = proxg
