# -*- coding: utf-8 -*-
# Copyright (C) 2020 by Cristina Garcia-Cardona <cgarciac@lanl.gov>
#                       Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Step size policies for PGM algorithms"""

from __future__ import division, print_function

import numpy as np


__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""



class StepSizePolicyBase(object):
    """Base class for computing step size policy for
    accelerated proximal gradient method.

    This class is intended to be a base class of other classes
    that specialise to specific step size policy options.

    After termination of the :meth:`update` method the new
    inverse of step size is returned.
    """

    def __init__(self):
        """Create an StepSizePolicyBase object"""

        super(StepSizePolicyBase, self).__init__()



    def update(self, solverobj):
        """Update step size.

        Overriding this method is required.
        """

        raise NotImplementedError()





class StepSizePolicyCauchy(StepSizePolicyBase):
    r"""Class for computing step size policy for
    accelerated proximal gradient based on Cauchy
    method :cite:`yuan-2008-stepsize`

    .. math::
       \alpha = \frac{\left\| \nabla f \right\|_2^2}
       {\nabla f^T \: \mathrm{Hess}_f \nabla f} \; \\
       L = 1 / \alpha \;\;.
    """

    def __init__(self):
        """Create an StepSizePolicyCauchy object"""
        super(StepSizePolicyCauchy, self).__init__()



    def update(self, solverobj, grad=None):
        """
        Parameters
        ----------
        solverobj : :class:`PGM` object
          containing the proximal gradient method state and functionality.
        grad : ndarray
          array containing the evaluation of gradient in current state.
        """

        if grad is None:
            grad = solverobj.grad_f()
        den = np.sum(np.real(np.conj(grad) * grad))
        num = np.sum(np.real(np.conj(grad) * solverobj.hessian_f(grad)))

        return num / den





class StepSizePolicyBB(StepSizePolicyBase):
    r"""Class for computing step size policy for
    accelerated proximal gradient based on
    Barzilai-Borwein method :cite:`barzilai-1988-stepsize`

    .. math::
       \Delta x = x_k - x_{k-1} \; \\
       \Delta g = \nabla f (x_k) - \nabla f (x_{k-1}) \; \\
       \alpha = \frac{\Delta x^T \Delta g}{\left\| \Delta g \right\|_2^2} \; \\
       L = 1 / \alpha \;\;.
    """

    def __init__(self):
        """Create an StepSizePolicyBB object"""

        super(StepSizePolicyBB, self).__init__()
        self.xprv = 0.0
        self.gradprv = 0.0



    def store_prev_state(self, xprv, gradprv):
        """
        Parameters
        ----------
        xprv : ndarray
          array containing the previous state.
        gradprv : ndarray
          array containing the evaluation of gradient in previous state.
        """

        self.xprv = xprv
        self.gradprv = gradprv



    def update(self, solverobj, grad=None):
        """
        Parameters
        ----------
        solverobj : :class:`PGM` object
          containing the proximal gradient method state and functionality.
        grad : ndarray
          array containing the evaluation of gradient in current state.
        """

        if grad is None:
            grad = solverobj.grad_f()
        dx = solverobj.var_x() - self.xprv
        dg = grad - self.gradprv
        den = np.sum(np.real(np.conj(dx) * dg))
        num = np.sum(np.real(np.conj(dg) * dg))

        L = num / den
        if L < 0.:
            L = solverobj.L

        return L
