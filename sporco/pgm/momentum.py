# -*- coding: utf-8 -*-
# Copyright (C) 2016-2020 by Cristina Garcia-Cardona <cgarciac@lanl.gov>
#                            Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Momentum coefficient options for PGM algorithms"""

from __future__ import division, print_function

import numpy as np


__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""



class MomentumBase(object):
    """Base class for computing momentum coefficient for
    accelerated proximal gradient method.

    This class is intended to be a base class of other classes
    that specialise to specific momentum coefficient options.

    After termination of the :meth:`update` method the new
    momentum coefficient is returned.
    """

    def __init__(self):
        super(MomentumBase, self).__init__()



    def update(self):
        """Update momentum coefficient.

        Overriding this method is required.
        """

        raise NotImplementedError()





class MomentumNesterov(MomentumBase):
    r"""Nesterov's momentum coefficient :cite:`beck-2009-fast`

    Applies the update

    .. math::
      t^{(k+1)} = \frac{1}{2} \left( 1
      + \sqrt{1 + 4 \; (t^{(k)})^2} \right) \;,

    with :math:`k` iteration.
    """

    def __init__(self):
        super(MomentumNesterov, self).__init__()



    def update(self, t):
        """Update momentum coefficient"""

        return 0.5 * float(1. + np.sqrt(1. + 4. * t**2))





class MomentumLinear(MomentumBase):
    r"""Linear momentum coefficient :cite:`chambolle-2015-convergence`

    Applies the update

    .. math::
       t^{(k+1)} = \frac{k + b}{b} \;,

    with :math:`b` corresponding to a positive constant such
    that :math:`b \geq 2` and :math:`k` iteration.
    """

    def __init__(self, b=2.):
        """
        Parameters
        ----------
        b : float
          Summand in numerator and factor in
          denominator of update.
        """

        super(MomentumLinear, self).__init__()
        self.b = b



    def update(self, k):
        """Update momentum coefficient"""

        return (k + self.b) / self.b





class MomentumGenLinear(MomentumBase):
    r"""Generalized linear momentum coefficient
    :cite:`rodriguez-2019-convergence`

    Applies the update

    .. math::
       t^{(k+1)} = \frac{k + a}{b} \;,

    with :math:`a, b` corresponding to postive constants such
    that :math:`a \geq b - 1` and :math:`b \geq 2`, and :math:`k`
    iteration.
    """

    def __init__(self, a=50., b=2.):
        """
        Parameters
        ----------
        a : float
          Summand in numerator of update.
        b : float
          Factor in denominator of update.
        """

        super(MomentumGenLinear, self).__init__()
        self.a = a
        self.b = b



    def update(self, k):
        """Update momentum coefficient"""

        return (k + self.a) / self.b
