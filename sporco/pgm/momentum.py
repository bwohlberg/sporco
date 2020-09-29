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

from sporco.cdict import ConstrainedDict


__author__ = """Cristina Garcia-Cardona <cgarciac@lanl.gov>"""



class MomentumBase(object):
    """Base class for computing momentum coefficient for
    accelerated proximal gradient method.

    This class is intended to be a base class of other classes
    that specialise to specific momentum coefficient options.

    After termination of the :meth:`update` method the new
    momentum coefficient is returned.
    """

    class Options(ConstrainedDict):
        """Momentum coefficient options.

        Options:

          ``a`` : float, used by :class:`Momentum_GenLinear`

          ``b`` : float, used by :class:`Momentum_Linear` and
            :class:`Momentum_GenLinear`
        """

        defaults = {'a': 50., 'b': 2.}

        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              PGM momentum coefficient options
            """

            if opt is None:
                opt = {}
            ConstrainedDict.__init__(self, opt)



    def __init__(self, opt=None):
        """
        Parameters
        ----------
        opt : :class:`MomentumBase.Options` object
          Momentum coefficient options
        """

        if opt is None:
            opt = MomentumBase.Options()
        if not isinstance(opt, MomentumBase.Options):
            raise TypeError('Parameter opt must be an instance of '
                            'MomentumBase.Options')
            self.opt = opt
        super(MomentumBase, self).__init__()



    def update(self):
        """Update momentum coefficient.

        Overriding this method is required.
        """

        raise NotImplementedError()





class Momentum_Nesterov(MomentumBase):
    """Nesterov's momentum coefficient :cite:`beck-2009-fast`
    """

    def __init__(self, opt=None):
        """
        Parameters
        ----------
        opt : :class:`MomentumBase.Options` object
          Momentum coefficient options
        """

        if opt is None:
            opt = MomentumBase.Options()
        if not isinstance(opt, MomentumBase.Options):
            raise TypeError('Parameter opt must be an instance of '
                            'MomentumBase.Options')
        super(Momentum_Nesterov, self).__init__(opt)



    def update(self, t):
        """Update momentum coefficient"""

        return 0.5 * float(1. + np.sqrt(1. + 4. * t**2))





class Momentum_Linear(MomentumBase):
    """Linear momentum coefficient :cite:`chambolle-2015-convergence`
    """

    def __init__(self, opt=None):
        """
        Parameters
        ----------
        opt : :class:`MomentumBase.Options` object
          Momentum coefficient options
        """

        if opt is None:
            opt = MomentumBase.Options()
        if not isinstance(opt, MomentumBase.Options):
            raise TypeError('Parameter opt must be an instance of '
                            'MomentumBase.Options')
        super(Momentum_Linear, self).__init__(opt)
        self.b = opt['b']



    def update(self, k):
        """Update momentum coefficient"""

        return ((k + 1.) - 1. + self.b) / self.b





class Momentum_GenLinear(MomentumBase):
    """Generalized linear momentum coefficient
    :cite:`rodriguez-2019-convergence`
    """

    def __init__(self, opt=None):
        """
        Parameters
        ----------
        opt : :class:`MomentumBase.Options` object
          Momentum coefficient options
        """

        if opt is None:
            opt = MomentumBase.Options()
        if not isinstance(opt, MomentumBase.Options):
            raise TypeError('Parameter opt must be an instance of '
                            'MomentumBase.Options')
        super(Momentum_GenLinear, self).__init__(opt)
        self.a = opt['a']
        self.b = opt['b']



    def update(self, k):
        """Update momentum coefficient"""

        return ((k + 1.) - 1. + self.a) / self.b
