# -*- coding: utf-8 -*-
# Copyright (C) 2015-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""Norms and their associated proximal maps and projections

   The :math:`p`-norm of a vector is defined as

   .. math::
    \| \mathbf{x} \|_p = \left( \sum_i | x_i |^p \right)^{1/p}

   where :math:`x_i` is element :math:`i` of vector :math:`\mathbf{x}`.
   The max norm is a special case

   .. math::
    \| \mathbf{x} \|_{\infty} = \max_i | x_i | \;\;.

   The mixed matrix norm :math:`\|X\|_{p,q}` is defined here as
   :cite:`kowalski-2009-sparse`

   .. math::
     \|X\|_{p,q} = \left( \sum_i \left( \sum_j |X_{i,j}|^p \right)^{q/p}
     \right)^{1/q} = \left( \sum_i \| \mathbf{x}_i  \|_p^q \right)^{1/q}

   where :math:`\mathbf{x}_i` is row :math:`i` of matrix
   :math:`X`. Note that some authors (e.g., see :cite:`sra-2011-fast`)
   use a notation that reverses the positions of :math:`p` and
   :math:`q`.

   The proximal operator of function :math:`f` with parameter
   :math:`\alpha` is defined as

   .. math::
    \mathrm{prox}_{\alpha f}(\mathbf{v}) = \mathrm{argmin}_{\mathbf{x}}
    \left\{ (1/2) \| \mathbf{x} - \mathbf{v} \|_2^2 + \alpha
    f(\mathbf{x}) \right\} \;\;.

   The projection operator of function :math:`f` is defined as

   .. math::
    \mathrm{proj}_{f,\gamma}(\mathbf{v}) &= \mathrm{argmin}_{\mathbf{x}}
    (1/2) \| \mathbf{x} - \mathbf{v} \|_2^2 \; \text{ s.t. } \;
    f(\mathbf{x}) \leq \gamma \\ &= \mathrm{prox}_g(\mathbf{v})

   where :math:`g(\mathbf{v}) = \iota_C(\mathbf{v})`, with
   :math:`\iota_C` denoting the indicator function of set
   :math:`C = \{ \mathbf{x} \; | \; f(\mathbf{x}) \leq \gamma \}`.

|
|

"""


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""

from ._util import *
from ._lp import *
from ._dl1l2 import *
from ._l1proj import *
from ._l21 import *
from ._nuclear import *

__all__ = ['ndto2d', 'ndfrom2d', 'norm_l0', 'prox_l0', 'norm_l1',
           'prox_l1', 'proj_l1', 'norm_2l2', 'norm_l2', 'norm_l21',
           'prox_l2', 'proj_l2', 'norm_dl1l2', 'prox_dl1l2',
           'prox_sl1l2', 'norm_nuclear',
           'prox_nuclear']
