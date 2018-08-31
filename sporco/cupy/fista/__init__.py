# -*- coding: utf-8 -*-
# Copyright (C) 2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Construct variant of fista subpackage that use cupy instead of numpy"""

from __future__ import absolute_import

import sys
import re

from sporco.cupy import sporco_cupy_patch_module
from sporco.cupy import cp
from sporco.cupy import util
from sporco.cupy import common
from sporco.cupy import linalg
from sporco.cupy import prox
from sporco.cupy import cnvrep


fista = sporco_cupy_patch_module('sporco.fista')

fista.fista = sporco_cupy_patch_module('sporco.fista.fista',
                                       {'util': util, 'common': common,
                                        'sl': linalg})


sysmod = {}
for mod in ('sporco.common', 'sporco.fista', 'sporco.fista.fista'):
    if mod in sys.modules:
        sysmod[mod] = sys.modules[mod]
sys.modules['sporco.common'] = common
sys.modules['sporco.fista'] = fista
sys.modules['sporco.fista.fista'] = fista.fista


fista.cbpdn = sporco_cupy_patch_module('sporco.fista.cbpdn',
                                       {'fista': fista.fista,
                                        'cr': cnvrep, 'sl': linalg,
                                        'sp': prox})


for mod in ('sporco.common', 'sporco.fista', 'sporco.fista.fista'):
    if mod in sysmod:
        sys.modules[mod] = sysmod[mod]
    else:
        del sys.modules[mod]


for n, pth in enumerate(sys.modules['sporco.cupy.fista'].__path__):
    pth = re.sub('sporco/', 'sporco/cupy/', pth)
    sys.modules['sporco.cupy.fista'].__path__[n] = pth
