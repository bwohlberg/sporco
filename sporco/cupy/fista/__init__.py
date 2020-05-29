# -*- coding: utf-8 -*-
# Copyright (C) 2018-2019 by Brendt Wohlberg <brendt@ieee.org>
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
from sporco.cupy import array
from sporco.cupy import linalg
from sporco.cupy import fft
from sporco.cupy import prox
from sporco.cupy import cnvrep


# Construct sporco.cupy.fista
fista = sporco_cupy_patch_module('sporco.fista')

# Construct sporco.cupy.fista.fista
fista.fista = sporco_cupy_patch_module('sporco.fista.fista',
             {'IterativeSolver': common.IterativeSolver,
              'rfftn': fft.rfftn, 'irfftn': fft.irfftn})


# Record current entries in sys.modules and then replace them with
# patched versions of the modules
sysmod = {}
for mod in ('sporco.common', 'sporco.fista', 'sporco.fista.fista'):
    if mod in sys.modules:
        sysmod[mod] = sys.modules[mod]
sys.modules['sporco.common'] = common
sys.modules['sporco.fista'] = fista
sys.modules['sporco.fista.fista'] = fista.fista


# Construct sporco.cupy.fista.cbpdn
fista.cbpdn = sporco_cupy_patch_module('sporco.fista.cbpdn',
            {'fista': fista.fista, 'inner':  linalg.inner,
             'CSC_ConvRepIndexing': cnvrep.CSC_ConvRepIndexing,
             'mskWshape': cnvrep.mskWshape, 'rfftn': fft.rfftn,
             'irfftn': fft.irfftn, 'empty_aligned': fft.empty_aligned,
             'rfftn_empty_aligned': fft.rfftn_empty_aligned,
             'rfl2norm2': fft.rfl2norm2, 'prox_l1': prox.prox_l1})


# Restore original entries in sys.modules
for mod in ('sporco.common', 'sporco.fista', 'sporco.fista.fista'):
    if mod in sysmod:
        sys.modules[mod] = sysmod[mod]
    else:
        del sys.modules[mod]


# In sporco.cupy.fista module, replace original module source path with
# corresponding path in 'sporco/cupy' directory tree
for n, pth in enumerate(sys.modules['sporco.cupy.fista'].__path__):
    pth = re.sub('sporco/', 'sporco/cupy/', pth)
    sys.modules['sporco.cupy.fista'].__path__[n] = pth
