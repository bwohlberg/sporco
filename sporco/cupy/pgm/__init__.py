# -*- coding: utf-8 -*-
# Copyright (C) 2018-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Construct variant of pgm subpackage that use cupy instead of numpy."""

from __future__ import absolute_import

import sys
import re

from sporco.cupy import sporco_cupy_patch_module
from sporco.cupy import common
from sporco.cupy import linalg
from sporco.cupy import fft
from sporco.cupy import prox
from sporco.cupy import cnvrep


# Construct sporco.cupy.pgm
pgm = sporco_cupy_patch_module('sporco.pgm')

# Construct cupy versions of sporco.pgm auxiliary modules
pgm.backtrack = sporco_cupy_patch_module('sporco.pgm.backtrack')
pgm.momentum = sporco_cupy_patch_module('sporco.pgm.momentum')
pgm.stepsize = sporco_cupy_patch_module('sporco.pgm.stepsize')

# Construct sporco.cupy.pgm.pgm
pgm.pgm = sporco_cupy_patch_module(
    'sporco.pgm.pgm',
    {'IterativeSolver': common.IterativeSolver,
     'rfftn': fft.rfftn, 'irfftn': fft.irfftn,
     'BacktrackStandard': pgm.backtrack.BacktrackStandard,
     'BacktrackRobust': pgm.backtrack.BacktrackRobust,
     'MomentumNesterov': pgm.momentum.MomentumNesterov,
     'MomentumLinear': pgm.momentum.MomentumLinear,
     'MomentumGenLinear': pgm.momentum.MomentumGenLinear,
     'StepSizePolicyCauchy': pgm.stepsize.StepSizePolicyCauchy,
     'StepSizePolicyBB': pgm.stepsize.StepSizePolicyBB})


# Record current entries in sys.modules and then replace them with
# patched versions of the modules
sysmod = {}
for mod in ('sporco.common', 'sporco.pgm', 'sporco.pgm.pgm'):
    if mod in sys.modules:
        sysmod[mod] = sys.modules[mod]
sys.modules['sporco.common'] = common
sys.modules['sporco.pgm'] = pgm
sys.modules['sporco.pgm.pgm'] = pgm.pgm


# Construct sporco.cupy.pgm.cbpdn
pgm.cbpdn = sporco_cupy_patch_module(
    'sporco.pgm.cbpdn',
    {'pgm': pgm.pgm, 'inner': linalg.inner,
     'CSC_ConvRepIndexing': cnvrep.CSC_ConvRepIndexing,
     'mskWshape': cnvrep.mskWshape, 'rfftn': fft.rfftn,
     'irfftn': fft.irfftn, 'empty_aligned': fft.empty_aligned,
     'rfftn_empty_aligned': fft.rfftn_empty_aligned,
     'rfl2norm2': fft.rfl2norm2, 'prox_l1': prox.prox_l1})


# Restore original entries in sys.modules
for mod in ('sporco.common', 'sporco.pgm', 'sporco.pgm.pgm'):
    if mod in sysmod:
        sys.modules[mod] = sysmod[mod]
    else:
        del sys.modules[mod]


# In sporco.cupy.pgm module, replace original module source path with
# corresponding path in 'sporco/cupy' directory tree
for n, pth in enumerate(sys.modules['sporco.cupy.pgm'].__path__):
    pth = re.sub('sporco/', 'sporco/cupy/', pth)
    sys.modules['sporco.cupy.pgm'].__path__[n] = pth
