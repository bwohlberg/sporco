# -*- coding: utf-8 -*-
# Copyright (C) 2018-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Construct variant of dictlrn subpackage that use cupy instead of numpy"""

from __future__ import absolute_import

import sys
import re

from sporco.cupy import sporco_cupy_patch_module
from sporco.cupy import cp
from sporco.cupy import array
from sporco.cupy import linalg
from sporco.cupy import fft
from sporco.cupy import cnvrep
from sporco.cupy.admm import cbpdn


# Construct sporco.cupy.dictlrn
dictlrn = sporco_cupy_patch_module('sporco.dictlrn')

# Construct sporco.cupy.dictlrn.onlinecdl
dictlrn.onlinecdl = sporco_cupy_patch_module('sporco.dictlrn.onlinecdl',
            {'cbpdn': cbpdn, 'inner': linalg.inner, 'rfftn': fft.rfftn,
             'irfftn': fft.irfftn, 'empty_aligned': fft.empty_aligned,
             'byte_aligned': fft.byte_aligned,
             'DictionarySize': cnvrep.DictionarySize,
             'stdformD': cnvrep.stdformD, 'Pcn': cnvrep.Pcn,
             'getPcn': cnvrep.getPcn,
             'CDU_ConvRepIndexing': cnvrep.CDU_ConvRepIndexing,
             'mskWshape': cnvrep.mskWshape, 'zpad': cnvrep.zpad})


# In sporco.cupy.dictlrn module, replace original module source path with
# corresponding path in 'sporco/cupy' directory tree
for n, pth in enumerate(sys.modules['sporco.cupy.dictlrn'].__path__):
    pth = re.sub('sporco/', 'sporco/cupy/', pth)
    sys.modules['sporco.cupy.dictlrn'].__path__[n] = pth
