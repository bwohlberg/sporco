# -*- coding: utf-8 -*-
# Copyright (C) 2018-2019 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Construct variant of admm subpackage that use cupy instead of numpy"""

from __future__ import absolute_import

import sys
import re

from sporco.cupy import sporco_cupy_patch_module
from sporco.cupy import cp
from sporco.cupy import util
from sporco.cupy import fft
from sporco.cupy import signal
from sporco.cupy import array
from sporco.cupy import common
from sporco.cupy import linalg
from sporco.cupy import prox
from sporco.cupy import cnvrep


# Construct sporco.cupy.admm
admm = sporco_cupy_patch_module('sporco.admm')

# Construct sporco.cupy.admm.admm
admm.admm = sporco_cupy_patch_module('sporco.admm.admm',
                                     {'util': util, 'common': common})


def _update_rho(self, k, r, s):
    """
    Patched version of :func:`sporco.admm.admm.ADMM.update_rho`."""

    if self.opt['AutoRho', 'Enabled']:
        tau = self.rho_tau
        mu = self.rho_mu
        xi = self.rho_xi
        if k != 0 and cp.mod(k + 1, self.opt['AutoRho', 'Period']) == 0:
            if self.opt['AutoRho', 'AutoScaling']:
                if s == 0.0 or r == 0.0:
                    rhomlt = tau
                else:
                    rhomlt = cp.sqrt(r / (s * xi) if r > s * xi
                                     else (s * xi) / r)
                    if rhomlt > tau:
                        rhomlt = tau
            else:
                rhomlt = tau
            rsf = 1.0
            if r > xi * mu * s:
                rsf = rhomlt
            elif s > (mu / xi) * r:
                rsf = 1.0 / rhomlt
            self.rho *= float(rsf)
            self.U /= rsf
            if rsf != 1.0:
                self.rhochange()


admm.admm.ADMM.update_rho = _update_rho


def _cnst0(self):
    return cp.array(0.0, dtype=self.dtype)


admm.admm.ADMMTwoBlockCnstrnt.cnst_c0 = _cnst0
admm.admm.ADMMTwoBlockCnstrnt.cnst_c1 = _cnst0


# Record current entries in sys.modules and then replace them with
# patched versions of the modules
sysmod = {}
for mod in ('sporco.common', 'sporco.admm', 'sporco.admm.admm'):
    if mod in sys.modules:
        sysmod[mod] = sys.modules[mod]
sys.modules['sporco.common'] = common
sys.modules['sporco.admm'] = admm
sys.modules['sporco.admm.admm'] = admm.admm


# Construct sporco.cupy.admm.rpca
admm.rpca = sporco_cupy_patch_module('sporco.admm.rpca',
                                     {'admm': admm.admm, 'sp': prox})

# Construct sporco.cupy.admm.tvl1
admm.tvl1 = sporco_cupy_patch_module('sporco.admm.tvl1',
            {'admm': admm.admm, 'rrs': linalg.rrs, 'prox_l1': prox.prox_l1,
             'prox_l2': prox.prox_l2, 'zpad': array.zpad,
             'atleast_nd': array.atleast_nd, 'zdivide': array.zdivide,
             'fftn_func': fft.fftn_func, 'ifftn_func': fft.ifftn_func,
             'gradient_filters': signal.gradient_filters,
             'grad': signal.grad, 'gradT': signal.gradT})

# Construct sporco.cupy.admm.tvl2
admm.tvl2 = sporco_cupy_patch_module('sporco.admm.tvl2',
            {'admm': admm.admm, 'rrs': linalg.rrs, 'prox_l2': prox.prox_l2,
             'zpad': array.zpad, 'atleast_nd': array.atleast_nd,
             'zdivide': array.zdivide,
             'fftn_func': fft.fftn_func, 'ifftn_func': fft.ifftn_func,
             'gradient_filters': signal.gradient_filters,
             'grad': signal.grad, 'gradT': signal.gradT})

# Construct sporco.cupy.admm.bpdn
admm.bpdn = sporco_cupy_patch_module('sporco.admm.bpdn',
                {'admm': admm.admm, 'sl': linalg, 'sp': prox})

# Construct sporco.cupy.admm.cbpdn
admm.cbpdn = sporco_cupy_patch_module('sporco.admm.cbpdn',
                {'admm': admm.admm, 'cr': cnvrep, 'sl': linalg, 'sp': prox,
                 'rfftn': fft.rfftn, 'irfftn': fft.irfftn,
                 'empty_aligned': fft.empty_aligned,
                 'empty_aligned_func': fft.empty_aligned_func,
                 'fftn_func': fft.fftn_func, 'ifftn_func': fft.ifftn_func,
                 'gradient_filters': signal.gradient_filters})


def _index_primary(self):
    return (Ellipsis, slice(0, -self.cri.Cd, None))


def _index_addmsk(self):
    return (Ellipsis, slice(-self.cri.Cd, None, None))


admm.cbpdn.AddMaskSim.index_primary = _index_primary
admm.cbpdn.AddMaskSim.index_addmsk = _index_addmsk


# Construct sporco.cupy.admm.cbpdnin
admm.cbpdnin = sporco_cupy_patch_module('sporco.admm.cbpdnin',
            {'cbpdn': admm.cbpdn, 'sl': linalg, 'sp': prox,
             'rfftn': fft.rfftn, 'irfftn': fft.irfftn})


# Construct sporco.cupy.admm.cbpdntv
admm.cbpdntv = sporco_cupy_patch_module('sporco.admm.cbpdntv',
            {'admm': admm.admm, 'cr': cnvrep, 'cbpdn': admm.cbpdn,
             'sl': linalg, 'sp': prox, 'rfftn': fft.rfftn,
             'irfftn': fft.irfftn, 'empty_aligned': fft.empty_aligned,
             'rfftn_empty_aligned': fft.rfftn_empty_aligned,
             'rfl2norm2': fft.rfl2norm2,
             'gradient_filters': signal.gradient_filters})

admm.cbpdntv.ConvBPDNScalarTV.cnst_c = _cnst0
admm.cbpdntv.ConvBPDNRecTV.cnst_c = _cnst0


# Construct sporco.cupy.admm.pdcsc
admm.pdcsc = sporco_cupy_patch_module('sporco.admm.pdcsc',
            {'admm': admm.admm, 'cr': cnvrep, 'cbpdn': admm.cbpdn,
            'rfftn': fft.rfftn, 'irfftn': fft.irfftn,
             'empty_aligned': fft.empty_aligned, 'rfl2norm2': fft.rfl2norm2,
             'dot': linalg.dot, 'inner': linalg.inner,
             'solvedbi_sm_c': linalg.solvedbi_sm_c,
             'solvedbi_sm': linalg.solvedbi_sm,
             'solvedbd_sm_c': linalg.solvedbd_sm_c,
             'solvedbd_sm': linalg.solvedbd_sm, 'rrs': linalg.rrs,
             'prox_l1': prox.prox_l1, 'prox_sl1l2': prox.prox_sl1l2
            })


# Restore original entries in sys.modules
for mod in ('sporco.common', 'sporco.admm', 'sporco.admm.admm'):
    if mod in sysmod:
        sys.modules[mod] = sysmod[mod]
    else:
        del sys.modules[mod]


# In sporco.cupy.admm module, replace original module source path with
# corresponding path in 'sporco/cupy' directory tree
for n, pth in enumerate(sys.modules['sporco.cupy.admm'].__path__):
    pth = re.sub('sporco/', 'sporco/cupy/', pth)
    sys.modules['sporco.cupy.admm'].__path__[n] = pth
