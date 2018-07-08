# -*- coding: utf-8 -*-
# Copyright (C) 2018 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Common infrastructure for some of the dictionary learning modules"""

from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import object

from sporco.util import u


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def evlmap(accdfid):
    """Return ``evlmap`` argument for ``.IterStatsConfig`` initialiser.
    """

    if accdfid:
        evlmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1'}
    else:
        evlmap = {}
    return evlmap



def isxmap(xmethod, opt):
    """Return ``isxmap`` argument for ``.IterStatsConfig`` initialiser.
    """

    if xmethod == 'admm':
        isxmap = {'XPrRsdl': 'PrimalRsdl', 'XDlRsdl': 'DualRsdl',
                  'XRho': 'Rho'}
    else:
        isxmap = {'X_F_Btrack': 'F_Btrack', 'X_Q_Btrack': 'Q_Btrack',
                  'X_ItBt': 'IterBTrack', 'X_L': 'L', 'X_Rsdl': 'Rsdl'}
    if not opt['AccurateDFid']:
        isxmap.update(evlmap(True))
    return isxmap



def isdmap(dmethod):
    """Return ``isdmap`` argument for ``.IterStatsConfig`` initialiser.
    """

    if dmethod == 'fista':
        isdmap = {'Cnstr': 'Cnstr', 'D_F_Btrack': 'F_Btrack',
                  'D_Q_Btrack': 'Q_Btrack', 'D_ItBt': 'IterBTrack',
                  'D_L': 'L', 'D_Rsdl': 'Rsdl'}
    else:
        isdmap= {'Cnstr':  'Cnstr', 'DPrRsdl': 'PrimalRsdl',
                 'DDlRsdl': 'DualRsdl', 'DRho': 'Rho'}
    return isdmap



def isfld(xmethod, dmethod, opt):
    """Return ``isfld`` argument for ``.IterStatsConfig`` initialiser.
    """

    isfld = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr']
    if xmethod == 'admm':
        isfld.extend(['XPrRsdl', 'XDlRsdl', 'XRho'])
    else:
        if opt['CBPDN', 'BackTrack', 'Enabled']:
            isfld.extend(['X_F_Btrack', 'X_Q_Btrack', 'X_ItBt', 'X_L',
                          'X_Rsdl'])
        else:
            isfld.extend(['X_L', 'X_Rsdl'])
    if dmethod != 'fista':
        isfld.extend([ 'DPrRsdl', 'DDlRsdl', 'DRho'])
    else:
        if opt['CCMOD', 'BackTrack', 'Enabled']:
            isfld.extend(['D_F_Btrack', 'D_Q_Btrack', 'D_ItBt', 'D_L',
                          'D_Rsdl'])
        else:
            isfld.extend(['D_L', 'D_Rsdl'])
    isfld.extend('Time')
    return isfld



def hdrtxt(xmethod, dmethod, opt):
    """Return ``hdrtxt`` argument for ``.IterStatsConfig`` initialiser.
    """

    hdrtxt = ['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr']
    if xmethod == 'admm':
        hdrtxt.extend(['r_X', 's_X', u('ρ_X')])
    else:
        if opt['CBPDN', 'BackTrack', 'Enabled']:
            hdrtxt.extend(['F_X', 'Q_X', 'It_X', 'L_X'])
        else:
            hdrtxt.append('L_X')
    if dmethod != 'fista':
        hdrtxt.extend(['r_D', 's_D', u('ρ_D')])
    else:
        if opt['CCMOD', 'BackTrack', 'Enabled']:
            hdrtxt.extend(['F_D', 'Q_D', 'It_D', 'L_D'])
        else:
            hdrtxt.append('L_D')
    return hdrtxt



def hdrmap(xmethod, dmethod, opt):
    """Return ``hdrmap`` argument for ``.IterStatsConfig`` initialiser.
    """

    hdrmap = {'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
              u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr'}
    if xmethod == 'admm':
        hdrmap.update({'r_X': 'XPrRsdl', 's_X': 'XDlRsdl',
                       u('ρ_X'): 'XRho'})
    else:
        if opt['CBPDN', 'BackTrack', 'Enabled']:
            hdrmap.update({'F_X': 'X_F_Btrack','Q_X': 'X_Q_Btrack',
                           'It_X': 'X_ItBt', 'L_X': 'X_L'})
        else:
            hdrmap.update({'L_X': 'X_L'})
    if dmethod != 'fista':
        hdrmap.update({'r_D': 'DPrRsdl', 's_D': 'DDlRsdl',
                       u('ρ_D'): 'DRho'})
    else:
        if opt['CCMOD', 'BackTrack', 'Enabled']:
            hdrmap.update({'F_D': 'D_F_Btrack', 'Q_D': 'D_Q_Btrack',
                           'It_D': 'D_ItBt', 'L_D': 'D_L'})
        else:
            hdrmap.update({'L_D': 'D_L'})
    return hdrmap
