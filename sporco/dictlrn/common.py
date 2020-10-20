# -*- coding: utf-8 -*-
# Copyright (C) 2018-2020 by Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Common infrastructure for some of the dictionary learning modules"""

from __future__ import print_function, absolute_import

from sporco.util import u


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""



def evlmap(accdfid):
    """Return ``evlmap`` argument for ``.IterStatsConfig`` initialiser.
    """

    if accdfid:
        evl = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1'}
    else:
        evl = {}
    return evl



def isxmap(xmethod, opt):
    """Return ``isxmap`` argument for ``.IterStatsConfig`` initialiser.
    """

    if xmethod == 'admm':
        isx = {'XPrRsdl': 'PrimalRsdl', 'XDlRsdl': 'DualRsdl',
               'XRho': 'Rho'}
    else:
        isx = {'X_F_Btrack': 'F_Btrack', 'X_Q_Btrack': 'Q_Btrack',
               'X_ItBt': 'IterBTrack', 'X_L': 'L', 'X_Rsdl': 'Rsdl'}
    if not opt['AccurateDFid']:
        isx.update(evlmap(True))
    return isx



def isdmap(dmethod):
    """Return ``isdmap`` argument for ``.IterStatsConfig`` initialiser.
    """

    if dmethod == 'pgm':
        isd = {'Cnstr': 'Cnstr', 'D_F_Btrack': 'F_Btrack',
               'D_Q_Btrack': 'Q_Btrack', 'D_ItBt': 'IterBTrack',
               'D_L': 'L', 'D_Rsdl': 'Rsdl'}
    else:
        isd = {'Cnstr': 'Cnstr', 'DPrRsdl': 'PrimalRsdl',
               'DDlRsdl': 'DualRsdl', 'DRho': 'Rho'}
    return isd



def isfld(xmethod, dmethod, opt):
    """Return ``isfld`` argument for ``.IterStatsConfig`` initialiser.
    """

    fld = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr']
    if xmethod == 'admm':
        fld.extend(['XPrRsdl', 'XDlRsdl', 'XRho'])
    else:
        if opt['CBPDN', 'Backtrack'] is not None:
            fld.extend(['X_F_Btrack', 'X_Q_Btrack', 'X_ItBt', 'X_L',
                        'X_Rsdl'])
        else:
            fld.extend(['X_L', 'X_Rsdl'])
    if dmethod != 'pgm':
        fld.extend(['DPrRsdl', 'DDlRsdl', 'DRho'])
    else:
        if opt['CCMOD', 'Backtrack'] is not None:
            fld.extend(['D_F_Btrack', 'D_Q_Btrack', 'D_ItBt', 'D_L',
                        'D_Rsdl'])
        else:
            fld.extend(['D_L', 'D_Rsdl'])
    fld.append('Time')
    return fld



def hdrtxt(xmethod, dmethod, opt):
    """Return ``hdrtxt`` argument for ``.IterStatsConfig`` initialiser.
    """

    txt = ['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr']
    if xmethod == 'admm':
        txt.extend(['r_X', 's_X', u('ρ_X')])
    else:
        if opt['CBPDN', 'Backtrack'] is not None:
            txt.extend(['F_X', 'Q_X', 'It_X', 'L_X'])
        else:
            txt.append('L_X')
    if dmethod != 'pgm':
        txt.extend(['r_D', 's_D', u('ρ_D')])
    else:
        if opt['CCMOD', 'Backtrack'] is not None:
            txt.extend(['F_D', 'Q_D', 'It_D', 'L_D'])
        else:
            txt.append('L_D')
    return txt



def hdrmap(xmethod, dmethod, opt):
    """Return ``hdrmap`` argument for ``.IterStatsConfig`` initialiser.
    """

    hdr = {'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
           u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr'}
    if xmethod == 'admm':
        hdr.update({'r_X': 'XPrRsdl', 's_X': 'XDlRsdl', u('ρ_X'): 'XRho'})
    else:
        if opt['CBPDN', 'Backtrack'] is not None:
            hdr.update({'F_X': 'X_F_Btrack', 'Q_X': 'X_Q_Btrack',
                        'It_X': 'X_ItBt', 'L_X': 'X_L'})
        else:
            hdr.update({'L_X': 'X_L'})
    if dmethod != 'pgm':
        hdr.update({'r_D': 'DPrRsdl', 's_D': 'DDlRsdl', u('ρ_D'): 'DRho'})
    else:
        if opt['CCMOD', 'Backtrack'] is not None:
            hdr.update({'F_D': 'D_F_Btrack', 'Q_D': 'D_Q_Btrack',
                        'It_D': 'D_ItBt', 'L_D': 'D_L'})
        else:
            hdr.update({'L_D': 'D_L'})
    return hdr
