#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate call graph images for sporco docs."""

from __future__ import print_function

import os
import sys

import jonga



def module_path(name):
    """
    Get path to module file from the fully qualified module name
    """

    return sys.modules[name].__file__



def is_newer_than(pth1, pth2):
    """
    Return true if either file pth1 or file pth2 don't exist, or if
    pth1 has been modified more recently than pth2
    """

    return not os.path.exists(pth1) or not os.path.exists(pth2) or \
       os.stat(pth1).st_mtime > os.stat(pth2).st_mtime



class CallGraph(jonga.ContextCallTracer):
    """
    Add some additional features to jonga.ContextCallTracer
    """

    def __init__(self, ct, mdnm, pth, fnm, **kwargs):
        """
        The actual image path for jonga.ContextCallTracer is obtained
        by joining pth and fnm. An attribute need_update depends on
        whether the module file corresponding to module name mdnm has
        been modified more recently than the image file referenced by
        this path.
        """

        img = os.path.join(pth, fnm)
        self.need_update = is_newer_than(module_path(mdnm), img)
        if self.need_update:
            print('constructing %s            ' % fnm, end='\r')
        super(CallGraph, self).__init__(ct, img, **kwargs)


    def __enter__(self):
        """
        Pass call on to corresponding method of parent class if need_update
        flag is True.
        """

        if self.need_update:
            return super(CallGraph, self).__enter__()
        else:
            return self


    def __exit__(self, type, value, traceback):
        """
        Pass call on to corresponding method of parent class if need_update
        flag is True.
        """

        if self.need_update:
            return super(CallGraph, self).__exit__(type, value, traceback)
        else:
            if type:
                return False
            else:
                return True



def gengraphs(pth):
    """
    Generate call graph images when necessary. Parameter pth is the path
    to the directory in which images are to be created.
    """

    srcmodflt = '^sporco.admm'
    srcqnmflt = r'^((?!<locals>|__new|_name_nested).)*$'
    dstqnmflt = r'^((?!<locals>|__new|_name_nested).)*$'

    fnmsub = ('^sporco.admm.', '')
    grpflt = r'^[^\.]*.[^\.]*'
    lnkpfx = '../../modules/'
    lnksub = (r'^([^\.]*).([^\.]*)(?:(.__init__|.__call__)|(.[^\.]*))',
              lnkpfx + r'sporco.admm.\1.html#sporco.admm.\1.\2\4')


    fntsz = 9
    fntfm = 'Vera Sans, DejaVu Sans, Liberation Sans, Arial, Helvetica, sans'
    kwargs = {'fntsz': fntsz, 'fntfm': fntfm, 'rmsz': True}


    ct = jonga.CallTracer(srcmodflt=srcmodflt, srcqnmflt=srcqnmflt,
                          dstqnmflt=dstqnmflt, fnmsub=fnmsub,
                          grpflt=grpflt, lnksub=lnksub)


    # Make destination directory if it doesn't exist
    if not os.path.exists(pth):
        os.makedirs(pth, exist_ok=True)

    import numpy as np
    np.random.seed(12345)


    #### bpdn module
    from sporco.admm import bpdn
    mdnm = 'sporco.admm.bpdn'

    D = np.random.randn(8, 16)
    s = np.random.randn(8, 1)
    lmbda = 0.1

    ## BPDN class
    opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'bpdn_init.svg', **kwargs):
        b = bpdn.BPDN(D, s, lmbda, opt)

    with CallGraph(ct, mdnm, pth, 'bpdn_solve.svg', **kwargs):
        b.solve()


    ## BPDNJoint class
    opt = bpdn.BPDNJoint.Options({'Verbose': False, 'MaxMainIter': 1})
    mu = 0.01

    with CallGraph(ct, mdnm, pth, 'bpdnjnt_init.svg', **kwargs):
        b = bpdn.BPDNJoint(D, s, lmbda, mu, opt)

    with CallGraph(ct, mdnm, pth, 'bpdnjnt_solve.svg', **kwargs):
        b.solve()


    ## ElasticNet class
    opt = bpdn.ElasticNet.Options({'Verbose': False, 'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'elnet_init.svg', **kwargs):
        b = bpdn.ElasticNet(D, s, lmbda, mu, opt)

    with CallGraph(ct, mdnm, pth, 'elnet_solve.svg', **kwargs):
        b.solve()


    # BPDNProjL1 class
    opt = bpdn.BPDNProjL1.Options({'Verbose': False, 'MaxMainIter': 1})
    gamma = 2.0

    with CallGraph(ct, mdnm, pth, 'bpdnprjl1_init.svg', **kwargs):
        b = bpdn.BPDNProjL1(D, s, gamma, opt)

    with CallGraph(ct, mdnm, pth, 'bpdnprjl1_solve.svg', **kwargs):
        b.solve()


    ## MinL1InL2Ball class
    opt = bpdn.MinL1InL2Ball.Options({'Verbose': False, 'MaxMainIter': 1})
    epsilon = 1.0

    with CallGraph(ct, mdnm, pth, 'bpdnml1l2_init.svg', **kwargs):
        b = bpdn.MinL1InL2Ball(D, s, epsilon, opt)

    with CallGraph(ct, mdnm, pth, 'bpdnml1l2_solve.svg', **kwargs):
        b.solve()




    #### cbpdn module
    from sporco.admm import cbpdn
    mdnm = 'sporco.admm.cbpdn'

    D = np.random.randn(4, 4, 16)
    s = np.random.randn(8, 8)
    lmbda = 0.1

    ## ConvBPDN class
    opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'cbpdn_init.svg', **kwargs):
        b = cbpdn.ConvBPDN(D, s, lmbda, opt)

    with CallGraph(ct, mdnm, pth, 'cbpdn_solve.svg', **kwargs):
        b.solve()


    ## ConvBPDNJoint class
    opt = cbpdn.ConvBPDNJoint.Options({'Verbose': False, 'MaxMainIter': 1})
    mu = 0.01

    with CallGraph(ct, mdnm, pth, 'cbpdnjnt_init.svg', **kwargs):
        b = cbpdn.ConvBPDNJoint(D, s, lmbda, mu, opt)

    with CallGraph(ct, mdnm, pth, 'cbpdnjnt_solve.svg', **kwargs):
        b.solve()


    ## ConvElasticNet class
    opt = cbpdn.ConvElasticNet.Options({'Verbose': False, 'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'celnet_init.svg', **kwargs):
        b = cbpdn.ConvElasticNet(D, s, lmbda, mu, opt)

    with CallGraph(ct, mdnm, pth, 'celnet_solve.svg', **kwargs):
        b.solve()


    ## ConvBPDNGradReg class
    opt = cbpdn.ConvBPDNGradReg.Options({'Verbose': False, 'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'cbpdngrd_init.svg', **kwargs):
        b = cbpdn.ConvBPDNGradReg(D, s, lmbda, mu, opt)

    with CallGraph(ct, mdnm, pth, 'cbpdngrd_solve.svg', **kwargs):
        b.solve()


    ## ConvBPDNProjL1 class
    opt = cbpdn.ConvBPDNProjL1.Options({'Verbose': False, 'MaxMainIter': 1})
    gamma = 0.5

    with CallGraph(ct, mdnm, pth, 'cbpdnprjl1_init.svg', **kwargs):
        b = cbpdn.ConvBPDNProjL1(D, s, gamma, opt)

    with CallGraph(ct, mdnm, pth, 'cbpdnprjl1_solve.svg', **kwargs):
        b.solve()


    ## ConvMinL1InL2Ball class
    opt = cbpdn.ConvMinL1InL2Ball.Options({'Verbose': False, 'MaxMainIter': 1})
    epsilon = 0.5

    with CallGraph(ct, mdnm, pth, 'cbpdnml1l2_init.svg', **kwargs):
        b = cbpdn.ConvMinL1InL2Ball(D, s, epsilon, opt)

    with CallGraph(ct, mdnm, pth, 'cbpdnml1l2_solve.svg', **kwargs):
        b.solve()


    ## ConvBPDNMaskDcpl class
    opt = cbpdn.ConvBPDNMaskDcpl.Options({'Verbose': False, 'MaxMainIter': 1})
    W = np.ones(s.shape)

    with CallGraph(ct, mdnm, pth, 'cbpdnmd_init.svg', **kwargs):
        b = cbpdn.ConvBPDNMaskDcpl(D, s, lmbda, W, opt)

    with CallGraph(ct, mdnm, pth, 'cbpdnmd_solve.svg', **kwargs):
        b.solve()


    ## ConvL1L1Grd class
    opt = cbpdn.ConvL1L1Grd.Options({'Verbose': False, 'MaxMainIter': 1})
    mu = 1e-2

    with CallGraph(ct, mdnm, pth, 'cl1l1grd_init.svg', **kwargs):
        b = cbpdn.ConvL1L1Grd(D, s, lmbda, mu, W, opt)

    with CallGraph(ct, mdnm, pth, 'cl1l1grd_solve.svg', **kwargs):
        b.solve()




    #### cbpdntv module
    from sporco.admm import cbpdntv
    mdnm = 'sporco.admm.cbpdntv'

    D = np.random.randn(4, 4, 16)
    s = np.random.randn(8, 8)
    lmbda = 0.1
    mu = 0.01

    ## ConvBPDNScalarTV class
    opt = cbpdntv.ConvBPDNScalarTV.Options({'Verbose': False,
                                            'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'cbpdnstv_init.svg', **kwargs):
        b = cbpdntv.ConvBPDNScalarTV(D, s, lmbda, mu, opt)

    with CallGraph(ct, mdnm, pth, 'cbpdnstv_solve.svg', **kwargs):
        b.solve()


    ## ConvBPDNVectorTV class
    opt = cbpdntv.ConvBPDNVectorTV.Options({'Verbose': False,
                                            'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'cbpdnvtv_init.svg', **kwargs):
        b = cbpdntv.ConvBPDNVectorTV(D, s, lmbda, mu, opt)

    with CallGraph(ct, mdnm, pth, 'cbpdnvtv_solve.svg', **kwargs):
        b.solve()


    ## ConvBPDNRecTV class
    opt = cbpdntv.ConvBPDNRecTV.Options({'Verbose': False, 'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'cbpdnrtv_init.svg', **kwargs):
        b = cbpdntv.ConvBPDNRecTV(D, s, lmbda, mu, opt)

    with CallGraph(ct, mdnm, pth, 'cbpdnrtv_solve.svg', **kwargs):
        b.solve()




    #### cbpdnin module
    from sporco.admm import cbpdnin
    mdnm = 'sporco.admm.cbpdnin'

    D = np.random.randn(4, 4, 32)
    s = np.random.randn(8, 8)
    lmbda = 0.1
    mu = 0.01
    Wg = np.append(np.eye(16), np.eye(16), axis=-1)

    ## ConvBPDNInhib class
    opt = cbpdnin.ConvBPDNInhib.Options({'Verbose': False, 'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'cbpdnin_init.svg', **kwargs):
        b = cbpdnin.ConvBPDNInhib(D, s, Wg, Whn=4, lmbda=lmbda, mu=mu,
                                   gamma=None, opt=opt)

    with CallGraph(ct, mdnm, pth, 'cbpdnin_solve.svg', **kwargs):
        b.solve()




    #### cmod module
    from sporco.admm import cmod
    mdnm = 'sporco.admm.cmod'

    X = np.random.randn(8, 16)
    S = np.random.randn(8, 16)

    ## CnstrMOD class
    opt = cmod.CnstrMOD.Options({'Verbose': False, 'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'cmod_init.svg', **kwargs):
        b = cmod.CnstrMOD(X, S, opt=opt)

    with CallGraph(ct, mdnm, pth, 'cmod_solve.svg', **kwargs):
        b.solve()




    #### ccmod module
    from sporco.admm import ccmod
    mdnm = 'sporco.admm.ccmod'

    X = np.random.randn(8, 8, 1, 2, 1)
    S = np.random.randn(8, 8, 2)
    dsz = (4, 4, 1)

    ## ConvCnstrMOD_IterSM class
    opt = ccmod.ConvCnstrMOD_IterSM.Options({'Verbose': False,
                                             'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'ccmodism_init.svg', **kwargs):
        b = ccmod.ConvCnstrMOD_IterSM(X, S, dsz=dsz, opt=opt)

    with CallGraph(ct, mdnm, pth, 'ccmodism_solve.svg', **kwargs):
        b.solve()


    ## ConvCnstrMOD_CG class
    opt = ccmod.ConvCnstrMOD_CG.Options({'Verbose': False,
                                         'MaxMainIter': 1,
                                         'CG': {'MaxIter': 1}})

    with CallGraph(ct, mdnm, pth, 'ccmodcg_init.svg', **kwargs):
        b = ccmod.ConvCnstrMOD_CG(X, S, dsz=dsz, opt=opt)

    with CallGraph(ct, mdnm, pth, 'ccmodcg_solve.svg', **kwargs):
        b.solve()


    ## ConvCnstrMOD_Consensus class
    opt = ccmod.ConvCnstrMOD_Consensus.Options({'Verbose': False,
                                                'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'ccmodcnsns_init.svg', **kwargs):
        b = ccmod.ConvCnstrMOD_Consensus(X, S, dsz=dsz, opt=opt)

    with CallGraph(ct, mdnm, pth, 'ccmodcnsns_solve.svg', **kwargs):
        b.solve()




    #### ccmodmd module
    from sporco.admm import ccmodmd
    mdnm = 'sporco.admm.ccmodmd'

    X = np.random.randn(8, 8, 1, 2, 1)
    S = np.random.randn(8, 8, 2)
    W = np.array([1.0])
    dsz = (4, 4, 1)

    ## ConvCnstrMODMaskDcpl_IterSM class
    opt = ccmodmd.ConvCnstrMODMaskDcpl_IterSM.Options({'Verbose': False,
                                             'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'ccmodmdism_init.svg', **kwargs):
        b = ccmodmd.ConvCnstrMODMaskDcpl_IterSM(X, S, W, dsz=dsz, opt=opt)

    with CallGraph(ct, mdnm, pth, 'ccmodmdism_solve.svg', **kwargs):
        b.solve()


    ## ConvCnstrMODMaskDcpl_CG class
    opt = ccmodmd.ConvCnstrMODMaskDcpl_CG.Options({'Verbose': False,
                                         'MaxMainIter': 1,
                                         'CG': {'MaxIter': 1}})

    with CallGraph(ct, mdnm, pth, 'ccmodmdcg_init.svg', **kwargs):
        b = ccmodmd.ConvCnstrMODMaskDcpl_CG(X, S, W, dsz=dsz, opt=opt)

    with CallGraph(ct, mdnm, pth, 'ccmodmdcg_solve.svg', **kwargs):
        b.solve()


    ## ConvCnstrMODMaskDcpl_Consensus class
    opt = ccmodmd.ConvCnstrMODMaskDcpl_Consensus.Options({'Verbose': False,
                                                'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'ccmodmdcnsns_init.svg', **kwargs):
        b = ccmodmd.ConvCnstrMODMaskDcpl_Consensus(X, S, W, dsz=dsz, opt=opt)

    with CallGraph(ct, mdnm, pth, 'ccmodmdcnsns_solve.svg', **kwargs):
        b.solve()




    #### tvl1 module
    from sporco.admm import tvl1
    mdnm = 'sporco.admm.tvl1'

    s = np.random.randn(16, 16)
    lmbda = 0.1

    ## TVL1Denoise class
    opt = tvl1.TVL1Denoise.Options({'Verbose': False, 'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'tvl1den_init.svg', **kwargs):
        b = tvl1.TVL1Denoise(s, lmbda, opt)

    with CallGraph(ct, mdnm, pth, 'tvl1den_solve.svg', **kwargs):
        b.solve()


    ## TVL1Deconv class
    opt = tvl1.TVL1Deconv.Options({'Verbose': False, 'MaxMainIter': 1})
    h = np.random.randn(3, 3)

    with CallGraph(ct, mdnm, pth, 'tvl1dcn_init.svg', **kwargs):
        b = tvl1.TVL1Deconv(h, s, lmbda, opt)

    with CallGraph(ct, mdnm, pth, 'tvl1dcn_solve.svg', **kwargs):
        b.solve()




    #### tvl2 module
    from sporco.admm import tvl2
    mdnm = 'sporco.admm.tvl2'

    s = np.random.randn(16, 16)
    lmbda = 0.1

    ## TVL2Denoise class
    opt = tvl2.TVL2Denoise.Options({'Verbose': False, 'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'tvl2den_init.svg', **kwargs):
        b = tvl2.TVL2Denoise(s, lmbda, opt)

    with CallGraph(ct, mdnm, pth, 'tvl2den_solve.svg', **kwargs):
        b.solve()


    ## TVL2Deconv class
    opt = tvl2.TVL2Deconv.Options({'Verbose': False, 'MaxMainIter': 1})
    h = np.random.randn(3, 3)

    with CallGraph(ct, mdnm, pth, 'tvl2dcn_init.svg', **kwargs):
        b = tvl2.TVL2Deconv(h, s, lmbda, opt)

    with CallGraph(ct, mdnm, pth, 'tvl2dcn_solve.svg', **kwargs):
        b.solve()



    srcmodflt = '^sporco.pgm'
    fnmsub = ('^sporco.pgm.', '')
    lnksub = (r'^([^\.]*).([^\.]*)(?:(.__init__|.__call__)|(.[^\.]*))',
              lnkpfx + r'sporco.pgm.\1.html#sporco.pgm.\1.\2\4')
    ct = jonga.CallTracer(srcmodflt=srcmodflt, srcqnmflt=srcqnmflt,
                          dstqnmflt=dstqnmflt, fnmsub=fnmsub,
                          grpflt=grpflt, lnksub=lnksub)


    #### pgm.cbpdn module
    from sporco.pgm import cbpdn
    mdnm = 'sporco.pgm.cbpdn'

    D = np.random.randn(4, 4, 16)
    s = np.random.randn(8, 8)
    lmbda = 0.1

    ## ConvBPDN class
    opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'pgm_cbpdn_init.svg', **kwargs):
        b = cbpdn.ConvBPDN(D, s, lmbda, opt)

    with CallGraph(ct, mdnm, pth, 'pgm_cbpdn_solve.svg', **kwargs):
        b.solve()




    #### pgm.ccmod module
    from sporco.pgm import ccmod
    mdnm = 'sporco.pgm.ccmod'

    X = np.random.randn(8, 8, 1, 2, 1)
    S = np.random.randn(8, 8, 2)
    dsz = (4, 4, 1)

    ## ConvCnstrMOD class
    opt = ccmod.ConvCnstrMOD.Options({'Verbose': False, 'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'ccmodpgm_init.svg', **kwargs):
        b = ccmod.ConvCnstrMOD(X, S, dsz=dsz, opt=opt)

    with CallGraph(ct, mdnm, pth, 'ccmodpgm_solve.svg', **kwargs):
        b.solve()


    ## ConvCnstrMODMask class
    opt = ccmod.ConvCnstrMODMask.Options({'Verbose': False,
                                          'MaxMainIter': 1})

    with CallGraph(ct, mdnm, pth, 'ccmodmdpgm_init.svg', **kwargs):
        b = ccmod.ConvCnstrMODMask(X, S, W, dsz=dsz, opt=opt)

    with CallGraph(ct, mdnm, pth, 'ccmodmdpgm_solve.svg', **kwargs):
        b.solve()




    srcmodflt = '^sporco.dictlrn'
    fnmsub = ('^sporco.dictlrn.', '')
    lnksub = (r'^([^\.]*).([^\.]*)(?:(.__init__|.__call__)|(.[^\.]*))',
              lnkpfx + r'sporco.dictlrn.\1.html#sporco.dictlrn.\1.\2\4')
    ct = jonga.CallTracer(srcmodflt=srcmodflt, srcqnmflt=srcqnmflt,
                          dstqnmflt=dstqnmflt, fnmsub=fnmsub,
                          grpflt=grpflt, lnksub=lnksub)



    #### bpdndl module
    from sporco.dictlrn import bpdndl
    mdnm = 'sporco.dictlrn.bpdndl'

    D0 = np.random.randn(8, 8)
    S = np.random.randn(8, 16)
    lmbda = 0.1

    ## BPDNDictLearn class
    opt = bpdndl.BPDNDictLearn.Options({'Verbose': False, 'MaxMainIter': 1,
                                        'AccurateDFid': True})

    with CallGraph(ct, mdnm, pth, 'bpdndl_init.svg', **kwargs):
        b = bpdndl.BPDNDictLearn(D0, S, lmbda, opt)

    with CallGraph(ct, mdnm, pth, 'bpdndl_solve.svg', **kwargs):
        b.solve()




    #### cbpdndl module
    from sporco.dictlrn import cbpdndl
    mdnm = 'sporco.dictlrn.cbpdndl'

    D0 = np.random.randn(4, 4, 16)
    s = np.random.randn(8, 8, 10)
    lmbda = 0.1

    ## ConvBPDNDictLearn class
    opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': False,
                    'MaxMainIter': 1, 'AccurateDFid': True})

    with CallGraph(ct, mdnm, pth, 'cbpdndl_init.svg', **kwargs):
        b = cbpdndl.ConvBPDNDictLearn(D0, s, lmbda, opt)

    with CallGraph(ct, mdnm, pth, 'cbpdndl_solve.svg', **kwargs):
        b.solve()




    #### cbpdndlmd module
    from sporco.dictlrn import cbpdndlmd
    mdnm = 'sporco.dictlrn.cbpdndlmd'

    ## ConvBPDNMaskDcplDictLearn class
    W = np.array([1.0])
    opt = cbpdndlmd.ConvBPDNMaskDictLearn.Options({'Verbose': False,
                    'MaxMainIter': 1, 'AccurateDFid': True})

    with CallGraph(ct, mdnm, pth, 'cbpdnmddl_init.svg', **kwargs):
        b = cbpdndlmd.ConvBPDNMaskDictLearn(D0, s, lmbda, W, opt)

    with CallGraph(ct, mdnm, pth, 'cbpdnmddl_solve.svg', **kwargs):
        b.solve()






def make_doc_func(fnm):
    """
    Construct a trivial function with a docstring that includes a
    specified call graph image
    """

    def doc_fun(*args):
        pass

    doc_fun.__doc__ = """

        **Call graph**

        .. image:: ../_static/jonga/%s
           :width: 20%%
           :target: ../_static/jonga/%s\n""" % (fnm, fnm)

    return doc_fun



def insert_solve_docs():
    global sporco

    import sporco.admm.bpdn
    import sporco.admm.cbpdn
    import sporco.admm.cbpdntv
    import sporco.admm.cmod
    import sporco.admm.ccmod
    import sporco.admm.ccmodmd
    import sporco.dictlrn.bpdndl
    import sporco.dictlrn.cbpdndl
    import sporco.dictlrn.cbpdndlmd
    import sporco.admm.tvl1
    import sporco.admm.tvl2


    # Classes that require a call graph for their solve method, and
    # corresponding call graph images
    clsgrph = {
        'sporco.admm.bpdn.BPDN': 'bpdn_solve.svg',
        'sporco.admm.bpdn.BPDNJoint': 'bpdnjnt_solve.svg',
        'sporco.admm.bpdn.ElasticNet': 'elnet_solve.svg',
        'sporco.admm.bpdn.BPDNProjL1': 'bpdnprjl1_solve.svg',
        'sporco.admm.bpdn.MinL1InL2Ball': 'bpdnml1l2_solve.svg',
        'sporco.admm.cbpdn.ConvBPDN': 'cbpdn_solve.svg',
        'sporco.admm.cbpdn.ConvBPDNJoint': 'cbpdnjnt_solve.svg',
        'sporco.admm.cbpdn.ConvElasticNet': 'cbpdnjnt_solve.svg',
        'sporco.admm.cbpdn.ConvBPDNGradReg': 'cbpdngrd_solve.svg',
        'sporco.admm.cbpdn.ConvBPDNProjL1': 'cbpdnprjl1_solve.svg',
        'sporco.admm.cbpdn.ConvMinL1InL2Ball': 'cbpdnml1l2_solve.svg',
        'sporco.admm.cbpdn.ConvBPDNMaskDcpl': 'cbpdnmd_solve.svg',
        'sporco.admm.cbpdn.ConvL1L1Grd': 'cl1l1grd_solve.svg',
        'sporco.admm.cbpdntv.ConvBPDNScalarTV': 'cbpdnstv_solve.svg',
        'sporco.admm.cbpdntv.ConvBPDNVectorTV': 'cbpdnvtv_solve.svg',
        'sporco.admm.cbpdntv.ConvBPDNRecTV': 'cbpdnrtv_solve.svg',
        'sporco.admm.cbpdnin.ConvBPDNInhib': 'cbpdnin_solve.svg',
        'sporco.admm.cmod.CnstrMOD': 'cmod_solve.svg',
        'sporco.admm.ccmod.ConvCnstrMOD_IterSM': 'ccmodism_solve.svg',
        'sporco.admm.ccmod.ConvCnstrMOD_CG': 'ccmodcg_solve.svg',
        'sporco.admm.ccmod.ConvCnstrMOD_Consensus': 'ccmodcnsns_solve.svg',
        'sporco.admm.ccmodmd.ConvCnstrMODMaskDcpl_IterSM':
            'ccmodmdism_solve.svg',
        'sporco.admm.ccmodmd.ConvCnstrMODMaskDcpl_CG': 'ccmodmdcg_solve.svg',
        'sporco.admm.ccmodmd.ConvCnstrMODMaskDcpl_Consensus':
            'ccmodmdcnsns_solve.svg',
        'sporco.dictlrn.bpdndl.BPDNDictLearn': 'bpdndl_solve.svg',
        'sporco.dictlrn.cbpdndl.ConvBPDNDictLearn': 'cbpdndl_solve.svg',
        'sporco.dictlrn.cbpdndlmd.ConvBPDNMaskDictLearn':
            'cbpdnmddl_solve.svg',
        'sporco.admm.tvl1.TVL1Denoise': 'tvl1den_solve.svg',
        'sporco.admm.tvl1.TVL1Deconv': 'tvl1dcn_solve.svg',
        'sporco.admm.tvl2.TVL2Denoise': 'tvl2den_solve.svg',
        'sporco.admm.tvl2.TVL2Deconv': 'tvl2dcn_solve.svg',
        'sporco.pgm.cbpdn.ConvBPDN': 'pgm_cbpdn_solve.svg',
        'sporco.pgm.ccmod.ConvCnstrMOD': 'ccmodpgm_solve.svg',
        'sporco.pgm.ccmod.ConvCnstrMODMask': 'ccmodmdpgm_solve.svg'
    }

    # Iterate over fully qualified class names in class/call graph image dict
    for fqclsnm in clsgrph:
        clspth = fqclsnm.split('.')
        # Name of module
        mdnm = '.'.join(clspth[0:-1])
        # Name of class
        clsnm = clspth[-1]
        # Get class reference
        cls = getattr(sys.modules[mdnm], clsnm)
        # Construct trivial function with appropriate docstring
        fnc = make_doc_func(clsgrph[fqclsnm])
        # Set solve method of current class to constructed function
        setattr(cls, 'solve', fnc)
