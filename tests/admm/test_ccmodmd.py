from builtins import object

import numpy as np

from sporco.admm import ccmodmd



class TestSet01(object):

    def setup_method(self, method):
        np.random.seed(12345)


    def test_01(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N)
        W = np.array([1.0])
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_IterSM.Options(
                           {'Verbose': False, 'MaxMainIter': 20,
                            'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_IterSM(X, S, W,
                    (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_02(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N)
        W = np.random.randn(N, N)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_IterSM.Options(
                           {'Verbose': False, 'MaxMainIter': 20,
                            'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_IterSM(X, S, W,
                    (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_03(self):
        N = 16
        K = 3
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, K, M)
        S = np.random.randn(N, N, K)
        W = np.random.randn(N, N)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_IterSM.Options(
                           {'Verbose': False, 'MaxMainIter': 20,
                            'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_IterSM(X, S, W,
                                    (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_04(self):
        N = 16
        K = 3
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, K, M)
        S = np.random.randn(N, N, K)
        W = np.random.randn(N, N, K)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_IterSM.Options(
                           {'Verbose': False, 'MaxMainIter': 20,
                            'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_IterSM(X, S, W,
                                    (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_05(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        W = np.random.randn(N, N, Nc)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_IterSM.Options(
                           {'Verbose': False, 'MaxMainIter': 20,
                            'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_IterSM(X, S, W,
                            (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_06(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, Nc, K)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_IterSM.Options(
                           {'Verbose': False, 'MaxMainIter': 20,
                            'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_IterSM(X, S, W,
                            (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_07(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, Nc)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_IterSM.Options(
                           {'Verbose': False, 'MaxMainIter': 20,
                            'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_IterSM(X, S, W,
                            (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_08(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, 1, K)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_IterSM.Options(
                           {'Verbose': False, 'MaxMainIter': 20,
                            'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_IterSM(X, S, W,
                            (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_09(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, Nc, K)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_IterSM.Options(
                           {'Verbose': False, 'MaxMainIter': 20,
                            'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_IterSM(X, S, W,
                            (Nd, Nd, Nc, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_10(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N)
        W = np.random.randn(N, N)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_CG.Options({'Verbose': False,
                        'MaxMainIter': 20, 'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_CG(X, S, W,
                        (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_11(self):
        N = 16
        K = 3
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, K, M)
        S = np.random.randn(N, N, K)
        W = np.random.randn(N, N, K)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_CG.Options({'Verbose': False,
                        'MaxMainIter': 20, 'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_CG(X, S, W,
                                    (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_12(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        W = np.random.randn(N, N, Nc)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_CG.Options({'Verbose': False,
                        'MaxMainIter': 20, 'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_CG(X, S, W,
                            (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_13(self):
        N = 16
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, 1, M)
        S = np.random.randn(N, N)
        W = np.random.randn(N, N)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_Consensus.Options(
                {'Verbose': False, 'MaxMainIter': 20,
                 'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_Consensus(X, S, W,
                                (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_14(self):
        N = 16
        K = 3
        M = 4
        Nd = 8
        X = np.random.randn(N, N, 1, K, M)
        S = np.random.randn(N, N, K)
        W = np.random.randn(N, N)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_Consensus.Options(
                {'Verbose': False, 'MaxMainIter': 20,
                 'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_Consensus(X, S, W,
                                        (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_15(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        W = np.random.randn(N, N)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_Consensus.Options(
                {'Verbose': False, 'MaxMainIter': 20,
                 'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_Consensus(X, S, W,
                                (Nd, Nd, 1, M), opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_16(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_Consensus.Options(
                {'Verbose': False, 'MaxMainIter': 20,
                 'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_Consensus(X, S, W,
                                (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_17(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_Consensus.Options(
                {'Verbose': False,  'MaxMainIter': 20,
                 'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_Consensus(X, S, W,
                                (Nd, Nd, Nc, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_18(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, Nc)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_Consensus.Options(
                           {'Verbose': False, 'MaxMainIter': 20,
                            'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_Consensus(X, S, W,
                            (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_19(self):
        N = 16
        M = 4
        K = 2
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, K, M)
        S = np.random.randn(N, N, Nc, K)
        W = np.random.randn(N, N, 1, K)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcpl_Consensus.Options(
                           {'Verbose': False, 'MaxMainIter': 20,
                            'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl_Consensus(X, S, W,
                            (Nd, Nd, 1, M), opt=opt)
            c.solve()
        except Exception as e:
            print(e)
            assert 0


    def test_20(self):
        N = 16
        M = 4
        Nc = 3
        Nd = 8
        X = np.random.randn(N, N, Nc, 1, M)
        S = np.random.randn(N, N, Nc)
        W = np.random.randn(N, N)
        try:
            opt = ccmodmd.ConvCnstrMODMaskDcplOptions({'Verbose': False,
                            'MaxMainIter': 20, 'LinSolveCheck': True})
            c = ccmodmd.ConvCnstrMODMaskDcpl(X, S, W, (Nd, Nd, 1, M),
                                             opt=opt, dimK=0)
            c.solve()
        except Exception as e:
            print(e)
            assert 0
        assert np.array(c.getitstat().XSlvRelRes).max() < 1e-5


    def test_21(self):
        opt = ccmodmd.ConvCnstrMODMaskDcplBase.Options({'AuxVarObj': False})
        assert opt['fEvalX'] is True and opt['gEvalY'] is False
        opt['AuxVarObj'] = True
        assert opt['fEvalX'] is False and opt['gEvalY'] is True


    def test_22(self):
        opt = ccmodmd.ConvCnstrMODMaskDcplBase.Options({'AuxVarObj': True})
        assert opt['fEvalX'] is False and opt['gEvalY'] is True
        opt['AuxVarObj'] = False
        assert opt['fEvalX'] is True and opt['gEvalY'] is False
