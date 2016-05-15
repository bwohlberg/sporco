from builtins import object

import pytest

from sporco import cdict


class Options(cdict.ConstrainedDict):

    defaults = {'C' : { 'CA' : { 'CAA' : 'caa', 'CAB' : None },
                        'CB' : 'cb' }}

    def __init__(self, opts={}):
        super(self.__class__, self).__init__(
            {'C' : { 'CA' : { 'CAB' : 'cab' }}}
            )
        self.update(opts)




class TestSet01(object):

    def setup_method(self, method):
        cdict.ConstrainedDict.defaults = \
            {'A' : 'a',
                'B' : { 'BA' : 'ba', 'BB' : 'bb'},
                'C' : { 'CA' : { 'CAA' : 'caa' }},
                'D' : { 'DA' : { 'DAA' : {'DAAA' : 'daaa'},
                                 'DAB' : 'dab', 'DAC' : 'dac'}}}

        self.a = cdict.ConstrainedDict()
        self.b = Options()
        self.c = Options({'C' : { 'CB' : 'cb2'}})


    def test_01(self):
        assert(self.a['A'] == 'a')

    def test_02(self):
        with pytest.raises(cdict.UnknownKeyError):
            self.a['Ax'] = 'a'

    def test_03(self):
        with pytest.raises(cdict.InvalidValueError):
            self.a['C', 'CA'] = 'ca'

    def test_04(self):
        assert(self.a['C','CA','CAA'] == 'caa')

    def test_05(self):
        with pytest.raises(cdict.UnknownKeyError):
            self.a['C','CA','CAAx'] == 'caa'

    def test_06(self):
        assert(isinstance(self.a['C','CA'], cdict.ConstrainedDict))

    def test_07(self):
        self.a['D', 'DA'].update({'DAB' : 'dab2'})
        assert(self.a['D', 'DA', 'DAA', 'DAAA'] == 'daaa')

    def test_08(self):
        assert(type(self.b) == Options)

    def test_09(self):
        assert(type(self.b['C']) == cdict.ConstrainedDict)

    def test_10(self):
        assert(self.b['C', 'CA', 'CAB'] == 'cab')

    def test_11(self):
        assert(self.c['C', 'CA', 'CAB'] == 'cab')
