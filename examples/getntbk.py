#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Get notebooks from sporco-notebooks on GitHub."""

from __future__ import print_function

import os
import sys
import zipfile

sys.path.insert(0, '..')
from sporco.util import netgetdata


if os.path.exists('notebooks'):
    print('Error: notebooks directory already exists')
else:
    url = 'https://codeload.github.com/bwohlberg/sporco-notebooks/zip/master'
    zipdat = netgetdata(url)
    zipobj = zipfile.ZipFile(zipdat)
    zipobj.extractall()
    os.symlink('sporco-notebooks-master', 'notebooks')
