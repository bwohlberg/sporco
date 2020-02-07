#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build notebooks from example scripts."""

from __future__ import print_function

import os
import os.path
import sys
from glob import glob

sys.path.insert(0, '../docs/source')
import docntbk


envpth = '../build/sphinx/doctrees/environment.pickle'
invpth = '../build/sphinx/html/objects.inv'
baseurl = 'http://sporco.rtfd.org/en/latest/'
ppth = 'scripts'
npth = 'notebooks'
nbexec = True


# Iterate over index files
for fp in glob(os.path.join(ppth, '*.rst')) + \
          glob(os.path.join(ppth, '*', '*.rst')):
    # Index basename
    b = os.path.splitext(os.path.basename(fp))[0]
    # Name of subdirectory of examples directory containing current index
    sd = os.path.split(os.path.dirname(fp))
    if sd[-1] == ppth:
        d = ''
    else:
        d = sd[-1]
    # Path to corresponding subdirectory in notebooks directory
    fd = os.path.join(npth, d)
    # Ensure notebook subdirectory exists
    docntbk.mkdir(fd)
    # Filename of notebook index file to be constructed
    fn = os.path.join(fd, b + '.ipynb')
    # Process current index file if corresponding notebook file
    # doesn't exist, or is older than index file
    if docntbk.update_required(fp, fn):
        print('Converting %s' % fp)
        diridx = True if fp == 'scripts/index.rst' else False
        # Convert index to notebook
        docntbk.rst_to_notebook(fp, fn, diridx=diridx)


# Get intersphinx inventory and sphinx environment and construct cross
# reference lookup object
try:
    inv = docntbk.fetch_intersphinx_inventory(invpth)
except Exception:
    inv = None
try:
    env = docntbk.read_sphinx_environment(envpth)
except Exception:
    env = None
if inv is not None and env is not None:
    cr = docntbk.CrossReferenceLookup(env, inv, baseurl)
else:
    cr = None
    print('Warning: intersphinx inventory or sphinx environment not found:'
          ' cross-references will not be handled correctly')

# Iterate over example scripts
for fp in sorted(glob(os.path.join(ppth, '*', '*.py'))):
    # Name of subdirectory of examples directory containing current script
    d = os.path.split(os.path.dirname(fp))[1]
    # Script basename
    b = os.path.splitext(os.path.basename(fp))[0]
    # Path to corresponding subdirectory in notebooks directory
    fd = os.path.join(npth, d)
    # Make notebooks subdirectory if it doesn't exist
    docntbk.mkdir(fd)
    # Filename of notebook file to be constructed
    fn = os.path.join(fd, b + '.ipynb')
    # Process current example script if corresponding notebook file
    # doesn't exist, or is older than example script file
    if docntbk.update_required(fp, fn):
        print('Converting %s' % fp)
        # Convert script to notebook
        docntbk.script_to_notebook(fp, fn, cr)

# Execute notebooks if requested
if nbexec:
    # Iterate over notebooks
    for fn in sorted(glob(os.path.join(npth, '*', '*.ipynb'))):
        if not docntbk.notebook_executed(fn):
            print('Executing %s    ' % fn, end='', flush=True)
            try:
                t = docntbk.execute_notebook(fn, fd)
            except Exception as ex:
                print('  execution error [%s]' % ex.__class__.__name__)
            else:
                print('  %.1f s' % t)
