#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate notebooks and documentation from python scripts."""


from __future__ import print_function

import os
import os.path
from pathlib import Path
from glob import glob
import re
import pickle
from timeit import default_timer as timer
import warnings

import py2jn
import pypandoc
import nbformat
from sphinx.ext import intersphinx
from nbconvert import RSTExporter
from nbconvert.preprocessors import ExecutePreprocessor



def mkdir(pth):
    """Make a directory if it doesn't exist."""

    if not os.path.exists(pth):
        os.mkdir(pth)



def pathsplit(pth, dropext=True):
    """Split a path into a tuple of all of its components."""

    if dropext:
        pth = os.path.splitext(pth)[0]
    parts = os.path.split(pth)
    if parts[0] == '':
        return parts[1:]
    elif len(parts[0]) == 1:
        return parts
    else:
        return pathsplit(parts[0], dropext=False) + parts[1:]



def update_required(srcpth, dstpth):
    """
    If the file at `dstpth` is generated from the file at `srcpth`,
    determine whether an update is required.  Returns True if `dstpth`
    does not exist, or if `srcpth` has been more recently modified
    than `dstpth`.
    """

    return not os.path.exists(dstpth) or \
        os.stat(srcpth).st_mtime > os.stat(dstpth).st_mtime



def fetch_intersphinx_inventory(uri):
    """
    Fetch and read an intersphinx inventory file at a specified uri,
    which can either be a url (e.g. http://...) or a local file system
    filename.
    """

    # See https://stackoverflow.com/a/30981554
    class MockConfig(object):
        intersphinx_cache_limit = None
        intersphinx_timeout = None
        tls_verify = False
        tls_cacerts = None
        user_agent = None

    class MockApp(object):
        srcdir = Path('')
        config = MockConfig()

        def warn(self, msg):
            warnings.warn(msg)

    return intersphinx.fetch_inventory(MockApp(), '', uri)



def read_sphinx_environment(pth):
    """Read the sphinx environment.pickle file at path `pth`."""

    with open(pth, 'rb') as fo:
        env = pickle.load(fo)
    return env



def parse_rst_index(rstpth):
    """
    Parse the top-level RST index file, at `rstpth`, for the example
    python scripts.  Returns a list of subdirectories in order of
    appearance in the index file, and a dict mapping subdirectory name
    to a description.
    """

    pthidx = {}
    pthlst = []
    with open(rstpth) as fd:
        lines = fd.readlines()
    for i, l in enumerate(lines):
        if i > 0:
            if re.match(r'^  \w+', l) is not None and \
               re.match(r'^\w+', lines[i - 1]) is not None:
                # List of subdirectories in order of appearance in index.rst
                pthlst.append(lines[i - 1][:-1])
                # Dict mapping subdirectory name to description
                pthidx[lines[i - 1][:-1]] = l[2:-1]
    return pthlst, pthidx



def preprocess_script_string(str):
    """
    Process python script represented as string `str` in preparation
    for conversion to a notebook.  This processing includes removal of
    the header comment, modification of the plotting configuration,
    and replacement of certain sphinx cross-references with
    appropriate links to online docs.
    """

    # Remove header comment
    str = re.sub(r'^(#[^#\n]+\n){5}\n*', r'', str)
    # Remove r from r""" ... """
    str = re.sub('^r"""', '"""', str, flags=re.MULTILINE)
    # Insert notebook plotting configuration function
    str = re.sub(r'from sporco import plot', r'from sporco import plot'
                 '\nplot.config_notebook_plotting()',
                 str, flags=re.MULTILINE)
    # Remove final input statement and preceding comment
    str = re.sub(r'\n*# Wait for enter on keyboard.*\ninput().*\n*',
                 r'', str, flags=re.MULTILINE)

    return str



def script_string_to_notebook_object(str):
    """
    Convert a python script represented as string `str` to a notebook
    object.
    """

    return py2jn.py_string_to_notebook(str, nbver=4)



def script_string_to_notebook(str, pth):
    """
    Convert a python script represented as string `str` to a notebook
    with filename `pth`.
    """

    nb = py2jn.py_string_to_notebook(str)
    py2jn.write_notebook(nb, pth)



def script_to_notebook(spth, npth, cr):
    """
    Convert the script at `spth` to a notebook at `npth`. Parameter `cr`
    is a CrossReferenceLookup object.
    """

    # Read entire text of example script
    with open(spth) as f:
        stxt = f.read()
    # Process script text
    stxt = preprocess_script_string(stxt)

    # If the notebook file exists and has been executed, try to
    # update markdown cells without deleting output cells
    if os.path.exists(npth) and notebook_executed(npth):
        # Read current notebook file
        nbold = nbformat.read(npth, as_version=4)
        # Construct updated notebook
        nbnew = script_string_to_notebook_object(stxt)
        if cr is not None:
            notebook_substitute_ref_with_url(nbnew, cr)
        # If the code cells of the two notebooks match, try to
        # update markdown cells without deleting output cells
        if same_notebook_code(nbnew, nbold):
            try:
                replace_markdown_cells(nbnew, nbold)
            except Exception:
                script_string_to_notebook_with_links(stxt, npth, cr)
            else:
                with open(npth, 'wt') as f:
                    nbformat.write(nbold, f)
        else:
            # Write changed text to output notebook file
            script_string_to_notebook_with_links(stxt, npth, cr)
    else:
        # Write changed text to output notebook file
        script_string_to_notebook_with_links(stxt, npth, cr)



def script_string_to_notebook_with_links(str, pth, cr=None):
    """
    Convert a python script represented as string `str` to a notebook
    with filename `pth` and replace sphinx cross-references with links
    to online docs. Parameter `cr` is a CrossReferenceLookup object.
    """

    if cr is None:
        script_string_to_notebook(str, pth)
    else:
        ntbk = script_string_to_notebook_object(str)
        notebook_substitute_ref_with_url(ntbk, cr)
        with open(pth, 'wt') as f:
            nbformat.write(ntbk, f)



def rst_to_notebook(infile, outfile, diridx=False):
    """Convert an rst file to a notebook file."""

    # Read infile into a string
    with open(infile, 'r') as fin:
        rststr = fin.read()
    # Convert string from rst to markdown
    mdfmt = 'markdown_github+tex_math_dollars+fenced_code_attributes'
    mdstr = pypandoc.convert_text(rststr, mdfmt, format='rst',
                                  extra_args=['--atx-headers'])
    # In links, replace .py extensions with .ipynb
    mdstr = re.sub(r'\(([^\)]+).py\)', r'(\1.ipynb)', mdstr)
    # Links to subdirectories require explicit index file inclusion
    if diridx:
        mdstr = re.sub(r']\(([^\)/]+)\)', r'](\1/index.ipynb)', mdstr)
    # Enclose the markdown within triple quotes and convert from
    # python to notebook
    mdstr = '"""' + mdstr + '"""'
    nb = py2jn.py_string_to_notebook(mdstr)
    py2jn.tools.write_notebook(nb, outfile, nbver=4)



def markdown_to_notebook(infile, outfile):
    """Convert a markdown file to a notebook file."""

    # Read infile into a string
    with open(infile, 'r') as fin:
        str = fin.read()
    # Enclose the markdown within triple quotes and convert from
    # python to notebook
    str = '"""' + str + '"""'
    nb = py2jn.py_string_to_notebook(str)
    py2jn.tools.write_notebook(nb, outfile, nbver=4)



def rst_to_docs_rst(infile, outfile):
    """Convert an rst file to a sphinx docs rst file."""

    # Read infile into a list of lines
    with open(infile, 'r') as fin:
        rst = fin.readlines()

    # Inspect outfile path components to determine whether outfile
    # is in the root of the examples directory or in a subdirectory
    # thererof
    ps = pathsplit(outfile)[-3:]
    if ps[-2] == 'examples':
        ps = ps[-2:]
        idx = 'index'
    else:
        idx = ''

    # Output string starts with a cross-reference anchor constructed from
    # the file name and path
    out = '.. _' + '_'.join(ps) + ':\n\n'

    # Iterate over lines from infile
    it = iter(rst)
    for line in it:
        if line[0:12] == '.. toc-start':  # Line has start of toc marker
            # Initialise current toc array and iterate over lines until
            # end of toc marker encountered
            toc = []
            for line in it:
                if line == '\n':  # Drop newline lines
                    continue
                elif line[0:10] == '.. toc-end':  # End of toc marker
                    # Add toctree section to output string
                    out += '.. toctree::\n   :maxdepth: 1\n\n'
                    for c in toc:
                        out += '   %s <%s>\n' % c
                    break
                else:  #  Still within toc section
                    # Extract link text and target url and append to
                    # toc array
                    m = re.search(r'`(.*?)\s*<(.*?)(?:.py)?>`', line)
                    if m:
                        if idx == '':
                            toc.append((m.group(1), m.group(2)))
                        else:
                            toc.append((m.group(1),
                                        os.path.join(m.group(2), idx)))
        else:  # Not within toc section
            out += line

    with open(outfile, 'w') as fout:
        fout.write(out)



def parse_notebook_index(ntbkpth):
    """
    Parse the top-level notebook index file at `ntbkpth`.  Returns a
    list of subdirectories in order of appearance in the index file,
    and a dict mapping subdirectory name to a description.
    """

    # Convert notebook to RST text in string
    rex = RSTExporter()
    rsttxt = rex.from_filename(ntbkpth)[0]
    # Clean up trailing whitespace
    rsttxt = re.sub(r'\n  ', r'', rsttxt, re.M | re.S)
    pthidx = {}
    pthlst = []
    lines = rsttxt.split('\n')
    for l in lines:
        m = re.match(r'^-\s+`([^<]+)\s+<([^>]+).ipynb>`__', l)
        if m:
            # List of subdirectories in order of appearance in index.rst
            pthlst.append(m.group(2))
            # Dict mapping subdirectory name to description
            pthidx[m.group(2)] = m.group(1)
    return pthlst, pthidx



def construct_notebook_index(title, pthlst, pthidx):
    """
    Construct a string containing a markdown format index for the list
    of paths in `pthlst`.  The title for the index is in `title`, and
    `pthidx` is a dict giving label text for each path.
    """

    # Insert title text
    txt = '"""\n## %s\n"""\n\n"""' % title
    # Insert entry for each item in pthlst
    for pth in pthlst:
        # If pth refers to a .py file, replace .py with .ipynb, otherwise
        # assume it's a directory name and append '/index.ipynb'
        if pth[-3:] == '.py':
            link = os.path.splitext(pth)[0] + '.ipynb'
        else:
            link = os.path.join(pth, 'index.ipynb')
        txt += '- [%s](%s)\n' % (pthidx[pth], link)
    txt += '"""'
    return txt



def notebook_executed(pth):
    """Determine whether the notebook at `pth` has been executed."""

    try:
        nb = nbformat.read(pth, as_version=4)
    except (AttributeError, nbformat.reader.NotJSONError):
        raise RuntimeError('Error reading notebook file %s' % pth)
    for n in range(len(nb['cells'])):
        if nb['cells'][n].cell_type == 'code' and \
                nb['cells'][n].execution_count is None:
            return False
    return True



def same_notebook_code(nb1, nb2):
    """
    Return true of the code cells of notebook objects `nb1` and `nb2`
    are the same.
    """

    # Notebooks do not match of the number of cells differ
    if len(nb1['cells']) != len(nb2['cells']):
        return False

    # Iterate over cells in nb1
    for n in range(len(nb1['cells'])):
        # Notebooks do not match if corresponding cells have different
        # types
        if nb1['cells'][n]['cell_type'] != nb2['cells'][n]['cell_type']:
            return False
        # Notebooks do not match if source of corresponding code cells
        # differ
        if nb1['cells'][n]['cell_type'] == 'code' and \
                nb1['cells'][n]['source'] != nb2['cells'][n]['source']:
            return False

    return True



def execute_notebook(npth, dpth, timeout=1800, kernel='python3'):
    """
    Execute the notebook at `npth` using `dpth` as the execution
    directory.  The execution timeout and kernel are `timeout` and
    `kernel` respectively.
    """

    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel)
    nb = nbformat.read(npth, as_version=4)
    t0 = timer()
    ep.preprocess(nb, {'metadata': {'path': dpth}})
    t1 = timer()
    with open(npth, 'wt') as f:
        nbformat.write(nb, f)
    return t1 - t0



def replace_markdown_cells(src, dst):
    """
    Overwrite markdown cells in notebook object `dst` with corresponding
    cells in notebook object `src`.
    """

    # It is an error to attempt markdown replacement if src and dst
    # have different numbers of cells
    if len(src['cells']) != len(dst['cells']):
        raise ValueError('notebooks do not have the same number of cells')

    # Iterate over cells in src
    for n in range(len(src['cells'])):
        # It is an error to attempt markdown replacement if any
        # corresponding pair of cells have different type
        if src['cells'][n]['cell_type'] != dst['cells'][n]['cell_type']:
            raise ValueError('cell number %d of different type in src and dst')
        # If current src cell is a markdown cell, copy the src cell to
        # the dst cell
        if src['cells'][n]['cell_type'] == 'markdown':
            dst['cells'][n]['source'] = src['cells'][n]['source']



def notebook_substitute_ref_with_url(ntbk, cr):
    """
    In markdown cells of notebook object `ntbk`, replace sphinx
    cross-references with links to online docs. Parameter `cr` is a
    CrossReferenceLookup object.
    """

    # Iterate over cells in notebook
    for n in range(len(ntbk['cells'])):
        # Only process cells of type 'markdown'
        if ntbk['cells'][n]['cell_type'] == 'markdown':
            # Get text of markdown cell
            txt = ntbk['cells'][n]['source']
            # Replace links to online docs with sphinx cross-references
            txt = cr.substitute_ref_with_url(txt)
            # Replace current cell text with processed text
            ntbk['cells'][n]['source'] = txt



def preprocess_notebook(ntbk, cr):
    """
    Process notebook object `ntbk` in preparation for conversion to an
    rst document.  This processing replaces links to online docs with
    corresponding sphinx cross-references within the local docs.
    Parameter `cr` is a CrossReferenceLookup object.
    """

    # Iterate over cells in notebook
    for n in range(len(ntbk['cells'])):
        # Only process cells of type 'markdown'
        if ntbk['cells'][n]['cell_type'] == 'markdown':
            # Get text of markdown cell
            txt = ntbk['cells'][n]['source']
            # Replace links to online docs with sphinx cross-references
            txt = cr.substitute_url_with_ref(txt)
            # Replace current cell text with processed text
            ntbk['cells'][n]['source'] = txt



def write_notebook_rst(txt, res, fnm, pth):
    """
    Write the converted notebook text `txt` and resources `res` to
    filename `fnm` in directory `pth`.
    """

    # Extended filename used for output images
    extfnm = fnm + '_files'
    # Directory into which output images are written
    extpth = os.path.join(pth, extfnm)
    # Make output image directory if it doesn't exist
    mkdir(extpth)
    # Iterate over output images in resources dict
    for r in res['outputs'].keys():
        # New name for current output image
        rnew = re.sub('output', fnm, r)
        # Partial path for current output image
        rpth = os.path.join(extfnm, rnew)
        # In RST text, replace old output name with the new one
        txt = re.sub(r'\.\. image:: ' + r, '.. image:: ' + rpth, txt, re.M)
        # Full path of the current output image
        fullrpth = os.path.join(pth, rpth)
        # Write the current output image to disk
        with open(fullrpth, 'wb') as fo:
            fo.write(res['outputs'][r])

    # Remove trailing whitespace in RST text
    txt = re.sub(r'[ \t]+$', '', txt, flags=re.M)

    # Write RST text to disk
    with open(os.path.join(pth, fnm + '.rst'), 'wt') as fo:
        fo.write(txt)



def notebook_to_rst(npth, rpth, rdir, cr=None):
    """
    Convert notebook at `npth` to rst document at `rpth`, in directory
    `rdir`. Parameter `cr` is a CrossReferenceLookup object.
    """

    # Read the notebook file
    ntbk = nbformat.read(npth, nbformat.NO_CONVERT)
    # Convert notebook object to rstpth
    notebook_object_to_rst(ntbk, rpth, rdir, cr)



def notebook_object_to_rst(ntbk, rpth, cr=None):
    """
    Convert notebook object `ntbk` to rst document at `rpth`, in
    directory `rdir`.  Parameter `cr` is a CrossReferenceLookup
    object.
    """

    # Parent directory of file rpth
    rdir = os.path.dirname(rpth)
    # File basename
    rb = os.path.basename(os.path.splitext(rpth)[0])

    # Pre-process notebook prior to conversion to rst
    if cr is not None:
        preprocess_notebook(ntbk, cr)
    # Convert notebook to rst
    rex = RSTExporter()
    rsttxt, rstres = rex.from_notebook_node(ntbk)
    # Replace `` with ` in sphinx cross-references
    rsttxt = re.sub(r':([^:]+):``(.*?)``', r':\1:`\2`', rsttxt)
    # Insert a cross-reference target at top of file
    reflbl = '.. _examples_' + os.path.basename(rdir) + '_' + \
             rb.replace('-', '_') + ':\n\n'
    rsttxt = reflbl + rsttxt
    # Write the converted rst to disk
    write_notebook_rst(rsttxt, rstres, rb, rdir)



def script_and_notebook_to_rst(spth, npth, rpth):
    """
    Convert a script and the corresponding executed notebook to rst.
    The script is converted to notebook format *without* replacement
    of sphinx cross-references with links to online docs, and the
    resulting markdown cells are inserted into the executed notebook,
    which is then converted to rst.
    """

    # Read entire text of script at spth
    with open(spth) as f:
        stxt = f.read()
    # Process script text
    stxt = preprocess_script_string(stxt)
    # Convert script text to notebook object
    nbs = script_string_to_notebook_object(stxt)

    # Read notebook file npth
    nbn = nbformat.read(npth, as_version=4)

    # Overwrite markdown cells in nbn with those from nbs
    try:
        replace_markdown_cells(nbs, nbn)
    except ValueError:
        raise ValueError('mismatch between source script %s and notebook %s' %
                         (spth, npth))

    # Convert notebook object to rst
    notebook_object_to_rst(nbn, rpth)




class IntersphinxInventory(object):
    """
    Class supporting look up of relevant information from an intersphinx
    inventory dict.
    """

    domainrole = {'py:module': 'mod', 'py:function': 'func',
                  'py:data': 'data', 'py:class': 'class',
                  'py:method': 'meth', 'py:attribute': 'attr',
                  'py:exception': 'exc'}
    """Dict providing lookup of sphinx role labels from domain labels"""

    roledomain = {r: d for d, r in domainrole.items()}
    """Dict providing lookup of sphinx domain labels from role labels"""


    def __init__(self, inv, baseurl, addbase=False):
        """
        Parameter are:
        `inv` : an intersphinx inventory dict
        `baseurl` : the base url for the objects in this inventory
        `addbase` : flag indicating whether it is necessary to append
                the base url onto the entries in the inventory
        """

        self.inv = inv
        self.baseurl = baseurl
        self.addbase = addbase
        # Initialise dicts used for reverse lookup and partial name lookup
        self.revinv, self.rolnam = IntersphinxInventory.inventory_maps(inv)



    def get_label_from_name(self, name):
        """
        Convert a sphinx reference name (or partial name) into a link
        label.
        """

        if name[0] == '.':
            return name[1:]
        else:
            return name



    def get_full_name(self, role, name):
        """
        If ``name`` is already the full name of an object, return
        ``name``.  Otherwise, if ``name`` is a partial object name,
        look up the full name and return it.
        """

        # An initial '.' indicates a partial name
        if name[0] == '.':
            # Find matches for the partial name in the string
            # containing all full names for this role
            ptrn = r'(?<= )[^,]*' + name + r'(?=,)'
            ml = re.findall(ptrn, self.rolnam[role])
            # Handle cases depending on the number of returned matches,
            # raising an error if exactly one match is not found
            if len(ml) == 0:
                raise KeyError('name matching %s not found' % name,
                               'name', len(ml))
            elif len(ml) > 1:
                raise KeyError('multiple names matching %s found' % name,
                               'name', len(ml))
            else:
                return ml[0]
        else:
            # The absence of an initial '.' indicates a full
            # name. Return the name if it is present in the inventory,
            # otherwise raise an error
            try:
                dom = IntersphinxInventory.roledomain[role]
            except KeyError:
                raise KeyError('role %s not found' % role, 'role', 0)
            if name in self.inv[dom]:
                return name
            else:
                raise KeyError('name %s not found' % name, 'name', 0)



    def get_docs_url(self, role, name):
        """
        Get a url for the online docs corresponding to a sphinx cross
        reference :role:`name`.
        """

        # Expand partial names to full names
        name = self.get_full_name(role, name)
        # Look up domain corresponding to role
        dom = IntersphinxInventory.roledomain[role]
        # Get the inventory entry tuple corresponding to the name
        # of the referenced type
        itpl = self.inv[dom][name]
        # Get the required path postfix from the inventory entry
        # tuple
        path = itpl[2]
        # Construct link url, appending the base url or note
        # depending on the addbase flag
        return self.baseurl + path if self.addbase else path



    def matching_base_url(self, url):
        """
        Return True if the initial part of `url` matches the base url
        passed to the initialiser of this object, and False otherwise.
        """

        n = len(self.baseurl)
        return url[0:n] == self.baseurl



    def get_sphinx_ref(self, url, label=None):
        """
        Get an internal sphinx cross reference corresponding to `url`
        into the online docs, associated with a link with label `label`
        (if not None).
        """

        # Raise an exception if the initial part of url does not match
        # the base url for this object
        n = len(self.baseurl)
        if url[0:n] != self.baseurl:
            raise KeyError('base of url %s does not match base url %s' %
                           (url, self.baseurl))
        # The reverse lookup key is either the full url or the postfix
        # to the base url, depending on flag addbase
        if self.addbase:
            pstfx = url[n:]
        else:
            pstfx = url

        # Look up the cross-reference role and referenced object
        # name via the postfix to the base url
        role, name = self.revinv[pstfx]

        # If the label string is provided and is shorter than the name
        # string we have lookup up, assume it is a partial name for
        # the same object: append a '.' at the front and use it as the
        # object name in the cross-reference
        if label is not None and len(label) < len(name):
            name = '.' + label

        # Construct cross-reference
        ref = ':%s:`%s`' % (role, name)
        return ref



    @staticmethod
    def inventory_maps(inv):
        """
        Construct dicts facilitating information lookup in an
        inventory dict. A reversed dict allows lookup of a tuple
        specifying the sphinx cross-reference role and the name of the
        referenced type from the intersphinx inventory url postfix
        string. A role-specific name lookup string allows the set of all
        names corresponding to a specific role to be searched via regex.
        """

        # Initialise dicts
        revinv = {}
        rolnam = {}
        # Iterate over domain keys in inventory dict
        for d in inv:
            # Since keys seem to be duplicated, ignore those not
            # starting with 'py:'
            if d[0:3] == 'py:' and d in IntersphinxInventory.domainrole:
                # Get role corresponding to current domain
                r = IntersphinxInventory.domainrole[d]
                # Initialise role-specific name lookup string
                rolnam[r] = ''
                # Iterate over all type names for current domain
                for n in inv[d]:
                    # Get the url postfix string for the current
                    # domain and type name
                    p = inv[d][n][2]
                    # Allow lookup of role and object name tuple from
                    # url postfix
                    revinv[p] = (r, n)
                    # Append object name to a string for this role,
                    # allowing regex searching for partial names
                    rolnam[r] += ' ' + n + ','
        return revinv, rolnam




class CrossReferenceLookup(object):
    """
    Class supporting cross reference lookup for citations and all
    document sets recorded by intersphinx.
    """

    def __init__(self, env, inv, baseurl):
        """
        Parameter are:
        `env` : a sphinx environment object
        `inv` : an intersphinx inventory dict
        `baseurl` : the base url for the objects in this inventory
        """

        self.baseurl = baseurl
        # Construct a list of IntersphinxInventory objects. The first
        # entry in the list is for the intersphinx inventory for the
        # package for which we are building sphinx docs
        self.invlst = [IntersphinxInventory(inv, baseurl, addbase=True),]

        self.env = env
        # Add additional entries to the list for each external package
        # docs set included by intersphinx
        for b in env.intersphinx_cache:
            self.invlst.append(IntersphinxInventory(
                env.intersphinx_cache[b][2], b))

        # Recent versions of sphinx environment do not have a
        # bibtex_cache attribute. In this case, extract citation data
        # from env.domaindata
        self.citenum = {}
        self.citeid = {}
        if not hasattr(env, 'bibtex_cache'):
            for cite in env.domaindata['cite']['citations']:
                self.citenum[cite.key] = cite.citation_id[2:]
                self.citeid[cite.key] = cite.citation_id


    def get_docs_url(self, role, name):
        """
        Get the online docs url for sphinx cross-reference :role:`name`.
        """

        if role == 'cite':
            # If the cross-reference is a citation, make sure that
            # the cite key is in the sphinx environment bibtex cache.
            # If it is, construct the url from the cite key, otherwise
            # raise an exception
            if hasattr(self.env, 'bibtex_cache'):
                id = name
                if name not in self.env.bibtex_cache.get_all_cited_keys():
                    raise KeyError('cite key %s not found' % name, 'cite', 0)
            else:
                id = self.citeid[name]
            url = self.baseurl + 'zreferences.html#' + id
        elif role == 'ref':
            try:
                reftpl = self.env.domaindata['std']['labels'][name]
            except Exception:
                raise KeyError('ref label %s not found' % name, 'ref', 0)
            url = self.baseurl + reftpl[0] + '.html#' + reftpl[1]
        else:
            # If the  cross-reference is not a citation, try to look it
            # up in each of the IntersphinxInventory objects in our list
            url = None
            for ii in self.invlst:
                try:
                    url = ii.get_docs_url(role, name)
                except KeyError as ex:
                    # Re-raise the exception if multiple matches found,
                    # otherwise ignore it
                    if ex.args[1] == 'role' or ex.args[2] > 1:
                        raise ex
                else:
                    # If an exception was not raised, the lookup must
                    # have succeeded: break from the loop to terminate
                    # further searching
                    break

            if url is None:
                raise KeyError('name %s not found' % name, 'name', 0)

        return url



    def get_docs_label(self, role, name):
        """Get an appropriate label to use in a link to the online docs."""

        if role == 'cite':
            # Get the string used as the citation label in the text
            if hasattr(self.env, 'bibtex_cache'):
                try:
                    cstr = self.env.bibtex_cache.get_label_from_key(name)
                except Exception:
                    raise KeyError('cite key %s not found' % name, 'cite', 0)
            else:
                try:
                    cstr = self.citenum[name]
                except KeyError:
                    raise KeyError('cite key %s not found' % name, 'cite', 0)
            # The link label is the citation label (number) enclosed
            # in square brackets
            return '[%s]' % cstr
        elif role == 'ref':
            try:
                reftpl = self.env.domaindata['std']['labels'][name]
            except Exception:
                raise KeyError('ref label %s not found' % name, 'ref', 0)
            return reftpl[2]
        else:
            # Use the object name as a label, omiting any initial '.'
            if name[0] == '.':
                return name[1:]
            else:
                return name



    def get_sphinx_ref(self, url, label=None):
        """
        Get an internal sphinx cross reference corresponding to `url`
        into the online docs, associated with a link with label `label`
        (if not None).
        """

        # A url is assumed to correspond to a citation if it contains
        # 'zreferences.html#'
        if 'zreferences.html#' in url:
            key = url.partition('zreferences.html#')[2]
            ref = ':cite:`%s`' % key
        else:
            # If the url does not correspond to a citation, try to look it
            # up in each of the IntersphinxInventory objects in our list
            ref = None
            # Iterate over IntersphinxInventory objects in our list
            for ii in self.invlst:
                # If the baseurl for the current IntersphinxInventory
                # object matches the url, try to look up the reference
                # from the url and terminate the loop of the look up
                # succeeds
                if ii.matching_base_url(url):
                    ref = ii.get_sphinx_ref(url, label)
                    break

            if ref is None:
                raise KeyError('no match found for url %s' % url)

        return ref



    def substitute_ref_with_url(self, txt):
        """
        In the string `txt`, replace sphinx references with
        corresponding links to online docs.
        """

        # Find sphinx cross-references
        mi = re.finditer(r':([^:]+):`([^`]+)`', txt)
        if mi:
            # Iterate over match objects in iterator returned by re.finditer
            for mo in mi:
                # Initialize link label and url for substitution
                lbl = None
                url = None
                # Get components of current match: full matching text, the
                # role label in the reference, and the name of the
                # referenced type
                mtxt = mo.group(0)
                role = mo.group(1)
                name = mo.group(2)

                # If role is 'ref', the name component is in the form
                # label <name>
                if role == 'ref':
                    ma = re.match(r'\s*([^\s<]+)\s*<([^>]+)+>', name)
                    if ma:
                        name = ma.group(2)
                        lbl = ma.group(1)

                # Try to look up the current cross-reference. Issue a
                # warning if the lookup fails, and do the substitution
                # if it succeeds.
                try:
                    url = self.get_docs_url(role, name)
                    if role != 'ref':
                        lbl = self.get_docs_label(role, name)
                except KeyError as ex:
                    if len(ex.args) == 1 or ex.args[1] != 'role':
                        print('Warning: %s' % ex.args[0])
                else:
                    # If the cross-reference lookup was successful, replace
                    # it with an appropriate link to the online docs
                    rtxt = '[%s](%s)' % (lbl, url)
                    txt = re.sub(mtxt, rtxt, txt, flags=re.M)

        return txt



    def substitute_url_with_ref(self, txt):
        """
        In the string `txt`, replace links to online docs with
        corresponding sphinx cross-references.
        """

        # Find links
        mi = re.finditer(r'\[([^\]]+|\[[^\]]+\])\]\(([^\)]+)\)', txt)
        if mi:
            # Iterate over match objects in iterator returned by
            # re.finditer
            for mo in mi:
                # Get components of current match: full matching text,
                # the link label, and the postfix to the base url in the
                # link url
                mtxt = mo.group(0)
                lbl = mo.group(1)
                url = mo.group(2)

                # Try to look up the current link url. Issue a warning if
                # the lookup fails, and do the substitution if it succeeds.
                try:
                    ref = self.get_sphinx_ref(url, lbl)
                except KeyError as ex:
                    print('Warning: %s' % ex.args[0])
                else:
                    txt = re.sub(re.escape(mtxt), ref, txt)

        return txt





def make_example_scripts_docs(spth, npth, rpth):
    """
    Generate rst docs from example scripts.  Arguments `spth`, `npth`,
    and `rpth` are the top-level scripts directory, the top-level
    notebooks directory, and the top-level output directory within the
    docs respectively.
    """

    # Ensure that output directory exists
    mkdir(rpth)

    # Iterate over index files
    for fp in glob(os.path.join(spth, '*.rst')) + \
              glob(os.path.join(spth, '*', '*.rst')):
        # Index basename
        b = os.path.basename(fp)
        # Index dirname
        dn = os.path.dirname(fp)
        # Name of subdirectory of examples directory containing current index
        sd = os.path.split(dn)
        # Set d to the name of the subdirectory of the root directory
        if dn == spth:  # fp is the root directory index file
            d = ''
        else:           # fp is a subdirectory index file
            d = sd[-1]
        # Path to corresponding subdirectory in docs directory
        fd = os.path.join(rpth, d)
        # Ensure notebook subdirectory exists
        mkdir(fd)
        # Filename of index file to be constructed
        fn = os.path.join(fd, b)
        # Process current index file if corresponding notebook file
        # doesn't exist, or is older than index file
        if update_required(fp, fn):
            print('Converting %s                ' % os.path.join(d, b),
                  end='\r')
            # Convert script index to docs index
            rst_to_docs_rst(fp, fn)

    # Iterate over example scripts
    for fp in sorted(glob(os.path.join(spth, '*', '*.py'))):
        # Name of subdirectory of examples directory containing current script
        d = os.path.split(os.path.dirname(fp))[1]
        # Script basename
        b = os.path.splitext(os.path.basename(fp))[0]
        # Path to corresponding notebook
        fn = os.path.join(npth, d, b + '.ipynb')
        # Path to corresponding sphinx doc file
        fr = os.path.join(rpth, d, b + '.rst')
        # Only proceed if script and notebook exist
        if os.path.exists(fp) and os.path.exists(fn):
            # Convert notebook to rst if notebook is newer than rst
            # file or if rst file doesn't exist
            if update_required(fn, fr):
                fnb = os.path.join(d, b + '.ipynb')
                print('Processing %s                ' % fnb, end='\r')
                script_and_notebook_to_rst(fp, fn, fr)
        else:
            print('WARNING: script %s or notebook %s not found' %
                  (fp, fn))
