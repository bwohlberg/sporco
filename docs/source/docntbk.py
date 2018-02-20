#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import os.path
import tempfile
import re
import pickle
from timeit import default_timer as timer

import py2nb.tools
import nbformat
from sphinx.ext import intersphinx
from nbconvert import RSTExporter
from nbconvert.preprocessors import ExecutePreprocessor



def mkdir(pth):
    """
    Make a directory if it doesn't exist.
    """

    if not os.path.exists(pth):
        os.mkdir(pth)



def update_required(srcpth, dstpth):
    """
    If the file at `dstpth` is generated from the file at `srcpth`, determine
    whether an update is required. Returns True if `dstpth` does not exist,
    or if `srcpth` has been more recently modified than `dstpth`.
    """

    return not os.path.exists(dstpth) or \
      os.stat(srcpth).st_mtime > os.stat(dstpth).st_mtime



def fetch_intersphinx_inventory(uri):
    """
    Fetch and read an intersphinx inventory file at a specified uri, which
    can either be a url (e.g. http://...) or a local file system filename.
    """

    # See https://stackoverflow.com/a/30981554
    class MockConfig(object):
        intersphinx_timeout = None
        tls_verify = False

    class MockApp(object):
        srcdir = ''
        config = MockConfig()

        def warn(self, msg):
            warnings.warn(msg)

    return intersphinx.fetch_inventory(MockApp(), '', uri)



def read_sphinx_environment(pth):
    """
    Read the sphinx environment.pickle file at path `pth`.
    """

    with open(pth, 'rb') as fo:
        env = pickle.load(fo)
    return env





def parse_rst_index(rstpth):
    """
    Parse the top-level RST index file, at `rstpth`, for the example
    python scripts. Returns a list of subdirectories in order of appearance
    in the index file, and a dict mapping subdirectory name to a description.
    """

    pthidx = {}
    pthlst = []
    with open(rstpth) as fd:
        lines = fd.readlines()
    for i, l in enumerate(lines):
        if i > 0:
            if re.match(r'^  \w+', l) is not None and \
                re.match(r'^\w+', lines[i-1]) is not None:
                # List of subdirectories in order of appearance in index.rst
                pthlst.append(lines[i-1][:-1])
                # Dict mapping subdirectory name to description
                pthidx[lines[i-1][:-1]] = l[2:-1]
    return pthlst, pthidx





def preprocess_script_string(str):
    """
    Process python script represented as string `str` in preparation for
    conversion to a notebook. This processing includes removal of the header
    comment, modification of the plotting configuration, and replacement of
    certain sphinx cross-references with appropriate links to online docs.
    """

    # Remove header comment
    str = re.sub(r'^(#[^#\n]+\n){5}\n*', r'', str)
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

    with tempfile.NamedTemporaryFile('r+t') as sf:
        sf.write(str)
        sf.flush()
        with tempfile.NamedTemporaryFile('r+t') as nf:
            py2nb.tools.python_to_notebook(sf.name, nf.name)
            nb = nbformat.read(nf.name, as_version=4)

    return nb



def script_string_to_notebook(str, pth):
    """
    Convert a python script represented as string `str` to a notebook
    with filename `pth`.
    """

    with tempfile.NamedTemporaryFile('r+t') as f:
        f.write(str)
        f.flush()
        py2nb.tools.python_to_notebook(f.name, pth)



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
            except:
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





def parse_notebook_index(ntbkpth):
    """
    Parse the top-level notebook index file at `ntbkpth`. Returns a list of
    subdirectories in order of appearance in the index file, and a dict
    mapping subdirectory name to a description.
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
    Construct a string containing a markdown format index for the list of
    paths in `pthlst`. The title for the index is in `title`, and `pthidx`
    is a dict giving label text for each path.
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
    """
    Determine whether the notebook at `pth` has been executed.
    """

    nb = nbformat.read(pth, as_version=4)
    for n in range(len(nb['cells'])):
        if nb['cells'][n].cell_type == 'code' and \
                nb['cells'][n].execution_count is None:
            return False
    return True



def same_notebook_code(nb1, nb2):
    """
    Return true of the code cells of notebook objects `nb1` and `nb2` are
    the same.
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



def execute_notebook(npth, dpth, timeout=600, kernel='python3'):
    """
    Execute the notebook at `npth` using `dpth` as the execution directory.
    The execution timeout and kernel are `timeout` and `kernel`
    respectively.
    """

    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel)
    nb = nbformat.read(npth, as_version=4)
    t0 = timer()
    ep.preprocess(nb, {'metadata': {'path': dpth}})
    t1 = timer()
    with open(npth, 'wt') as f:
        nbformat.write(nb, f)
    return t1-t0



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
        # If current src cell is a markdown cell, copy the src cell to the
        # dst cell
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
    Process notebook object `ntbk` in preparation for conversion to an rst
    document. This processing replaces links to online docs with
    corresponding sphinx cross-references within the local docs. Parameter
    `cr` is a CrossReferenceLookup object.
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
    Write the converted notebook text `txt` and resources `res` to filename
    `fnm` in directory `pth`.
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
        txt = re.sub('\.\. image:: ' + r, '.. image:: ' + rpth, txt, re.M)
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



def notebook_object_to_rst(ntbk, rpth, rdir, cr=None):
    """
    Convert notebook object `ntbk` to rst document at `rpth`, in directory
    `rdir`. Parameter `cr` is a CrossReferenceLookup object.
    """

    # Pre-process notebook prior to conversion to rst
    if cr is not None:
        preprocess_notebook(ntbk, cr)
    # Convert notebook to rst
    rex = RSTExporter()
    rsttxt, rstres = rex.from_notebook_node(ntbk)
    # Replace `` with ` in sphinx cross-references
    rsttxt = re.sub(r':([^:]+):``(.*?)``', r':\1:`\2`', rsttxt)
    # Insert a cross-reference target at top of file
    reflbl = '.. _example_' + os.path.basename(rdir) + '_' + \
            rpth.replace('-', '_') + ':\n'
    rsttxt = reflbl + rsttxt
    # Write the converted rst to disk
    write_notebook_rst(rsttxt, rstres, rpth, rdir)



def script_and_notebook_to_rst(spth, npth, rpth, rdir):
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
    replace_markdown_cells(nbs, nbn)

    # Convert notebook object to rst
    notebook_object_to_rst(nbn, rpth, rdir)




class IntersphinxInventory(object):
    """
    Class supporting look up of relevant information from an intersphinx
    inventory dict.
    """

    domainrole = {'py:module': 'mod', 'py:function': 'func',
            'py:data': 'data', 'py:class': 'class', 'py:method': 'meth',
            'py:attribute': 'attr', 'py:exception': 'exc'}
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
        Convert a sphinx reference name (or partial name) into a link label.
        """

        if name[0] == '.':
            return name[1:]
        else:
            return name



    def get_full_name(self, role, name):
        """
        If ``name`` is already the full name of an object, return ``name``.
        Otherwise, if ``name`` is a partial object name, look up the full
        name and return it.
        """

        # An initial '.' indicates a partial name
        if name[0] == '.':
            # Find matches for the partial name in the string containing all
            # full names for this role
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
            # The absence of an initial '.' indicates a full name. Return
            # the name if it is present in the inventory, otherwise raise
            # an error
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
        # string we have lookup up, assume it is a partial name for the
        # same object: append a '.' at the front and use it as the object
        # name in the cross-reference
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
            # Since keys seem to be duplicated, ignore those not starting
            # with 'py:'
            if d[0:3] == 'py:' and d in IntersphinxInventory.domainrole:
                # Get role corresponding to current domain
                r = IntersphinxInventory.domainrole[d]
                # Initialise role-specific name lookup string
                rolnam[r] = ''
                # Iterate over all type names for current domain
                for n in inv[d]:
                    # Get the url postfix string for the current domain and
                    # type name
                    p = inv[d][n][2]
                    # Allow lookup of role and object name tuple from url
                    # postfix
                    revinv[p] = (r, n)
                    # Append object name to a string for this role, allowing
                    # regex searching for partial names
                    rolnam[r] += ' ' + n + ','
        return revinv, rolnam




class CrossReferenceLookup(object):
    """
    Class supporting cross reference lookup for citations and all document
    sets recorded by intersphinx.
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
        # Add additional entries to the list for each external package
        # docs set included by intersphinx
        for b in env.intersphinx_cache:
            self.invlst.append(IntersphinxInventory(
                env.intersphinx_cache[b][2], b))

        self.btxcch = env.bibtex_cache



    def get_docs_url(self, role, name):
        """
        Get the online docs url for sphinx cross-reference :role:`name`.
        """

        if role == 'cite':
            # If the cross-reference is a citation, make sure that
            # the cite key is in the sphinx environment bibtex cache.
            # If it is, construct the url from the cite key, otherwise
            # raise an exception
            if name not in self.btxcch.get_all_cited_keys():
                raise KeyError('cite key %s not found' % name, 'cite', 0)
            url = self.baseurl + 'zreferences.html#' + name
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
                    # If an exception was not raised, the lookup must have
                    # succeeded: break from the loop to terminate further
                    # searching
                    break

            if url is None:
                raise KeyError('name %s not found' % name, 'name', 0)

        return url



    def get_docs_label(self, role, name):
        """
        Get an appropriate label to use in a link to the online docs.
        """

        if role == 'cite':
            # Get the string used as the citation label in the text
            cstr = self.btxcch.get_label_from_key(name)
            # The link label is the citation label (number) enclosed
            # in square brackets
            return '[%s]' % cstr
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
        In the string `txt`, replace sphinx references with corresponding
        links to online docs.
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

                # Try to look up the current cross-reference. Issue a
                # warning if the lookup fails, and do the substitution if
                # it succeeds.
                try:
                    url = self.get_docs_url(role, name)
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
        In the string `txt`, replace links to online docs with corresponding
        sphinx cross-references.
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

                # Try to look up the current link url. Issue a warning if the
                # lookup fails, and do the substitution if it succeeds.
                try:
                    ref = self.get_sphinx_ref(url, lbl)
                except KeyError as ex:
                    print('Warning: %s' % ex.args[0])
                else:
                    txt = re.sub(re.escape(mtxt), ref, txt)

        return txt





def make_example_scripts_docs(spth, npth, rpth):
    """
    Generate rst docs from example scripts. Arguments `spth`, `npth`, and
    `rpth` are the top-level scripts directory, the top-level notebooks
    directory, and the top-level output directory within the docs
    respectively.
    """

    # Ensure that output directory exists
    mkdir(rpth)

    # Top-level index files
    nfn = os.path.join(npth, 'index.ipynb')
    rfn = os.path.join(rpth, 'index.rst')

    # Parse the top-level notebook index file to extract a list of index files
    # in subdirectories
    pthlst, pthidx = parse_notebook_index(nfn)

    # Strip index file names from index file list to obtain a list of
    # subdirectories in which index files are present
    dirlst = list(map(os.path.dirname, pthlst))

    # Construct a dict mapping subdirectory names to parsed index files in
    # that subdirectory
    diridx = {}
    for d in dirlst:
        diridx[d] = parse_notebook_index(os.path.join(npth, d, 'index.ipynb'))

    # Construct top-level rst index
    if update_required(nfn, rfn):
        with open(rfn, 'wt') as fo:
            print('Usage Examples\n--------------\n', file=fo)
            print('.. toctree::\n   :maxdepth: 1\n', file=fo)
            for p in pthlst:
                print('   %s' % p, file=fo)
            print('\n.. toctree::\n   :hidden:\n', file=fo)
            for d in dirlst:
                print('   %s/index' % d, file=fo)
                for p in diridx[d][0]:
                    print('   %s/%s' % (d, p), file=fo)

    # Iterate over notebook subdirectories
    for d in dirlst:
        # Construct path to corresponding rst subdirectory
        rdir = os.path.join(rpth, d)
        # Get list of notebooks and notebook name to description
        # mapping for current subdirectory
        nlst, nidx = diridx[d]
        # Make corresponding rst subdirectory if it doesn't exist
        mkdir(rdir)

        # Write rst index for current subdirectory
        nifn = os.path.join(npth, d, 'index.ipynb')
        ifn = os.path.join(rpth, d, 'index.rst')
        if update_required(nifn, ifn):
            with open(ifn, 'wt') as fo:
                title = pthidx[os.path.join(d, 'index')]
                reflbl = '.. _example_' + title.lower().replace(' ', '_') + \
                        '_index:'
                print('%s\n' % reflbl, file=fo)
                print('%s\n%s\n' % (title, '-' * len(title)), file=fo)
                print('.. toctree::\n   :maxdepth: 1\n', file=fo)
                # Write notebook list into index
                for n in nlst:
                    text = nidx[n]
                    print('   %s <%s>' % (text, n), file=fo)

        # Iterate over notebooks in current subdirectory
        for n in nlst:
            # Full path of current notebook
            nfn = os.path.join(npth, d, n + '.ipynb')
            # Full path of correspond script
            sfn = os.path.join(spth, d, n + '.py')
            # Only proceed if script and notebook exist
            if os.path.exists(sfn) and os.path.exists(nfn):
                # Full path to output rst file
                rstpth = os.path.join(rpth, d, n + '.rst')
                # Convert notebook to rst if notebook is newer than rst file
                # or if rst file doesn't exist
                if update_required(nfn, rstpth):
                    print('processing %s                ' % nfn, end='\r')
                    script_and_notebook_to_rst(sfn, nfn, n, rdir)
            else:
                print('WARNING: script %s or notebook %s not found' %
                          (sfn, nfn))
