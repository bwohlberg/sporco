#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Construct source for module docs similarly to sphinx.ext.autosummary."""

import sys
import os
import pkgutil
import inspect
import importlib
from types import CodeType

from jinja2 import FileSystemLoader
from jinja2.sandbox import SandboxedEnvironment

from sphinx.ext.autosummary.generate import _underline
from sphinx.util.rst import escape as rst_escape

from callgraph import is_newer_than


# Disable nested Options class name fix hack
import sporco.common
def _fix_disable(cls, nstnm):
    return cls
co = sporco.common._fix_nested_class_lookup.__code__
coattr = [co.co_argcount,]
if hasattr(co, 'co_posonlyargcount'):
    coattr.append(co.co_posonlyargcount)
coattr.extend([co.co_kwonlyargcount, co.co_nlocals, co.co_stacksize,
               co.co_flags, _fix_disable.__code__.co_code, co.co_consts,
               co.co_names, co.co_varnames, co.co_filename, co.co_name])
if hasattr(co, 'co_qualname'):
    coattr.append(co.co_qualname)
coattr.extend([co.co_firstlineno, co.co_lnotab])
if hasattr(co, 'co_exceptiontable'):
    coattr.append(co.co_exceptiontable)
coattr.extend([co.co_freevars, co.co_cellvars])
sporco.common._fix_nested_class_lookup.__code__ = CodeType(*coattr)



def sort_by_list_order(sortlist, reflist, reverse=False, fltr=False,
                       slemap=None):
    """
    Sort a list according to the order of entries in a reference list.

    Parameters
    ----------
    sortlist : list
      List to be sorted
    reflist : list
      Reference list defining sorting order
    reverse : bool, optional (default False)
      Flag indicating whether to sort in reverse order
    fltr : bool, optional (default False)
      Flag indicating whether to filter `sortlist` to remove any entries
      that are not in `reflist`
    slemap : function or None, optional (default None)
       Function mapping a sortlist entry to the form of an entry in
       `reflist`

    Returns
    -------
    sortedlist : list
      Sorted (and possibly filtered) version of sortlist
    """

    def keyfunc(entry):
        if slemap is not None:
            rle = slemap(entry)
        if rle in reflist:
            # Ordering index taken from reflist
            return reflist.index(rle)
        else:
            # Ordering index taken from sortlist, offset
            # by the length of reflist so that entries
            # that are not in reflist retain their order
            # in sortlist
            return sortlist.index(entry) + len(reflist)

    if fltr:
        if slemap:
            sortlist = filter(lambda x: slemap(x) in reflist, sortlist)
        else:
            sortlist = filter(lambda x: x in reflist, sortlist)

    return sorted(sortlist, key=keyfunc, reverse=reverse)


def get_module_names(filepath, modpath=None, skipunder=True):
    """
    Get a list of modules in a package/subpackage.

    Parameters
    ----------
    filepath : string
      Filesystem path to the root directory of the package/subpackage
    modpath : string or None, optional (default None)
      Name of package or module path (in the form 'a.b.c') to a
      subpackage
    skipunder : bool, optional (default True)
      Flag indicating whether to skip modules with names with an
      initial underscore

    Returns
    -------
    modlist : list
      List of module names
    """

    if modpath is None:
        modpath = os.path.split(filepath)[-1]

    modlst = []
    for ff, name, ispkg in pkgutil.iter_modules([filepath]):
        if not skipunder or name[0] != '_':
            if ispkg:
                sublst = get_module_names(os.path.join(filepath, name),
                                          modpath + '.' + name,
                                          skipunder=skipunder)
                if sublst:
                    modlst.extend(sublst)
                else:
                    modlst.append(modpath + '.' + name)
            else:
                modlst.append(modpath + '.' + name)

    return modlst


def sort_module_names(modlst):
    """
    Sort a list of module names in order of subpackage depth.

    Parameters
    ----------
    modlst : list
      List of module names

    Returns
    -------
    srtlst : list
      Sorted list of module names
    """

    return sorted(modlst, key=lambda x: x.count('.'))


def _member_sort_key(member):

    try:
        file = inspect.getsourcefile(member)
        _, line = inspect.getsourcelines(member)
    except OSError:
        file = '~~~~~~~~~~~~~~~'
        line = 999999

    return '%s line: %06d' % (file, line)


def _member_module_name(member):

    module = inspect.getmodule(member)
    if module:
        return module.__name__
    else:
        return 'None'


def get_module_members(module, type=None):
    """
    Get a list of module member objects, excluding members that are
    imported from other modules that are not submodules of the specified
    moodule. An attempt is made to sort the list in order of definition
    in the source file. If the module has an `__all__` attribute, the
    list is sorted in the same order, and members that are not in the
    list are filtered out.

    Parameters
    ----------
    module : string or module object
      Module for which member list is to be generated
    type : inspect predicate function (e.g. inspect.isfunction) or None
      Restrict returned members to specified type

    Returns
    -------
    mbrlst : list
      List of module members
    """

    # if module argument is a string, try to load the module with the
    # specified name
    if isinstance(module, str):
        module = importlib.import_module(module)

    # Get members in the module
    members = map(lambda x: x[1], inspect.getmembers(module, type))

    # Filter out members that are not defined in the specified module
    # or a submodule thereof
    members = filter(lambda x: module.__name__ in _member_module_name(x),
                     members)

    if hasattr(module, '__all__'):
        # If the module has an __all__ attribute, sort the members in the
        # same order as the entries in __all__, and filter out members
        # that are not listed in __all__. The slemap parameter of
        # sort_by_list_order is needed because the list to be sorted
        # consist of module member ojects, while __all__ is a list of
        # module name strings.
        members = sort_by_list_order(members, getattr(module, '__all__'),
                                     fltr=True, slemap=lambda x: x.__name__)
    else:
        # If the module does not have an __all__ attribute, attempt to
        # sort the members in the order in which they are defined in the
        # source file(s)
        members = sorted(members, key=_member_sort_key)

    return members


def get_module_functions(module):
    """
    Get a list of module member functions.

    Parameters
    ----------
    module : string or module object
      Module for which member list is to be generated

    Returns
    -------
    mbrlst : list
      List of module functions
    """
    return get_module_members(module, type=inspect.isfunction)


def get_module_classes(module):
    """
    Get a list of module member classes.

    Parameters
    ----------
    module : string or module object
      Module for which member list is to be generated

    Returns
    -------
    mbrlst : list
      List of module functions
    """

    clslst = get_module_members(module, type=inspect.isclass)
    return list(filter(lambda cls: not issubclass(cls, Exception),
                       clslst))


def get_module_exceptions(module):
    """
    Get a list of module member exceptions.

    Parameters
    ----------
    module : string or module object
      Module for which member list is to be generated

    Returns
    -------
    mbrlst : list
      List of module functions
    """

    clslst = get_module_members(module, type=inspect.isclass)
    return list(filter(lambda cls: issubclass(cls, Exception),
                       clslst))


class DocWriter:
    """
    Write module documentation source, using the same template format
    as `sphinx.ext.autosummary
    <www.sphinx-doc.org/en/stable/ext/autosummary.html>`__.
    """

    def __init__(self, outpath, tmpltpath):
        """
        Parameters
        ----------
        outpath : string
          Directory path for RST output files
        tmpltpath : string
          Directory path for autosummary template files
        """

        self.outpath = outpath
        self.template_loader = FileSystemLoader(tmpltpath)
        self.template_env = SandboxedEnvironment(loader=self.template_loader)
        self.template_env.filters['underline'] = _underline
        self.template_env.filters['escape'] = rst_escape
        self.template_env.filters['e'] = rst_escape
        self.template = self.template_env.get_template('module.rst')


    def write(self, module):
        """
        Write the RST source document for generating the docs for
        a specified module.

        Parameters
        ----------
        module : module object
          Module for which member list is to be generated
        """

        modname = module.__name__

        # Based on code in generate_autosummary_docs in https://git.io/fxpJS
        ns = {}
        ns['members'] = dir(module)
        ns['functions'] = list(map(lambda x: x.__name__,
                                   get_module_functions(module)))
        ns['classes'] = list(map(lambda x: x.__name__,
                                 get_module_classes(module)))
        ns['exceptions'] = list(map(lambda x: x.__name__,
                                    get_module_exceptions(module)))
        ns['fullname'] = modname
        ns['module'] = modname
        ns['objname'] = modname
        ns['name'] = modname.split('.')[-1]
        ns['objtype'] = 'module'
        ns['underline'] = len(modname) * '='
        rndr = self.template.render(**ns)

        rstfile = os.path.join(self.outpath, modname + '.rst')
        with open(rstfile, 'w') as f:
            f.write(rndr)


def write_module_docs(pkgname, modpath, tmpltpath, outpath):
    """
    Write the autosummary style docs for the specified package.

    Parameters
    ----------
    pkgname : string
      Name of package to document
    modpath : string
      Path to package source root directory
    tmpltpath : string
      Directory path for autosummary template files
    outpath : string
      Directory path for RST output files
    """

    dw = DocWriter(outpath, tmpltpath)

    modlst = get_module_names(modpath, pkgname)
    print('Making api docs:', end='')
    for modname in modlst:

        # Don't generate docs for cupy or cuda subpackages
        if 'cupy' in modname or 'cuda' in modname:
            continue

        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError:
            print('Error importing module %s' % modname)
            continue

        # Skip any virtual modules created by the copy-and-patch
        # approach in sporco.cupy. These should already have been
        # skipped due to the test for cupy above.
        if mod.__file__ == 'patched':
            continue

        # Construct api docs for the current module if the docs file
        # does not exist, or if its source file has been updated more
        # recently than an existing docs file
        if hasattr(mod, '__path__'):
            srcpath = mod.__path__[0]
        else:
            srcpath = mod.__file__
        dstpath = os.path.join(outpath, modname + '.rst')

        if is_newer_than(srcpath, dstpath):
            print(' %s' % modname, end='')
            dw.write(mod)

    print('')



# Allow module to be run as a script
if __name__ == "__main__":
    pkgname = 'sporco'
    modpath = '../../sporco'
    tmpltpath = '_templates/autosummary'
    outpath = 'modules'

    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        print('Cleaning auto-generated doc files')
        files = os.listdir(outpath)
        for f in ['index.rst', 'sporco.cupy.rst', 'sporco.cuda.rst']:
            files.remove(f)
        for f in files:
            os.remove(os.path.join(outpath, f))
    else:
        write_module_docs(pkgname, modpath, tmpltpath, outpath)
