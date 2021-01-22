# -*- coding: utf-8 -*-
#
# SPORCO documentation build configuration file, created by
# sphinx-quickstart on Tue Apr  7 06:02:44 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
from builtins import next
from builtins import filter
from ast import parse
import re, shutil, tempfile
import glob
import fileinput
import inspect

if sys.version[0] == '3':
    from unittest.mock import MagicMock
elif sys.version[0] == '2':
    from mock import Mock as MagicMock
else:
    raise ImportError("Can't determine how to import MagicMock.")

confpath = os.path.dirname(__file__)
sys.path.append(confpath)
import callgraph
import docntbk


on_rtd = os.environ.get('READTHEDOCS') == 'True'


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
rootpath = os.path.abspath('../..')
sys.path.insert(0, rootpath)


# Code to disable sporco.common._fix_nested_class_lookup function
# no longer needed here as it is now handled in the imported automodule
# module


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.5.4'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
 #   'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.graphviz',
    'sphinx_tabs.tabs',
    'sphinx_fontawesome'
]


bibtex_bibfiles = ['references.bib']


# Copied from scikit-learn sphinx configuration
if os.environ.get('NO_MATHJAX'):
    extensions.append('sphinx.ext.imgmath')
    imgmath_image_format = 'svg'
else:
    extensions.append('sphinx.ext.mathjax')
    mathjax_path = 'https://cdn.mathjax.org/mathjax/latest/' \
                   'MathJax.js?config=TeX-AMS_HTML'

# generate autosummary pages
# autosummary_generate = True
# autosummary_generate = False


# See https://stackoverflow.com/questions/5599254
autoclass_content = 'both'

#autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'SPORCO'
copyright = u'2015-2020, Brendt Wohlberg'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
with open(os.path.join('../../sporco', '__init__.py')) as f:
    version = parse(next(filter(
        lambda line: line.startswith('__version__'),
        f))).body[0].value.s
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['tmp', '*.tmp.*', '*.tmp']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#html_theme = 'default'
#import sphinx_rtd_theme
#import sphinx_readable_theme
#html_theme = "sphinx_rtd_theme"
#html_theme = "bizstyle"
html_theme = "haiku"
#html_theme = "agogo"
#html_theme = 'readable'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
#html_theme_path = [sphinx_readable_theme.get_html_theme_path()]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None
html_logo = '_static/logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
#html_style = 'sporco.css'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
if on_rtd:
    html_static_path = []
else:
    html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'SPORCOdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  ('index', 'SPORCO.tex', u'SPORCO Documentation',
   u'Brendt Wohlberg', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True



# Intersphinx mapping
intersphinx_mapping = {'https://docs.python.org/3/': None,
                       'https://docs.scipy.org/doc/numpy/': None,
                       'https://docs.scipy.org/doc/scipy/reference/': None,
                       'https://matplotlib.org/': None,
                       'http://hgomersall.github.io/pyFFTW/': None,
                       'http://docs-cupy.chainer.org/en/stable': None
                      }
# Added timeout due to periodic scipy.org down time
#intersphinx_timeout = 30

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

graphviz_output_format = 'svg'
inheritance_graph_attrs = dict(rankdir="LR", fontsize=9, ratio='compress',
                               bgcolor='transparent')
inheritance_node_attrs = dict(shape='box', fontsize=9, height=0.4,
                              margin='"0.08, 0.03"', style='"rounded,filled"',
                              fillcolor='"#f4f4ffff"')


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'sporco', u'SPORCO Documentation',
     [u'Brendt Wohlberg'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'SPORCO', u'SPORCO Documentation',
   u'Brendt Wohlberg', 'SPORCO', 'SParse Optimization Research COde (SPORCO)',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False


MOCK_MODULES = ['sporco.cuda', 'sporco.cupy', 'mpi4py']

if on_rtd:
    print('Building on ReadTheDocs')
    print
    print("Current working directory: {}" . format(os.path.abspath(os.curdir)))

    import numpy as np
    print('NumPy version: %s' % np.__version__)

    import matplotlib
    matplotlib.use('agg')


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


print('rootpath: %s' % rootpath)
print('confpath: %s' % confpath)


# See https://developer.ridgerun.com/wiki/index.php/How_to_generate_sphinx_documentation_for_python_code_running_in_an_embedded_system

# Sort members by type
#autodoc_member_order = 'groupwise'
autodoc_member_order = 'bysource'
#autodoc_default_flags = ['members', 'inherited-members', 'show-inheritance']
#autodoc_default_flags = ['show-inheritance']
autodoc_default_options = {'show-inheritance': True}
autodoc_docstring_signature = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '**tests**', '**spi**']



# Ensure that the __init__ method gets documented.
def skip_member(app, what, name, obj, skip, options):
    #if name == "__init__":
    #    return False
    if name == "IterationStats":
        return True
    #if name == "timer":
    #    return True
    return skip


def process_docstring(app, what, name, obj, options, lines):
    if "IterationStats." in name:
        print("------> %s" % name)
        for n in xrange(len(lines)):
            print(lines[n])
        #for n in xrange(len(lines)):
        #    lines[n] = ''


def process_signature(app, what, name, obj, options, signature,
                      return_annotation):
    if "IterationStats." in name:
        print("%s : %s, %s" % (name, signature, return_annotation))


def subpackage_summary(*args):

    import automodule

    pkgname = 'sporco'
    modpath = os.path.join(rootpath, 'sporco')
    tmpltpath = os.path.join(confpath, '_templates/autosummary')
    outpath = os.path.join(confpath, 'modules')
    automodule.write_module_docs(pkgname, modpath, tmpltpath, outpath)


def insertsolve(_):
    # Insert documentation for inherited solve methods
    callgraph.insert_solve_docs()


def gencallgraph(_):

    print('Constructing call graph images')
    if on_rtd:
        cgpth = '_static/jonga'
    else:
        cgpth = os.path.join(confpath, '_static/jonga')
    callgraph.gengraphs(cgpth)



def genexamples(_):

    import zipfile
    from sporco.util import netgetdata

    url = 'https://codeload.github.com/bwohlberg/sporco-notebooks/zip/master'
    print('Constructing docs from example scripts')
    if on_rtd:
        epth = '../../examples'
    else:
        epth = os.path.join(rootpath, 'examples')
    spth = os.path.join(epth, 'scripts')
    npth = os.path.join(epth, 'notebooks')
    if on_rtd:
        rpth = 'examples'
    else:
        rpth = os.path.join(confpath, 'examples')

    if not os.path.exists(npth):
        print('Notebooks required for examples section not found: '
              'downloading from sporco-notebooks repo on GitHub')
        zipdat = netgetdata(url)
        zipobj = zipfile.ZipFile(zipdat)
        zipobj.extractall(path=epth)
        os.rename(os.path.join(epth, 'sporco-notebooks-master'),
                  os.path.join(epth, 'notebooks'))

    docntbk.make_example_scripts_docs(spth, npth, rpth)



def fix_inherit_diagram(*args):

    if on_rtd:
        buildpath = os.path.join(confpath, '_build/html')
    else:
        buildpath = os.path.join(rootpath, 'build/sphinx/html')
    images = os.path.join(buildpath, '_images/inheritance-*svg')

    for fnm in glob.glob(images):
        f = fileinput.FileInput(fnm, inplace=True)
        for line in f:
            line = re.sub(r'\.\./sporco', '../modules/sporco', line.rstrip())
            print(line)


def setup(app):

    app.add_css_file('sporco.css')
    app.add_css_file('http://netdna.bootstrapcdn.com/font-awesome/4.7.0/'
                     'css/font-awesome.min.css')
    app.connect('autodoc-skip-member', skip_member)
    app.connect('builder-inited', insertsolve)
    app.connect('builder-inited', genexamples)
    app.connect('build-finished', fix_inherit_diagram)

    subpackage_summary()
    gencallgraph(None)
