"""SPORCO package configuration."""

from builtins import next
from builtins import filter

import os
from glob import glob
from setuptools import setup
import os.path
from ast import parse


name = 'sporco'

# Get version number from sporco/__init__.py
# See http://stackoverflow.com/questions/2058802
with open(os.path.join(name, '__init__.py')) as f:
    version = parse(next(filter(
        lambda line: line.startswith('__version__'),
        f))).body[0].value.value

packages = ['sporco', 'sporco.prox', 'sporco.admm', 'sporco.pgm',
            'sporco.dictlrn', 'sporco.cuda', 'sporco.data',
            'sporco.cupy', 'sporco.cupy.pgm', 'sporco.cupy.admm', 'sporco.cupy.dictlrn']

docdirbase = 'share/doc/%s-%s' % (name, version)
data = [(os.path.join(docdirbase, 'examples/scripts'),
        ['examples/scripts/index.rst'])]
for d in glob('examples/scripts/*'):
    if os.path.isdir(d):
        data.append((os.path.join(docdirbase, d),
                    [os.path.join(d, 'index.rst')] +
                    glob(os.path.join(d, '*.py'))))


longdesc = \
"""
SPORCO is a Python package for solving optimisation problems with
sparsity-inducing regularisation. These consist primarily of sparse
coding and dictionary learning problems, including convolutional
sparse coding and dictionary learning, but there is also support for
other problems such as Total Variation regularisation and Robust
PCA. The optimisation algorithms in the current version are based
on the Alternating Direction Method of Multipliers (ADMM) or on
the Fast Iterative Shrinkage-Thresholding Algorithm (PGM).
"""

install_requires = ['future', 'numpy', 'scipy', 'imageio', 'filetype', 'matplotlib']
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    print("Building on ReadTheDocs")
    install_requires.append('ipython')
else:
    install_requires.append('pyfftw')

tests_require = ['pytest', 'pytest-runner']


setup(
    name             = name,
    version          = version,
    description      = 'Sparse Optimisation Research Code: A Python package ' \
                       'for sparse coding and dictionary learning',
    long_description = longdesc,
    keywords         = ['Sparse Representations', 'Sparse Coding',
                        'Dictionary Learning',
                        'Convolutional Sparse Representations',
                        'Convolutional Sparse Coding', 'Optimization',
                        'ADMM', 'PGM'],
    platforms        = 'Any',
    license          = 'BSD-3-Clause',
    url              = 'https://github.com/bwohlberg/sporco',
    author           = 'Brendt Wohlberg',
    author_email     = 'brendt@ieee.org',
    packages         = packages,
    package_data     = {'sporco': ['data/*.png', 'data/*.jpg', 'data/*.npz']},
    data_files       = data,
    include_package_data = True,
    install_requires = install_requires,
    extras_require   = {
        'tests': tests_require,
        'docs': ['sphinx >=2.2', 'numpydoc', 'sphinxcontrib-bibtex',
                 'sphinx_tabs', 'sphinx_fontawesome', 'jonga',
                 'ipython >=6.3.1', 'jupyter', 'py2jn', 'pypandoc'],
        'gpu': ['cupy', 'gputil', 'wurlitzer'],
        'optional': ['numexpr', 'mpldatacursor']},
    classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    zip_safe = False
)
