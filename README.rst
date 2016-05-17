SParse Optimization Research COde (SPORCO)
==========================================

.. image:: https://travis-ci.org/bwohlberg/sporco.svg
    :target: https://travis-ci.org/bwohlberg/sporco
    :alt: Build Status
.. image:: https://readthedocs.org/projects/sporco/badge/?version=latest
    :target: http://sporco.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://badge.fury.io/py/sporco.svg
    :target: https://badge.fury.io/py/sporco
    :alt: PyPi Release

SPORCO is a Python package for solving optimisation problems with
sparsity-inducing regularisation. These consist primarily of sparse
coding and dictionary learning problems, including convolutional
sparse coding and dictionary learning, but there is also support for
other problems such as Total Variation regularisation and Robust
PCA. In the current version all of the optimisation algorithms are
based on the Alternating Direction Method of Multipliers (ADMM).


Requirements
------------

The primary requirements are Python itself, and modules scipy, numpy,
future, pyfftw, and matplotlib. Module numexpr is not required, but
some functions will be faster if it is installed.

Installation of these requirements is system dependent. Under a recent
version of Ubuntu Linux, the following commands should be sufficient
for Python 2

::

   sudo apt-get install python-numpy
   sudo apt-get install python-scipy
   sudo apt-get install python-numexpr
   sudo apt-get install python-matplotlib
   sudo apt-get install python-pytest
   sudo apt-get install python-numpydoc
   sudo apt-get install python-pip
   sudo apt-get install libfftw3-dev
   sudo pip install future
   sudo pip install pyfftw
   sudo pip install sphinxcontrib-bibtex

or Python 3

::

   sudo apt-get install python3-numpy
   sudo apt-get install python3-scipy
   sudo apt-get install python3-numexpr
   sudo apt-get install python3-matplotlib
   sudo apt-get install python3-pytest
   sudo apt-get install python3-pip
   sudo apt-get install libfftw3-dev
   sudo pip3 install future
   sudo pip3 install pytest-runner
   sudo pip3 install pyfftw



Installation
------------

::

   python setup.py build
   python setup.py install

The install command will usually have to be performed with root permissions.


Usage
-----

Scripts illustrating usage of the package can be found in the
``examples`` directory. These examples can be run from the root
directory of the package by, for example

::

   python examples/demo_bpdn.py


To run these scripts prior to installing the package it will be
necessary to first set the ``PYTHONPATH`` environment variable to
include the root directory of the package. For example, in a ``bash``
shell

::

   export PYTHONPATH=$PYTHONPATH:`pwd`


from the root directory of the package.


Documentation
-------------

If the source has been obtained from a source distribution package
then HTML documentation can be built in the ``build/sphinx/html``
directory (the top-level document is ``index.html``) by the command

::

   python setup.py build_sphinx


If the source has been cloned from the project github, it is necessary
to first issue the command

::

   sphinx-apidoc --separate -d 2 -o source ../sporco modules.rst

within the ``docs`` directory.


License
-------

This package is distributed with a BSD license; see the ``LICENSE``
file for details.


Acknowledgments
---------------

Thanks for Aric Hagberg for valuable advice on python packaging and
related issues.
