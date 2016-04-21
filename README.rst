SParse Optimization Research COde (SPORCO)
==========================================

SPORCO is a Python package for solving optimisation problems with
sparsity-inducing regularisation. These consist primarily of sparse
coding and dictionary learning problems, including convolutional
sparse coding and dictionary learning, but there is also support for
other problems such as Total Variation regularisation and Robust
PCA. In the current version all of the optimisation algorithms are
based on the Alternating Direction Method of Multipliers (ADMM).


Requirements
------------

The primary requirements are Python itself (SPORCO has only been
tested on version 2.7), and modules scipy, numpy, pyfftw, and
matplotlib. Module numexpr is not required, but some functions will be
faster if it is installed.

Installation of these requirements is system dependent. Under a recent
version of Ubuntu Linux, the following commands should be sufficient:

::

   sudo apt-get install python-numpy
   sudo apt-get install python-scipy
   sudo apt-get install python-matplotlib
   sudo apt-get install python-pip
   sudo apt-get install libfftw3-dev
   sudo pip install pyfftw


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

HTML documentation can be built in the ``build/sphinx/html`` directory
(the top-level document is ``index.html``) by the command

::

   python setup.py build_sphinx


License
-------

This package is distributed with a BSD license; see the
``LICENSE.txt`` file for details.
