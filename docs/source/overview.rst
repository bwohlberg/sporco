Overview
========

SParse Optimization Research COde (SPORCO) is a Python package for
solving optimisation problems with sparsity-inducing
regularisation. These consist primarily of sparse coding
:cite:`chen-1998-atomic` and dictionary learning
:cite:`engan-1999-method` problems, including convolutional sparse
coding and dictionary learning :cite:`wohlberg-2016-efficient`, but
there is also support for other problems such as Total Variation
regularisation :cite:`rudin-1992-nonlinear`
:cite:`alliney-1992-digital` and Robust PCA
:cite:`cai-2010-singular`. In the current version all of the
optimisation algorithms are based on the Alternating Direction Method
of Multipliers (ADMM) :cite:`boyd-2010-distributed`.


.. _usage-section:

Usage Examples
--------------

Usage examples are available as Python scripts and Jupyter Notebooks.


.. _example-scripts-section:

Example Scripts
^^^^^^^^^^^^^^^

A large collection of scripts illustrating usage of the package can be
found in the ``examples`` directory of the source distribution. These
examples can be run from the root directory of the package by, for
example

::

   python examples/stdsparse/demo_bpdn.py


To run these scripts prior to installing the package it will be
necessary to first set the ``PYTHONPATH`` environment variable to
include the root directory of the package. For example, in a ``bash``
shell

::

   export PYTHONPATH=$PYTHONPATH:`pwd`


from the root directory of the package. If SPORCO has been installed
via ``pip``, the examples can be found in the directory in which ``pip``
installs documentation, e.g. ``/usr/local/share/doc/sporco-x.y.z/examples/``.


Jupyter Notebooks
^^^^^^^^^^^^^^^^^

`Jupyter Notebook <http://jupyter.org/>`_ versions of some of the
demos in ``examples`` are also available in the same directories as
the corresponding demo scripts. The scripts can also be viewed online
via `nbviewer <https://nbviewer.jupyter.org/github/bwohlberg/sporco/blob/master/index.ipynb>`_,
or run interactively at `binder <http://mybinder.org/repo/bwohlberg/sporco>`_.



Citing
------

If you use this library for published work, please cite it as in
:cite:`wohlberg-2016-sporco` (see bibtex entry ``wohlberg-2016-sporco`` in
``docs/source/references.bib`` in the source distribution). If you make
use of any of the convolutional sparse representation functions,
please also cite :cite:`wohlberg-2016-efficient` and any other papers
relevant to the specific functionality that is used, e.g.
:cite:`wohlberg-2016-convolutional`, :cite:`wohlberg-2016-convolutional2`,
:cite:`wohlberg-2016-boundary`.



Contact
-------

Please submit bug reports, comments, etc. to brendt@ieee.org. Bugs and
feature requests can also be reported via the
`GitHub Issues interface <https://github.com/bwohlberg/sporco/issues>`_.



BSD License
-----------

This library was developed at Los Alamos National Laboratory, and has
been approved for public release under the approval number
LA-CC-14-057. It is made available under the terms of the BSD 3-Clause
License; please see the ``LICENSE`` file for further details.



Acknowledgments
---------------

Thanks to Aric Hagberg for valuable advice on python packaging,
documentation, and related issues.
