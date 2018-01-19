Overview
========

SParse Optimization Research COde (SPORCO) is a Python package for solving optimisation problems with sparsity-inducing regularisation :cite:`mairal-2014-sparse`. These consist primarily of sparse coding :cite:`chen-1998-atomic` and dictionary learning :cite:`engan-1999-method` problems, including convolutional sparse coding and dictionary learning :cite:`wohlberg-2016-efficient`, but there is also support for other problems such as Total Variation regularisation :cite:`rudin-1992-nonlinear` :cite:`alliney-1992-digital` and Robust PCA :cite:`cai-2010-singular`. The optimisation algorithms in the current version are based on the Alternating Direction Method of Multipliers (ADMM) :cite:`boyd-2010-distributed` or on the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) :cite:`beck-2009-fast`.

In addition to this documentation, an overview of the design and functionality of SPORCO is also available in :cite:`wohlberg-2017-sporco`.


.. _usage-section:

Usage Examples
--------------

Usage examples are available as Python scripts and Jupyter Notebooks.


.. _example-scripts-section:

Example Scripts
^^^^^^^^^^^^^^^

A large collection of scripts illustrating usage of the package can be found in the ``examples`` directory of the source distribution. These examples can be run from the root directory of the package by, for example

::

   python examples/scripts/sc/bpdn.py


To run these scripts prior to installing the package it will be necessary to first set the ``PYTHONPATH`` environment variable to include the root directory of the package. For example, in a ``bash`` shell

::

   export PYTHONPATH=$PYTHONPATH:`pwd`

from the root directory of the package, or in a Windows Command Prompt shell

::

   set PYTHONPATH=%PYTHONPATH%;C:\path_to_sporco_root

If SPORCO has been installed via ``pip``, the examples can be found in the directory in which ``pip`` installs documentation, e.g. ``/usr/local/share/doc/sporco-x.y.z/examples/``.


Jupyter Notebooks
^^^^^^^^^^^^^^^^^

`Jupyter Notebook <http://jupyter.org/>`_ examples are also `available <https://github.com/bwohlberg/sporco-notebooks>`_. These examples can be viewed online via `nbviewer <https://nbviewer.jupyter.org/github/bwohlberg/sporco-notebooks/blob/master/index.ipynb>`_, or run interactively at `binder <https://mybinder.org/v2/gh/bwohlberg/sporco-notebooks/master?filepath=index.ipynb>`_.



Citing
------

If you use this library for published work, please cite :cite:`wohlberg-2016-sporco` or :cite:`wohlberg-2017-sporco` (see bibtex entries ``wohlberg-2016-sporco`` and ``wohlberg-2017-sporco`` in ``docs/source/references.bib`` in the source distribution). If you use of any of the convolutional sparse representation functions, please also cite :cite:`wohlberg-2016-efficient` and any other papers relevant to the specific functionality that is used, e.g.  :cite:`wohlberg-2016-convolutional`, :cite:`wohlberg-2016-convolutional2`, :cite:`wohlberg-2016-boundary`, :cite:`garcia-2017-convolutional`.



Contact
-------

Please submit bug reports, comments, etc. to brendt@ieee.org. Bugs and feature requests can also be reported via the `GitHub Issues interface <https://github.com/bwohlberg/sporco/issues>`_.



License
-------

This package was developed at Los Alamos National Laboratory, and has been approved for public release under the approval number LA-CC-14-057. It is made available under the terms of the BSD 3-Clause License (see the `LICENSE <https://github.com/bwohlberg/sporco/blob/master/LICENSE>`__ file for details).

This material was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software. NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
