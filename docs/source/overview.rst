Overview
========

SParse Optimization Research COde (SPORCO) is a Python package for solving optimisation problems with sparsity-inducing regularisation :cite:`mairal-2014-sparse`. These consist primarily of sparse coding :cite:`chen-1998-atomic` and dictionary learning :cite:`engan-1999-method` problems, including convolutional sparse coding and dictionary learning :cite:`wohlberg-2016-efficient`, but there is also support for other problems such as Total Variation regularisation :cite:`rudin-1992-nonlinear` :cite:`alliney-1992-digital`, Robust PCA :cite:`cai-2010-singular`, and Plug and Play Priors :cite:`venkatakrishnan-2013-plugandplay2`. The optimisation algorithms in the current version are based on the Alternating Direction Method of Multipliers (ADMM) :cite:`boyd-2010-distributed` or on the Fast Iterative Shrinkage-Thresholding Algorithm (PGM) :cite:`beck-2009-fast`.

In addition to this documentation, an overview of the design and functionality of SPORCO is also available in :cite:`wohlberg-2017-sporco`.


GPU Acceleration
----------------

GPU accelerated versions of some of the SPORCO solvers are provided within the :mod:`sporco.cuda` and :mod:`sporco.cupy` subpackages. Use of the former requires installation of the `SPORCO-CUDA <https://github.com/bwohlberg/sporco-cuda>`__ extension package, while the latter requires installation of `CuPy <https://cupy.dev>`__ :cite:`okuta-2017-cupy`. The :mod:`sporco.cupy` subpackage supports a much wider range of problems than :mod:`sporco.cuda`, which only supports four different variants of convolutional sparse coding. However, the :mod:`sporco.cuda` functions are substantially faster than the corresponding functions in :mod:`sporco.cupy` since those in :mod:`sporco.cuda` are implemented directly in CUDA, while those in :mod:`sporco.cupy` use `CuPy <https://cupy.chainer.org/>`__ as a replacement for `NumPy <http://www.numpy.org/>`__.


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

If you use this library for published work, please cite :cite:`wohlberg-2016-sporco` or :cite:`wohlberg-2017-sporco` (see bibtex entries ``wohlberg-2016-sporco`` and ``wohlberg-2017-sporco`` in ``docs/source/references.bib`` in the source distribution). If you use of any of the convolutional sparse representation classes, please also cite any other papers relevant to the specific functionality that is used, e.g. :cite:`wohlberg-2016-efficient`, :cite:`wohlberg-2016-convolutional`, :cite:`wohlberg-2016-convolutional2`, :cite:`wohlberg-2016-boundary`, :cite:`garcia-2018-convolutional1`.



Contact
-------

Please submit bug reports, feature requests, etc. via the `GitHub Issues interface <https://github.com/bwohlberg/sporco/issues>`_.



License
-------

This package was developed at Los Alamos National Laboratory, and has been approved for public release under the approval number LA-CC-14-057. It is made available under the terms of the BSD 3-Clause License (see the `LICENSE <https://github.com/bwohlberg/sporco/blob/master/LICENSE>`__ file for details).

This package was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. If this software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
