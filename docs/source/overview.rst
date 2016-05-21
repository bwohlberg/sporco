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



Usage
-----

Each optimisation algorithm is implemented as a separate
class. Solving a problem is straightforward, as illustrated in the
following example for solving

   .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \;
       \lambda \| \mathbf{x} \|_1 \quad . \;

Assume that :math:`D` and :math:`\mathbf{s}` are existing numpy arrays
representing the dictionary matrix and the signal vector to be
decomposed. After importing the appropriate module

::

   from sporco.admm import bpdn

create an object representing the desired algorithm options

::

   opt = bpdn.BPDN.Options({'Verbose' : True, 'MaxMainIter' : 500, 'RelStopTol' : 1e-6, 'AutoRho' : {'Enabled' : True}})

then initialise the solver object

::

  lmbda = 25.0
  b = bpdn.BPDN(D, s, lmbda, opt)

and call the ``solve`` method

::

  x = b.solve()

leaving the result in ``x``.



Usage Examples
--------------

Scripts illustrating usage of the package in more detail can be found
in the ``examples`` directory of the source distribution. These
examples can be run from the root directory of the package by, for
example

::

   python examples/demo_bpdn.py


To run these scripts prior to installing the package it will be
necessary to first set the ``PYTHONPATH`` environment variable to
include the root directory of the package. For example, in a ``bash``
shell

::

   export PYTHONPATH=$PYTHONPATH:`pwd`


from the root directory of the package.



Citing
------

If you use this library for published work, please cite it as in
:cite:`wohlberg-2016-sporco` (see bibtex entry ``wohlberg-2016-sporco`` in
``docs/source/references.bib`` in the source distribution). If you make
use of any of the convolutional sparse representation functions,
please also cite :cite:`wohlberg-2016-efficient`.



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
