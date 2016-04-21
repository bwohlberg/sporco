Overview
========

SParse Optimization Research COde (SPORCO) is a Python package for
solving optimisation problems with sparsity-inducing
regularisation. These consist primarily of sparse coding and
dictionary learning problems, including convolutional sparse coding
and dictionary learning, but there is also support for other problems
such as Total Variation regularisation and Robust PCA. In the current
version all of the optimisation algorithms are based on the Alternating
Direction Method of Multipliers (ADMM).



Installation
------------

To install SPORCO, enter the following commands in the package root
directory:

::

   python setup.py build
   python setup.py install

The install command will usually have to be performed with root permissions.


Usage
-----

Each optimisation algorithm is implemented as a separate
class. Solving a problem is straightforward, as illustrated in the
following example. After importing the appropriate module

::

   from sporco.admm import bpdn

create an object representing the desired algorithm options

::

   opt = bpdn.BPDN.Options({'Verbose' : True, 'MaxMainIter' : 500, 'RelStopTol' : 1e-6, 'AutoRho' : {'RsdlTarget' : 1.0}})

then initialise the solver object

::

  lmbda = 25.0
  b = bpdn.BPDN(D, s, lmbda, opt)

(here ``D`` and ``s`` are numpy arrays representing the dictionary and
the signal to be decomposed) and call the ``solve`` method

::

  b.solve()

leaving the result in ``b.X``.


Usage Examples
--------------

Scripts illustrating usage of the package in more detail can be found
in the ``examples`` directory. These examples can be run from the root
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




Citing
------

If you use this library for published work, please cite it as (see
bibtex entry wohlberg-2016-sporco in ``docs/source/references.bib``):

  B. Wohlberg, "SParse Optimization Research COde (SPORCO)", Software library 
  available from http://math.lanl.gov/~brendt/Software/SPORCO/ , 
  Version (P) 0.0.1, 2016.

If you make use of any of the convolutional sparse representation
functions, please also cite :cite:`wohlberg-2016-efficient`.



Contact
-------

Please submit bug reports, comments, etc. to brendt@ieee.org



License
-------

This library was developed at Los Alamos National Laboratory, and has
been approved for public release under the approval number
LA-CC-14-057. It is made available under the terms of the BSD 3-Clause
License; please see the ``LICENSE`` file for further details.
