Utility/Support
===============

In addition to the main set of classes for solving inverse problems,
SPORCO provides a number of supporting functions and classes, within
the following modules:

* :mod:`.util`

  Various utility functions and classes, including a
  parallel-processing grid search for parameter optimisation, access
  to a set of pre-learned convolutional dictionaries, and access to a
  set of example images.

* :mod:`.mpiutil`

  A parallel-processing grid search for parameter optimisation that
  distributes processes via MPI.

* :mod:`.array`

  Various functions operating on NumPy arrays.

* :mod:`.linalg`

  Various linear algebra and related functions, including solvers for
  specific forms of linear system and filters for computing image
  gradients.

* :mod:`.fft`

  Variants of the Fast Fourier Transform and associated functions.

* :mod:`.interp`

  Interpolation and regression functions.

* :mod:`.prox`

  Evaluation of various norms and their proximal operators and projection
  operators.

* :mod:`.metric`

  Various image quality metrics including standard metrics such as
  MSE, SNR, and PSNR.

* :mod:`.signal`

  Signal and image processing and associated functions.

* :mod:`.plot`

  Functions for plotting graphs or 3D surfaces and visualising images.

* :mod:`.cnvrep`

  Support classes and functions for working with convolutional
  representations.

* :mod:`.cdict`

  A constrained dictionary class that constrains the allowed dict
  keys, and also initialises the dict with default content on
  instantiation. All of the inverse problem algorithm options classes
  are derived from this class.


The usage of many of these utility and support functions/classes is
demonstrated in the :doc:`usage examples <examples/index>`.
