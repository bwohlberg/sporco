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

* :mod:`.plot`

  Functions for plotting graphs or 3D surfaces and visualising images.

* :mod:`.linalg`

  Various linear algebra and related functions, including solvers for
  specific forms of linear system and filters for computing image
  gradients.

* :mod:`.metric`

  Various image quality metrics including standard metrics such as
  MSE, SNR, and PSNR.

* :mod:`.cdict`

  A constrained dictionary class that constrains the allowed dict
  keys, and also initialises the dict with default content on
  instantiation. All of the inverse problem algorithm options classes
  are derived from this class.


The usage of many of these utility and support functions/classes is
demonstrated in the usage examples in the ``examples`` directory of
the source distribution.
