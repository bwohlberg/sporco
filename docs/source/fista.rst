FISTA Algorithms
================

FISTA algorithm development is supported by a base class from which classes for specific optimisation problems may be derived. This base class provides the machinery for managing iterations and stopping conditions, etc. so that only methods providing the details of a specific problem need to be defined in a derived class.

* :doc:`FISTA <fista/fista>` A completely generic FISTA algorithm, for
  problems of the form

  .. math::
       \mathrm{argmin}_{\mathbf{x}} \;
       f(\mathbf{x}) + g(\mathbf{x}) \;\;,

  where :math:`f, g` are convex functions and :math:`f` is smooth.
  All the updates and gradients are computed in input domain.

* :doc:`FISTADFT <fista/fistadft>` A generic FISTA algorithm, for
  problems of the form

  .. math::
       \mathrm{argmin}_{\mathbf{x}} \;
       f(\mathbf{x}) + g(\mathbf{x}) \;\;,

  where :math:`f, g` are convex functions and :math:`f` is smooth.
  All the updates and gradients are computed in DFT domain.
