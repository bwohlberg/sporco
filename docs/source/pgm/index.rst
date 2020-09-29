PGM Algorithms
==============

PGM algorithm development is supported by a base class from which
classes for specific optimisation problems may be derived. This base
class provides the machinery for managing iterations and stopping
conditions, etc. so that only methods providing the details of a
specific problem need to be defined in a derived class.

* :doc:`PGM <pgm>` A completely generic PGM algorithm, for
  problems of the form

  .. math::
       \mathrm{argmin}_{\mathbf{x}} \;
       f(\mathbf{x}) + g(\mathbf{x}) \;\;,

  where :math:`f, g` are convex functions and :math:`f` is smooth.
  All the updates and gradients are computed in input domain.

* :doc:`PGMDFT <pgmdft>` A generic PGM algorithm, for
  problems of the form

  .. math::
       \mathrm{argmin}_{\mathbf{x}} \;
       f(\mathbf{x}) + g(\mathbf{x}) \;\;,

  where :math:`f, g` are convex functions and :math:`f` is smooth.
  All the updates and gradients are computed in DFT domain.

.. _sec-pgm-auxiliary-classes:


Auxiliary Classes
-----------------

The following classes provide auxiliary functionality for the PGM algorithms.

* :doc:`BacktrackBase <backtrack>` an iterative process to find the optimal step size for the gradient descent (:math:`L^{-1}`).

* :doc:`StepSizePolicyBase <stepsize>` a process to adapt the step size for the gradient descent (:math:`L^{-1}`).

* :doc:`MomentumBase <momentum>` a process to update the momentum coefficient :math:`t^{(j)}`.


.. toctree::
   :hidden:

   pgm
   pgmdft
   backtrack
   stepsize
   momentum
