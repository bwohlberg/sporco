Inverse Problems
================

SPORCO provides a set of :doc:`classes <admm>` supporting the
construction of new optimisation algorithms within the ADMM framework
:cite:`boyd-2010-distributed`, as well as a set of complete
algorithms, based on this framework, for solving optimisation problems
within the following categories:


Standard Sparse Representations
-------------------------------

* :doc:`bpdn <invprob/bpdn>` (Basis Pursuit DeNoising)
* :doc:`cmod <invprob/cmod>` (Constrained Method of Optimal Directions)
* :doc:`bpdndl <invprob/bpdndl>` (Basis Pursuit DeNoising Dictionary Learning)


Convolutional Sparse Representations
------------------------------------

* :doc:`cbpdn <invprob/cbpdn>` (Convolutional Basis Pursuit DeNoising)
* :doc:`ccmod <invprob/ccmod>` (Convolutional Constrained Method of Optimal Directions)
* :doc:`cbpdndl <invprob/cbpdndl>` (Convolutional Basis Pursuit DeNoising Dictionary Learning)



Miscellaneous
-------------

* :doc:`tvl1 <invprob/tvl1>` (:math:`\ell^1` Total Variation)
* :doc:`tvl2 <invprob/tvl2>` (:math:`\ell^2` Total Variation)
* :doc:`spline <invprob/spline>` (:math:`\ell^1` Spline)
* :doc:`rpca <invprob/rpca>` (Robust Principal Component Analysis)



.. toctree::
   :hidden:

   invprob/bpdn
   invprob/cmod
   invprob/bpdndl
   invprob/cbpdn
   invprob/ccmod
   invprob/cbpdndl
   invprob/tvl1
   invprob/tvl2
   invprob/spline
   invprob/rpca
