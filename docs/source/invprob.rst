Inverse Problems
================

SPORCO provides a :doc:`set of classes <admm>` supporting the
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


Dictionary Learning
-------------------

The :class:`.DictLearn` class in the :mod:`.dictlrn` module provides a
flexible framework for more general :doc:`dictionary learning
<invprob/dictlrn>` problems than those in modules :doc:`bpdndl
<invprob/bpdndl>` and :doc:`cbpdndl <invprob/cbpdndl>`.


Miscellaneous
-------------

* :doc:`tvl1 <invprob/tvl1>` (:math:`\ell_1` Total Variation)
* :doc:`tvl2 <invprob/tvl2>` (:math:`\ell_2` Total Variation)
* :doc:`spline <invprob/spline>` (:math:`\ell_1` Spline)
* :doc:`rpca <invprob/rpca>` (Robust Principal Component Analysis)


Extensions
----------

The :doc:`ADMM classes <admm>` are designed to for ease of
implementation of new problems and simplicity of extension of existing
problems. See :class:`.BPDN` as an example of the methods that need to
be defined to implement a complete ADMM algorithm, and
:class:`.BPDNJoint` as an example of the minor additional code
required to extend an existing problem.
