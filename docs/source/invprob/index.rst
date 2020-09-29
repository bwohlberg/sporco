Inverse Problems
================

SPORCO provides a set of classes supporting the construction of new optimisation algorithms within the :doc:`ADMM <../admm/index>` :cite:`boyd-2010-distributed` and :doc:`PGM <../pgm/index>` :cite:`beck-2009-fast` frameworks, as well as a set of complete algorithms, based on these frameworks, for solving optimisation problems within the following categories:


Standard Sparse Representations
-------------------------------

* :doc:`bpdn <bpdn>` (Basis Pursuit DeNoising)
* :doc:`cmod <cmod>` (Constrained Method of Optimal Directions)
* :doc:`bpdndl <bpdndl>` (Basis Pursuit DeNoising Dictionary Learning)


Convolutional Sparse Representations
------------------------------------

* :doc:`cbpdn <cbpdn>` (Convolutional Basis Pursuit DeNoising)
* :doc:`ccmod <ccmod>` (Convolutional Constrained Method of Optimal Directions)
* :doc:`cbpdndl <cbpdndl>` (Convolutional Basis Pursuit DeNoising Dictionary Learning)


Dictionary Learning
-------------------

The :class:`.DictLearn` class in the :mod:`.dictlrn.dictlrn` module provides a flexible framework for more general :doc:`dictionary learning <dictlrn>` problems than those in modules :doc:`bpdndl <bpdndl>` and :doc:`cbpdndl / cbpdndlmd / prlcnscdl <cbpdndl>`.


Miscellaneous
-------------

* :doc:`tvl1 <tvl1>` (:math:`\ell_1` Total Variation)
* :doc:`tvl2 <tvl2>` (:math:`\ell_2` Total Variation)
* :doc:`spline <spline>` (:math:`\ell_1` Spline)
* :doc:`rpca <rpca>` (Robust Principal Component Analysis)
* :doc:`ppp <ppp>` (Plug and Play Priors)


Extensions
----------

The :doc:`ADMM <../admm/index>` and :doc:`PGM <../pgm/index>` classes are designed to for ease of implementation of new problems and simplicity of extension of existing problems. See :class:`.admm.bpdn.BPDN` as an example of the methods that need to be defined to implement a complete ADMM algorithm, and :class:`.BPDNJoint` as an example of the minor additional code required to extend an existing problem.



.. toctree::
   :hidden:

   bpdn
   cmod
   bpdndl
   cbpdn
   ccmod
   cbpdndl
   dictlrn
   tvl1
   tvl2
   spline
   rpca
   ppp
