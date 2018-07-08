Modules cbpdndl, cbpdndlmd, and prlcnscdl
=========================================

These modules includes classes :class:`.dictlrn.cbpdndl.ConvBPDNDictLearn` and :class:`.dictlrn.prlcnscdl.ConvBPDNDictLearn_Consensus` for solving the problem

.. math::
   \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
   \frac{1}{2} \sum_k \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right \|_2^2 + \lambda \sum_k \sum_m \| \mathbf{x}_{k,m} \|_1
   \quad \text{ s.t. } \quad \mathbf{d}_m \in C \;\; \forall m \;,

where :math:`C` is the feasible set consisting of filters with unit norm
and constrained support.

In addition classes :class:`.dictlrn.cbpdndlmd.ConvBPDNMaskDictLearn` and :class:`.dictlrn.prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus` support solving
the problem

.. math::
   \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
   \frac{1}{2} \sum_k \left \|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right) \right \|_2^2 + \lambda \sum_k \sum_m \|
   \mathbf{x}_{k,m} \|_1 \quad \text{ s.t. } \quad \mathbf{d}_m \in C \;\;
   \forall m \;,

where :math:`W` is a mask array.

On a multi-core host, :class:`.dictlrn.prlcnscdl.ConvBPDNDictLearn_Consensus` and :class:`.dictlrn.prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus` can be
substantially faster than :class:`.dictlrn.cbpdndl.ConvBPDNDictLearn` and
:class:`.dictlrn.cbpdndlmd.ConvBPDNMaskDictLearn` respectively.


:ref:`Usage examples <example_convolutional_dictionary_learning_index>` are available.
