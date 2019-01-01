Modules cbpdndl, cbpdndlmd, prlcnscdl, and onlinecdl
====================================================

These modules includes classes :class:`.cbpdndl.ConvBPDNDictLearn`, :class:`.prlcnscdl.ConvBPDNDictLearn_Consensus`, and :class:`.onlinecdl.OnlineConvBPDNDictLearn` for solving the problem

.. math::
   \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
   \frac{1}{2} \sum_k \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right \|_2^2 + \lambda \sum_k \sum_m \| \mathbf{x}_{k,m} \|_1
   \quad \text{ s.t. } \quad \mathbf{d}_m \in C \;\; \forall m \;,

where :math:`C` is the feasible set consisting of filters with unit norm
and constrained support.

In addition classes :class:`.cbpdndlmd.ConvBPDNMaskDictLearn`, :class:`.prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus`, and :class:`.onlinecdl.OnlineConvBPDNMaskDictLearn` support solving
the problem

.. math::
   \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
   \frac{1}{2} \sum_k \left \|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right) \right \|_2^2 + \lambda \sum_k \sum_m \|
   \mathbf{x}_{k,m} \|_1 \quad \text{ s.t. } \quad \mathbf{d}_m \in C \;\;
   \forall m \;,

where :math:`W` is a mask array.

On a multi-core host, :class:`.prlcnscdl.ConvBPDNDictLearn_Consensus` and :class:`.prlcnscdl.ConvBPDNMaskDcplDictLearn_Consensus` can be
substantially faster than :class:`.cbpdndl.ConvBPDNDictLearn` and
:class:`.cbpdndlmd.ConvBPDNMaskDictLearn` respectively. For large training datasets, the online CDL classes :class:`.onlinecdl.OnlineConvBPDNDictLearn` and :class:`.onlinecdl.OnlineConvBPDNMaskDictLearn` may be the best option, particularly for use on a host without very large amounts of main memory, and  with a GPU (see :ref:`cuda_package`).


:ref:`Usage examples <examples_cdl_index>` are available.
