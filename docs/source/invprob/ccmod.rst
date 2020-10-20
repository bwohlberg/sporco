Modules ccmod and ccmodmd
=========================

Modules :mod:`.admm.ccmod` and :mod:`.pgm.ccmod` include classes for
solving the problem

.. math::
   \mathrm{argmin}_\mathbf{d} \;
   \frac{1}{2} \sum_k \left \| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right \|_2^2 \quad \text{ such that } \quad \mathbf{d}_m
   \in C \;\; \forall m \;,

where :math:`C` is the feasible set consisting of filters with unit
norm and constrained support. Classes :class:`.ConvCnstrMOD_IterSM`,
:class:`.ConvCnstrMOD_CG`, and :class:`.ConvCnstrMOD_Consensus`
provide different methods of solving this problem, and
:func:`.admm.ccmod.ConvCnstrMOD` provides a mechanism for choosing one
of these classes via the ``method`` parameter specifying the solution
method.

A :ref:`usage example <examples_cdl_ccmod_cns_gry>` is available.

|

Modules :mod:`.admm.ccmodmd` and :mod:`.pgm.ccmod` include classes for solving the problem

.. math::
   \mathrm{argmin}_\mathbf{d} \;
   \frac{1}{2} \sum_k \left \| W \left(\sum_m \mathbf{d}_m *
   \mathbf{x}_{k,m} - \mathbf{s}_k \right) \right \|_2^2 \quad
   \text{ such that } \quad \mathbf{d}_m \in C \;\; \forall m \;,

where :math:`C` is the feasible set as above, and :math:`W` is a mask
array. Classes :class:`.ConvCnstrMODMaskDcpl_IterSM`,
:class:`.ConvCnstrMODMaskDcpl_CG`, and
:class:`.ConvCnstrMODMaskDcpl_Consensus` provide different methods of
solving this problem, and :func:`.ConvCnstrMODMaskDcpl` provides a
mechanism for choosing one of these classes via the ``method``
parameter specifying the solution method.
