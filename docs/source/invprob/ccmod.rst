Modules ccmod and ccmodmd
=========================

Module :mod:`.admm.ccmod` includes classes for solving the problem

.. math::
   \mathrm{argmin}_\mathbf{d} \;
   \frac{1}{2} \sum_k \left \| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right \|_2^2 \quad \text{ such that } \quad \mathbf{d}_m
   \in C \;\; \forall m \;,

where :math:`C` is the feasible set consisting of filters with unit norm and
constrained support. Classes :class:`.ConvCnstrMOD_IterSM`,
:class:`.ConvCnstrMOD_CG`, and :class:`.ConvCnstrMOD_Consensus` provide
different methods of solving this problem, and :func:`.ConvCnstrMOD` provides
a mechanism for choosing one of these classes via the ``method`` parameter specifying the solution method.

|

Module :mod:`.admm.ccmodmd` includes classes for solving the problem

.. math::
   \mathrm{argmin}_\mathbf{d} \;
   \frac{1}{2} \sum_k \left \| W \left(\sum_m \mathbf{d}_m *
   \mathbf{x}_{k,m} - \mathbf{s}_k \right) \right \|_2^2 \quad
   \text{ such that } \quad \mathbf{d}_m \in C \;\; \forall m \;,

where :math:`C` is the feasible set as above, and :math:`W` is a mask array. Classes :class:`.ConvCnstrMODMaskDcpl_IterSM`, :class:`.ConvCnstrMODMaskDcpl_CG`, and :class:`.ConvCnstrMODMaskDcpl_Consensus` provide different methods of solving this problem, and :func:`.ConvCnstrMODMaskDcpl` provides a mechanism for choosing one of these classes via the ``method`` parameter specifying the solution method.



Usage Examples
--------------

.. container:: toggle

    .. container:: header

	:class:`.ConvCnstrMOD_IterSM` usage (greyscale images, see
	:cite:`wohlberg-2016-efficient`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_ccmod_ism_gry.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

	:class:`.ConvCnstrMOD_Consensus` usage (greyscale images, see
	:cite:`sorel-2016-fast`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_ccmod_cns_gry.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

	:func:`.ConvCnstrMOD` usage (colour images, greyscale
	dictionary, see :cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_ccmod_clr_gd.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

	:class:`.ConvCnstrMOD_IterSM` usage (colour images, colour
	dictionary, see :cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_ccmod_ism_clr_cd.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

	:class:`.ConvCnstrMOD_Consensus` usage (colour images, colour
	dictionary, see :cite:`sorel-2016-fast`
	:cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_ccmod_cns_clr_cd.py
       :language: python
       :lines: 9-
