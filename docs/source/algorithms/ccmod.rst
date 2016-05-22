ccmod
=====

This module includes the :class:`.ConvCnstrMOD` class for solving the
problem

    .. math::
       \mathrm{argmin}_\mathbf{d} \;
       (1/2) \sum_k \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} - 
       \mathbf{s}_k \|_2^2 \quad \text{ s.t } \quad  \|\mathbf{d}_m\|_2 = 1


Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.ConvCnstrMOD` usage (greyscale images) 

    .. literalinclude:: ../../../examples/demo_ccmod_gry.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.ConvCnstrMOD` usage (colour images)

    .. literalinclude:: ../../../examples/demo_ccmod_clr.py
       :language: python
       :lines: 14-
