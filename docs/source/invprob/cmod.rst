Module cmod
===========

This module includes the :class:`.CnstrMOD` class for solving the
problem

    .. math::
       \mathrm{argmin}_D (1/2) \| D X - S \|_2^2 \quad \text{ s.t }
       \quad \|\mathbf{d}_m\|_2 = 1



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.CnstrMOD` usage

    .. literalinclude:: ../../../examples/stdsparse/demo_cmod.py
       :language: python
       :lines: 9-
