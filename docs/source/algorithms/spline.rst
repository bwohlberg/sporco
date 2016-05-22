spline
======

This module includes the :class:`.SplineL1` class for solving the
problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
        \| W(\mathbf{x} - \mathbf{s}) \|_1 + \frac{\lambda}{2} \;
        \| D \mathbf{x} \|_2^2



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.SplineL1` usage

    .. literalinclude:: ../../../examples/demo_spline.py
       :language: python
       :lines: 14-
