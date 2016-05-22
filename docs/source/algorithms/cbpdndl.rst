cbpdndl
=======

This module includes the :class:`.ConvBPDNDictLearn` class for solving the
problem

    .. math::
       \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
       (1/2) \sum_k \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \|_2^2 + \lambda \sum_k \sum_m \| \mathbf{x}_{k,m} \|_1
       \quad \text{ s.t } \quad \|\mathbf{d}_m\|_2 = 1


Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.ConvBPDNDictLearn` usage (greyscale images, example 1)

    .. literalinclude:: ../../../examples/demo_cbpdndl_gry_1.py
       :language: python
       :lines: 14-

.. container:: toggle

    .. container:: header

        :class:`.ConvBPDNDictLearn` usage (greyscale images, example 2)

    .. literalinclude:: ../../../examples/demo_cbpdndl_gry_2.py
       :language: python
       :lines: 14-

.. container:: toggle

    .. container:: header

        :class:`.ConvBPDNDictLearn` usage (colour images, example 1)

    .. literalinclude:: ../../../examples/demo_cbpdndl_clr_1.py
       :language: python
       :lines: 14-

.. container:: toggle

    .. container:: header

        :class:`.ConvBPDNDictLearn` usage (colour images, example 2)

    .. literalinclude:: ../../../examples/demo_cbpdndl_clr_2.py
       :language: python
       :lines: 14-
