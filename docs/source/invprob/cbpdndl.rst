Module cbpdndl
==============

This module includes the :class:`.ConvBPDNDictLearn` class for solving the
problem

.. math::
   \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
   \frac{1}{2} \sum_k \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right \|_2^2 + \lambda \sum_k \sum_m \| \mathbf{x}_{k,m} \|_1
   \quad \text{ s.t } \quad \|\mathbf{d}_m\|_2 = 1



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.ConvBPDNDictLearn` usage (greyscale images,
        single-scale dictionary, as in :cite:`wohlberg-2016-efficient`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_gry_ssd.py
       :language: python
       :lines: 9-

.. container:: toggle

    .. container:: header

        :class:`.ConvBPDNDictLearn` usage (greyscale images,
	multi-scale dictionary, as in :cite:`wohlberg-2016-efficient`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_gry_msd.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.ConvBPDNDictLearn` usage (colour images,
        single-scale colour dictionary, as in
	:cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_clr_ssd.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.ConvBPDNDictLearn` usage (colour images,
        multi-scale colour dictionary, as in
	:cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_clr_msd.py
       :language: python
       :lines: 9-
