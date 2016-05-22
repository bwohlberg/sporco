tvl2
====

This module includes the :class:`.TVL2Denoise` and :class:`.TVL2Deconv`
classes, solving the following problems:

* :class:`.TVL2Denoise`

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
        \| W_{\mathrm{df}}  (\mathbf{x} - \mathbf{s}) \|_2^2 +
             \lambda \| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 + 
             (G_c \mathbf{x})^2}\|_1

* :class:`.TVL2Deconv`

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       \| A * \mathbf{x} - \mathbf{s} \|_2^2 +
       \lambda \| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
       (G_c \mathbf{x})^2} \|_1



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.TVL2Denoise` usage

    .. literalinclude:: ../../../examples/demo_tvl1denoise_1.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.TVL2Deconv` usage (example 1)

    .. literalinclude:: ../../../examples/demo_tvl1deconv_1.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.TVL2Deconv` usage (example 2)

    .. literalinclude:: ../../../examples/demo_tvl1deconv_2.py
       :language: python
       :lines: 14-
