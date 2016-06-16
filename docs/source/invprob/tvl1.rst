tvl1
====

This module includes the :class:`.TVL1Denoise` and :class:`.TVL1Deconv`
classes, solving the following problems:

* :class:`.TVL1Denoise`

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
        \| W_{\mathrm{df}}  (\mathbf{x} - \mathbf{s}) \|_1 +
             \lambda \| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 + 
             (G_c \mathbf{x})^2}\|_1

* :class:`.TVL1Deconv`

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       \| A * \mathbf{x} - \mathbf{s} \|_1 +
       \lambda \| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
       (G_c \mathbf{x})^2} \|_1



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.TVL1Denoise` usage

    .. literalinclude:: ../../../examples/demo_tvl1denoise_1.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.TVL1Deconv` usage (example 1)

    .. literalinclude:: ../../../examples/demo_tvl1deconv_1.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.TVL1Deconv` usage (example 2)

    .. literalinclude:: ../../../examples/demo_tvl1deconv_2.py
       :language: python
       :lines: 14-
