tvl1
====

This module includes the following classes:

* :class:`.TVL1Denoise`

  Solve the :math:`\ell_1`-TV denoising problem

  .. math::
     \mathrm{argmin}_\mathbf{x} \;
     \| W_{\mathrm{df}}  (\mathbf{x} - \mathbf{s}) \|_1 +
     \lambda \left \| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 + 
     (G_c \mathbf{x})^2} \right \|_1

* :class:`.TVL1Deconv`

  Solve the :math:`\ell_1`-TV deconvolution problem

  .. math::
     \mathrm{argmin}_\mathbf{x} \;
     \| A * \mathbf{x} - \mathbf{s} \|_1 +
     \lambda \left \| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
     (G_c \mathbf{x})^2} \right \|_1



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.TVL1Denoise` usage

    .. literalinclude:: ../../../examples/misc/demo_tvl1denoise.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.TVL1Deconv` usage (denoising problem)

    .. literalinclude:: ../../../examples/misc/demo_tvl1deconv_den.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.TVL1Deconv` usage (deconvolution problem)

    .. literalinclude:: ../../../examples/misc/demo_tvl1deconv_dcn.py
       :language: python
       :lines: 9-
