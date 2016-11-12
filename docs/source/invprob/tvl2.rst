tvl2
====

This module includes the following classes:

* :class:`.TVL2Denoise`

  Solve the :math:`\ell_2`-TV denoising problem

  .. math::
     \mathrm{argmin}_\mathbf{x} \;
     (1/2) \| W_{\mathrm{df}}  (\mathbf{x} - \mathbf{s}) \|_2^2 +
     \lambda \left \| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
     (G_c \mathbf{x})^2} \right \|_1

* :class:`.TVL2Deconv`

  Solve the :math:`\ell_2`-TV deconvolution problem

  .. math::
     \mathrm{argmin}_\mathbf{x} \;
     (1/2) \| A * \mathbf{x} - \mathbf{s} \|_2^2 +
     \lambda \left \| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
     (G_c \mathbf{x})^2} \right \|_1



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.TVL2Denoise` usage

    .. literalinclude:: ../../../examples/misc/demo_tvl2denoise.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.TVL2Deconv` usage (denoising problem)

    .. literalinclude:: ../../../examples/misc/demo_tvl2deconv_den.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.TVL2Deconv` usage (deconvolution problem)

    .. literalinclude:: ../../../examples/misc/demo_tvl2deconv_dcn.py
       :language: python
       :lines: 9-
