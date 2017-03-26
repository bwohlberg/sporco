Module tvl1
===========

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
     \| W_{\mathrm{df}} (H \mathbf{x} - \mathbf{s}) \|_1 +
     \lambda \left \| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
     (G_c \mathbf{x})^2} \right \|_1



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.TVL1Denoise` usage (greyscale image)

    .. literalinclude:: ../../../examples/misc/demo_tvl1den_gry.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.TVL1Denoise` usage (colour image)

    .. literalinclude:: ../../../examples/misc/demo_tvl1den_clr.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.TVL1Deconv` usage (denoising problem)

    .. literalinclude:: ../../../examples/misc/demo_tvl1dcn_den.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.TVL1Deconv` usage (deconvolution problem, greyscale image)

    .. literalinclude:: ../../../examples/misc/demo_tvl1dcn_dcn_gry.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.TVL1Deconv` usage (deconvolution problem, colour image)

    .. literalinclude:: ../../../examples/misc/demo_tvl1dcn_dcn_clr.py
       :language: python
       :lines: 9-
