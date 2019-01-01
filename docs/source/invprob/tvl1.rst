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

:ref:`Usage examples <examples_tv_index>` are available.
