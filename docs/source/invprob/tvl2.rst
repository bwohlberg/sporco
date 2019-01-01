Module tvl2
===========

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
     (1/2) \| H \mathbf{x} - \mathbf{s} \|_2^2 +
     \lambda \left \| W_{\mathrm{tv}} \sqrt{(G_r \mathbf{x})^2 +
     (G_c \mathbf{x})^2} \right \|_1

:ref:`Usage examples <examples_tv_index>` are available.
