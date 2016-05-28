cbpdn
=====

This module includes the :class:`.ConvBPDN` and :class:`.ConvElasticNet`
classes, solving the following problems:

* :class:`.ConvBPDN`

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \|  \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2 +
       \lambda \sum_m \| \mathbf{x}_m \|_1

    or

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \sum_k \| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
       \mathbf{s}_k \|_2^2 + \lambda \sum_k \sum_m \| \mathbf{x}_{k,m} \|_1


* :class:`.ConvElasticNet`

   .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \|  \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2 +
       \lambda \sum_m \| \mathbf{x}_m \|_1 +
       (\mu/2) \sum_m \| \mathbf{x}_m \|_2^2

   or

   .. math::
      \mathrm{argmin}_\mathbf{x} \;
      (1/2) \sum_k \| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
      \mathbf{s}_k \|_2^2 + \lambda \sum_k \sum_m \| \mathbf{x}_{k,m} \|_1
      + (\mu/2) \sum_k \sum_m \| \mathbf{x}_{k,m} \|_2^2



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.ConvBPDN` usage (greyscale image)

    .. literalinclude:: ../../../examples/demo_cbpdn_gry.py
       :language: python
       :lines: 14-

.. container:: toggle

    .. container:: header

        :class:`.ConvBPDN` usage (colour image)

    .. literalinclude:: ../../../examples/demo_cbpdn_clr.py
       :language: python
       :lines: 14-

.. container:: toggle

    .. container:: header

        :class:`.ConvElasticNet` usage (greyscale image)

    .. literalinclude:: ../../../examples/demo_celnet_gry.py
       :language: python
       :lines: 14-

.. container:: toggle

    .. container:: header

        :class:`.ConvElasticNet` usage (colour image)

    .. literalinclude:: ../../../examples/demo_celnet_clr.py
       :language: python
       :lines: 14-
