bpdn
====

This module includes the :class:`.BPDN` and :class:`.ElasticNet`
classes, solving the following problems:

* :class:`.BPDN`

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1

    or

    .. math::
       \mathrm{argmin}_X \;
       (1/2) \| D X - S \|_F^2 + \lambda \| X \|_1

* :class:`.ElasticNet`

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1
                                                 + (\mu/2) \| \mathbf{x} \|_2^2

    or

    .. math::
       \mathrm{argmin}_X \;
       (1/2) \| D X - S \|_F^2 + \lambda \| X \|_1 + (\mu/2) \| X \|_2^2




Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.BPDN` usage

    .. literalinclude:: ../../../examples/demo_bpdn.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.ElasticNet` usage

    .. literalinclude:: ../../../examples/demo_elnet.py
       :language: python
       :lines: 14-
