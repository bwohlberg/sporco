Module bpdn
===========

This module includes the following classes:

* :class:`.BPDN`

  Solve the BPDN problem

  .. math::
     \mathrm{argmin}_\mathbf{x} \;
     (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1


* :class:`.BPDNJoint`

  Solve the BPDN problem with an additional :math:`\ell_{2,1}` norm
  term to promote joint sparsity

  .. math::
     \mathrm{argmin}_X \; (1/2) \| D X - S \|_2^2 + \lambda \| X \|_1
     + \mu \| X \|_{2,1}


* :class:`.ElasticNet`

  Solve the Elastic Net problem

  .. math::
     \mathrm{argmin}_\mathbf{x} \;
     (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1
     + (\mu/2) \| \mathbf{x} \|_2^2



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.BPDN` usage

    .. literalinclude:: ../../../examples/stdsparse/demo_bpdn.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.BPDNJoint` usage

    .. literalinclude:: ../../../examples/stdsparse/demo_bpdnjnt.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.ElasticNet` usage

    .. literalinclude:: ../../../examples/stdsparse/demo_elnet.py
       :language: python
       :lines: 9-
