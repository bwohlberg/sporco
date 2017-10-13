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


* :class:`.BPDNProjL1`

  Solve the problem with :math:`\ell_2` objective and an
  :math:`\ell_1` constraint

  .. math::
     \mathrm{argmin}_\mathbf{x} \;
     (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 \; \text{such that} \;
     \| \mathbf{x} \|_1 \leq \gamma


* :class:`.MinL1InL2Ball`

  Solve the problem with :math:`\ell_1` objective and an
  :math:`\ell_2` constraint

    .. math::
       \mathrm{argmin}_\mathbf{x} \| \mathbf{x} \|_1 \; \text{such that} \;
       \| D \mathbf{x} - \mathbf{s} \|_2 \leq \epsilon



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


.. container:: toggle

    .. container:: header

        :class:`.BPDNProjL1` usage

    .. literalinclude:: ../../../examples/stdsparse/demo_bpdnl1prj.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.MinL1InL2Ball` usage

    .. literalinclude:: ../../../examples/stdsparse/demo_minl1.py
       :language: python
       :lines: 9-
