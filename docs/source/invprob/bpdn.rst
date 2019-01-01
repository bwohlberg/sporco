Module bpdn
===========

This module includes the following classes:

* :class:`.admm.bpdn.BPDN`

  Solve the BPDN problem

  .. math::
     \mathrm{argmin}_\mathbf{x} \;
     (1/2) \| D \mathbf{x} - \mathbf{s} \|_2^2 + \lambda \| \mathbf{x} \|_1
     :label: bpdn


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
     :label: lasso


* :class:`.MinL1InL2Ball`

  Solve the problem with :math:`\ell_1` objective and an
  :math:`\ell_2` constraint

  .. math::
     \mathrm{argmin}_\mathbf{x} \| \mathbf{x} \|_1 \; \text{such that} \;
     \| D \mathbf{x} - \mathbf{s} \|_2 \leq \epsilon
     :label: minl1inl2


:ref:`Usage examples <examples_sc_index>` are available.


A Note on Problem Naming
------------------------

Unfortunately there is no consistent use of names for problems :eq:`bpdn`, :eq:`lasso`, and :eq:`minl1inl2` in the literature. Problem :eq:`bpdn` is referred to here as Basis Pursuit DeNoising (BPDN) since this is the form of the problem to which this name was first applied :cite:`chen-1998-atomic`, but one can also find problem :eq:`minl1inl2` referred to as BPDN, and problem :eq:`bpdn` referred to as the lasso, which is the name that was originally applied to problem :eq:`lasso` :cite:`tibshirani-1996-regression`.
