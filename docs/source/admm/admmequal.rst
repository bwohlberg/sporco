ADMMEqual
=========

:class:`.ADMMEqual` specialises :class:`.ADMM` to the case
:math:`A = I`, :math:`B = -I`, and  :math:`\mathbf{c} = \mathbf{0}`,
for solving problems of the form

.. math::
   \mathrm{argmin}_{\mathbf{x},\mathbf{y}} \;
   f(\mathbf{x}) + g(\mathbf{y}) \;\mathrm{such\;that}\;
   \mathbf{x} = \mathbf{y} \;\;.

See :class:`.admm.bpdn.BPDN` as an example of a class derived from
:class:`.ADMMEqual`.

The methods and attributes that should be overridden or defined are
largely the same is described in :doc:`admm`, except that the
specialisation removes the need for derived classes to override
:meth:`.ADMMEqual.cnst_A`, :meth:`.ADMMEqual.cnst_AT`,
:meth:`.ADMMEqual.cnst_B`, and :meth:`.ADMMEqual.cnst_c`. These
differences are described in the following sections.


Update Steps
------------

The ADMM updates steps in :class:`.ADMMEqual` are effectively
(via the constraint definition methods):


* :meth:`.ADMM.xstep`

  Solve

  .. math::
     \mathbf{x}^{(j+1)} = \mathrm{argmin}_{\mathbf{x}} \;\;
     f(\mathbf{x}) + \frac{\rho}{2} \left\| \mathbf{x} -
     \left(\mathbf{y}^{(j)} - \mathbf{u}^{(j)} \right)
     \right\|_2^2


* :meth:`.ADMM.ystep`

  Solve

  .. math::
     \mathbf{y}^{(j+1)} = \mathrm{argmin}_{\mathbf{y}} \;\;
     g(\mathbf{y}) + \frac{\rho}{2} \left\| \mathbf{y} - \left(
     \mathbf{x}^{(j+1)} + \mathbf{u}^{(j)} \right)
     \right\|_2^2


* :meth:`.ADMM.ustep`

  Update dual variable

  .. math::
     \mathbf{u}^{(j+1)} = \mathbf{u}^{(j)} + \mathbf{x}^{(j+1)} -
     \mathbf{y}^{(j+1)}

|

As discussed in :ref:`sec-admm-update-steps`, it is usually not
necessary to override :meth:`.ADMM.ustep`.




Residual Evaluation
-------------------

The residual evaluation methods in :class:`.ADMMEqual` are effectively
(via the constraint definition methods):

* :meth:`.ADMM.rsdl_r`

  Compute primal residual

  .. math::
     \mathbf{r} = \mathbf{x}^{(j+1)} - \mathbf{y}^{(j+1)}


* :meth:`.ADMM.rsdl_s`

  Compute dual residual

  .. math::
     \mathbf{s} = \rho (\mathbf{y}^{(j)} - \mathbf{y}^{(j+1)})


* :meth:`.ADMM.rsdl_rn`

  Compute primal residual normalisation factor

  .. math::
     \mathrm{rn} = \mathrm{max}(\|\mathbf{x}^{(j+1)}\|_2,
     \|\mathbf{y}^{(j+1)}\|_2)


* :meth:`.ADMM.rsdl_sn`

  Compute dual residual normalisation factor

  .. math::
     \mathrm{sn} = \rho \|\mathbf{u}^{(j+1)} \|_2

|

These methods need not be overridden since the constraint definition
methods in :ref:`sec-admm-constraint-def` are defined in
:class:`.ADMMEqual`.
