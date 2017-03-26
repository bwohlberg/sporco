ADMMEqual
=========

:class:`.ADMMEqual` specialises :class:`.ADMM` to the case
:math:`A = I`, :math:`B = -I`, and  :math:`\mathbf{c} = \mathbf{0}`,
for solving problems of the form

.. math::
   \mathrm{argmin}_{\mathbf{x},\mathbf{y}} \;
   f(\mathbf{x}) + g(\mathbf{y}) \;\mathrm{such\;that}\;
   \mathbf{x} = \mathbf{y} \;\;.

This specialisation removes the need for derived classes to override
:meth:`.ADMMEqual.cnst_A`, :meth:`.ADMMEqual.cnst_AT`,
:meth:`.ADMMEqual.cnst_B`, and :meth:`.ADMMEqual.cnst_c`.

See :class:`.BPDN` as an example of a class derived from :class:`.ADMMEqual`.
