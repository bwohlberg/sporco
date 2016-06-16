ADMM Algorithms
===============

ADMM algorithm development is supported by a base class from which
classes for specific optimisation problems may be derived. The base
class provides the machinery for managing iterations and stopping
conditions, computing residuals, etc. so that only methods providing
the details of a specific problem need to be defined in a derived
class.

:class:`.ADMM`
--------------

The fundamental class from which all ADMM algorithms are derived is
:class:`.ADMM`, which supports problems of the form 

.. math::
   \mathrm{argmin}_{\mathbf{x},\mathbf{y}} \;\;
   f(\mathbf{x}) + g(\mathbf{y}) \;\mathrm{such\;that}\;
   A\mathbf{x} + B\mathbf{y} = \mathbf{c} \;\;.

Derived classes should override the methods:

* :meth:`.ADMM.xstep`

  Solve

  .. math::
     \mathbf{x}^{(j+1)} = \mathrm{argmin}_{\mathbf{x}} \;\;
     f(\mathbf{x}) + \frac{\rho}{2} \left\| A\mathbf{x} -
     \left( -B\mathbf{y}^{(j)} + \mathbf{c} - \mathbf{u}^{(j)} \right) 
     \right\|_2^2 

* :meth:`.ADMM.ystep`

  Solve

  .. math::
     \mathbf{y}^{(j+1)} = \mathrm{argmin}_{\mathbf{y}} \;\;
     g(\mathbf{y}) + \frac{\rho}{2} \left\| B\mathbf{y} - \left(
     -A\mathbf{x}^{(j+1)} + \mathbf{c} - \mathbf{u}^{(j)} \right)
     \right\|_2^2

* :meth:`.ADMM.ustep`

  Update dual variable

  .. math::
     \mathbf{u}^{(j+1)} = \mathbf{u}^{(j)} + A\mathbf{x}^{(j+1)} +
     B\mathbf{y}^{(j+1)} - \mathbf{c}

as well as either

* :meth:`.ADMM.iteration_stats`

  Construct a :py:func:`collections.namedtuple` recording
  iterations statistics such as objective function value and the norms
  of the primal and dual residuals.

or both of

* :meth:`.ADMM.obfn_f`

  Compute the value of :math:`f(\mathbf{x})`

* :meth:`.ADMM.obfn_g`

  Compute the value of :math:`g(\mathbf{y})`

as well as either all of:

* :meth:`.ADMM.cnst_A`

  Compute :math:`A \mathbf{x}`

* :meth:`.ADMM.cnst_AT`

  Compute :math:`A^T \mathbf{u}`

* :meth:`.ADMM.cnst_B`

  Compute :math:`B \mathbf{y}`

* :meth:`.ADMM.cnst_c`

  Return constant :math:`\mathbf{c}`

or all of:

* :meth:`.ADMM.rsdl_r`

  Compute primal residual

  .. math::
     \mathbf{r} = A\mathbf{x}^{(j+1)} + B\mathbf{y}^{(j+1)} - \mathbf{c}

* :meth:`.ADMM.rsdl_s`

  Compute dual residual

  .. math::
     \mathbf{s} = \rho A^T B (\mathbf{y}^{(j+1)} - \mathbf{y}^{(j)})

* :meth:`.ADMM.rsdl_rn`

  Compute primal residual normalisation factor

  .. math::
     \mathrm{rn} = \mathrm{max}(\|A\mathbf{x}^{(j+1)}\|_2,
     \|B\mathbf{y}^{(j+1)}\|_2, \|\mathbf{c}\|_2)

* :meth:`.ADMM.rsdl_sn`

  Compute dual residual normalisation factor

  .. math::
     \mathrm{sn} = \rho \|A^T \mathbf{u}^{(j+1)} \|_2


See :class:`.SplineL1` as an example of a class derived directly from
:class:`.ADMM`. Additional classes within the :mod:`.admm.admm` module
support less general forms of problem; this specialisation allows for
a smaller number of methods that have to be overriden in derived
classes.



:class:`.ADMMEqual`
-------------------

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
