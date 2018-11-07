ADMM
====

The fundamental class from which all ADMM algorithms are derived is
:class:`.ADMM`, which supports problems of the form

.. math::
   \mathrm{argmin}_{\mathbf{x},\mathbf{y}} \;\;
   f(\mathbf{x}) + g(\mathbf{y}) \;\;\mathrm{such\;that}\;\;
   A\mathbf{x} + B\mathbf{y} = \mathbf{c} \;\;.


See :class:`.SplineL1` as an example of a class derived directly from
:class:`.ADMM`. Additional classes within the :mod:`.admm.admm` module
support less general forms of problem; this specialisation allows for
a smaller number of methods that have to be overriden in derived
classes.


Classes derived from :class:`.ADMM` should override/define the methods
and attributes in the following sections.


Initialisation
--------------

The ``__init__`` method of the derived class should call the
:class:`.ADMM` ``__init__`` method to ensure proper initialisation.

State variables :math:`\mathbf{y}` and :math:`\mathbf{u}` are
initialised to zero by :meth:`.ADMM.yinit` and :meth:`.ADMM.uinit`
respectively. These methods should be overridden if a different
initialization is desired.


.. _sec-admm-update-steps:

Update Steps
------------

The ADMM updates steps are defined by the following methods:


* :meth:`.ADMM.xstep`

  Solve

  .. math::
     \mathbf{x}^{(j+1)} = \mathrm{argmin}_{\mathbf{x}} \;\;
     f(\mathbf{x}) + \frac{\rho}{2} \left\| A\mathbf{x} -
     \left( -B\mathbf{y}^{(j)} + \mathbf{c} - \mathbf{u}^{(j)} \right)
     \right\|_2^2

  This method should set ``self.X`` as a function of ``self.Y`` and
  ``self.U``.


* :meth:`.ADMM.ystep`

  Solve

  .. math::
     \mathbf{y}^{(j+1)} = \mathrm{argmin}_{\mathbf{y}} \;\;
     g(\mathbf{y}) + \frac{\rho}{2} \left\| B\mathbf{y} - \left(
     -A\mathbf{x}^{(j+1)} + \mathbf{c} - \mathbf{u}^{(j)} \right)
     \right\|_2^2

  This method should set ``self.Y`` as a function of ``self.AX`` and
  ``self.U``. The use of ``self.AX`` (i.e. :math:`A
  \mathbf{x}^{(j+1)}` in the equation above) instead of ``self.X``
  (i.e. :math:`\mathbf{x}^{(j+1)}` in the equation above) is to allow
  relaxation (see Sec. 3.4.3 of :cite:`boyd-2010-distributed`), as
  implemented in :meth:`.ADMM.relax_AX`, to occur without need for
  explicit implementation in classes derived from :class:`.ADMM`.


* :meth:`.ADMM.ustep`

  Update dual variable

  .. math::
     \mathbf{u}^{(j+1)} = \mathbf{u}^{(j)} + A\mathbf{x}^{(j+1)} +
     B\mathbf{y}^{(j+1)} - \mathbf{c}

  This method should set ``self.U`` as a function of the previous
  value of ``self.U`` and ``self.X`` and ``self.Y``.

|

A derived class implementing a fully-specified ADMM problem (as
opposed to a partial specialisation) must define :meth:`.ADMM.xstep`
and :meth:`.ADMM.ystep`. It is usually not necessary to override
:meth:`.ADMM.ustep` since it is defined in :class:`.ADMM` in terms of
a call to :meth:`.ADMM.rsdl_r`: the :math:`\mathbf{u}` update is

.. math::
   \mathbf{u}^{(j+1)} = \mathbf{u}^{(j)} + A\mathbf{x}^{(j+1)} +
   B\mathbf{y}^{(j+1)} - \mathbf{c}

and the primal residual is

.. math::
   \mathbf{r}^{(j+1)} = A\mathbf{x}^{(j+1)} + B\mathbf{y}^{(j+1)} -
   \mathbf{c} \;,

so we can express the :math:`\mathbf{u}` update as

.. math::

   \mathbf{u}^{(j+1)} = \mathbf{u}^{(j)} + \mathbf{r}^{(j+1)} \;.

|

It is quite common for one of the update steps (:meth:`.ADMM.xstep` in
particular) to make use of pre-computed values, such as factorisations
of matrices involved in computing the update. If these pre-computed
values depend on the penalty parameter ``self.rho`` they need to be
recomputed when the penalty parameter changes when the ``AutoRho``
mechanism is enabled (see :meth:`.ADMM.update_rho`); this will take
place automatically if :meth:`.ADMM.rhochange` is overridden with a
method that updates these pre-computed values.



.. _sec-admm-constraint-def:

Constraint Definition
---------------------

The ADMM problem constraint is defined by the following methods, which
define the linear operators :math:`A` and :math:`B` and the transpose
of :math:`A` (required for computing the dual residual), and the
constant vector :math:`\mathbf{c}`:

* :meth:`.ADMM.cnst_A`

  Compute and return :math:`A \mathbf{x}`

* :meth:`.ADMM.cnst_AT`

  Compute and return :math:`A^T \mathbf{u}`

* :meth:`.ADMM.cnst_B`

  Compute and return :math:`B \mathbf{y}`

* :meth:`.ADMM.cnst_c`

  Return constant :math:`\mathbf{c}`

|

A derived class implementing a fully-specified ADMM problem (as
opposed to a partial specialisation) will usually define all of these
methods. If it does not, it is necessary to override all of the
methods in :ref:`sec-admm-residual-eval`.


.. _sec-admm-residual-eval:

Residual Evaluation
-------------------

The following methods support evaluation of the primal and dual
residuals:

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

|

These methods need not be overridden if those in
:ref:`sec-admm-constraint-def` are defined since :class:`.ADMM`
includes definitions of :meth:`.ADMM.rsdl_r`, :meth:`.ADMM.rsdl_s`,
:meth:`.ADMM.rsdl_rn`, and :meth:`.ADMM.rsdl_sn` in terms of
:meth:`.ADMM.cnst_A`, :meth:`.ADMM.cnst_AT`, :meth:`.ADMM.cnst_B`, and
:meth:`.ADMM.cnst_c`.


.. _sec-admm-iteration-stats:

Iteration Statistics
--------------------

There is a flexible but relatively complex mechanism supporting the
recording of statistics such as objective function and residual values
for each iteration.


IterationStats Definition
^^^^^^^^^^^^^^^^^^^^^^^^^

These statistics are recorded as a :func:`collections.namedtuple`
class, ``self.IterationStats``. The fields of this ``namedtuple`` are
defined by class method :meth:`.IterativeSolver.itstat_fields`, which
returns a tuple of fields consisting of the following components:

* ``Iter`` : Iteration number
* A tuple of field names in :attr:`.ADMM.itstat_fields_objfn` : Fields
  representing the objective function and and its individual terms
* ``PrimalRsdl`` : Norm of primal residual
* ``DualRsdl`` : Norm of dual Residual
* ``EpsPrimal`` : Primal residual stopping tolerance
  :math:`\epsilon_{\mathrm{pri}}`
* ``EpsDual`` : Dual residual stopping tolerance
  :math:`\epsilon_{\mathrm{dua}}`
* ``Rho`` : Penalty parameter
* A tuple of field names in :attr:`.ADMM.itstat_fields_extra` :
  Optional extra fields
* ``Time`` : Cumulative run time

In most cases a derived class will simply override
:attr:`.ADMM.itstat_fields_objfn` and
:attr:`.ADMM.itstat_fields_extra` to customise the desired iteration
statistics fields, but if more flexibility is required,
:meth:`.IterativeSolver.itstat_fields` should be overridden.


.. _sec-admm-itstat-cnstr:

IterationStats Construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The actual construction of the ``self.IterationStats`` ``namedtuple``
for each iteration is performed by :meth:`.ADMM.iteration_stats`,
which expects that ``self.IterationStats`` follows the structure
defined by :meth:`.IterativeSolver.itstat_fields`. Tuples of values
corresponding to the fields defined in
:attr:`.ADMM.itstat_fields_objfn` and
:attr:`.ADMM.itstat_fields_extra` should be returned by
:meth:`.ADMM.eval_objfn` and :meth:`.ADMM.itstat_extra` respectively.

In :class:`.ADMM`, :attr:`.ADMM.itstat_fields_objfn` is defined as the
tuple ``('ObjFun', 'FVal', 'GVal')``, and :meth:`.ADMM.eval_objfn`
constructs the corresponding field values by calls to
:meth:`.ADMM.obfn_f` and :meth:`.ADMM.obfn_g`, which are expected to
return the values of :math:`f(\mathbf{x})` and :math:`g(\mathbf{y})`
respectively. In the simplest case it is sufficient to just define
:meth:`.ADMM.obfn_f` and :meth:`.ADMM.obfn_g` in a derived class, but
in most cases one would instead override
:attr:`.ADMM.itstat_fields_objfn` and :meth:`.ADMM.eval_objfn` (and
possibly :attr:`.ADMM.itstat_fields_extra` and
:meth:`.ADMM.itstat_extra` as well).



Status Display
^^^^^^^^^^^^^^

When option ``Verbose`` is enabled, a summary of the iterations
statistics is printed to the standard output. The printing of this
summary is controlled by :meth:`.ADMM.display_start`,
:meth:`.ADMM.display_status`, and :meth:`.ADMM.display_end`, which
will usually *not* need to be overridden since there is a flexible
method of customising the information displayed by these methods.

Class method :meth:`.ADMM.hdrtxt` returns a tuple of strings which
will be displayed as the headings for the displayed columns of
iteration statistics, and class method :meth:`.ADMM.hdrval` constructs
a dictionary that defines a mapping between these column heading
strings and corresponding field names in the ``self.IterationStats``
``namedtuple``. These two methods can be overridden if necessary, but
in most cases it is sufficient to override :attr:`.ADMM.hdrtxt_objfn`
and :attr:`.ADMM.hdrval_objfun`, which respectively define the header
strings and mappings for the statistics related to the objective
function (see :attr:`.ADMM.itstat_fields_objfn` and
:meth:`.ADMM.eval_objfn` in :ref:`sec-admm-itstat-cnstr`).
