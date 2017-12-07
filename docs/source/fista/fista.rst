FISTA
=====

The fundamental class from which all FISTA algorithms are derived is
:class:`.FISTA`, which supports problems of the form

.. math::
   \mathrm{argmin}_{\mathbf{x}} \;
    f(\mathbf{x}) + g(\mathbf{x}) \;\;,

where :math:`f, g` are convex functions and :math:`f` is smooth. All the updates are made in input domain.

Additional classes within the :mod:`.fista.fista` module
support less general forms of problem; this specialisation allows for
a smaller number of methods that have to be overriden in derived
classes.

Classes derived from :class:`.FISTA` should override/define the methods
and attributes in the following sections.


Initialisation
--------------

The ``__init__`` method of the derived class should call
:meth:`.FISTA.__init__` to ensure proper initialisation.

State variable :math:`\mathbf{x}` is initialised to zero by :meth:`.FISTA.xinit`. This method should be overridden if a different initialization is desired.


.. _sec-fista-update-steps:

Update Steps
------------

The FISTA updates steps are defined by the following methods:


* :meth:`.FISTA.proximal_step`

  Compute

  .. math::
     \mathbf{x}^{(j+1)} = \mathrm{prox}_{t}(g) \left(
     \mathbf{y}^{(j)} - \frac{1}{L}\; \nabla_{\mathbf{x}} f(\mathbf{y}^{(j)}) \right) \;,

  where the proximal operation is defined as:

  .. math::
     \mathrm{prox}_{t}(g)\left( \mathbf{x} \right) = \mathrm{argmin}_{\mathbf{u}} \;\; g(\mathbf{u}) + \frac{1}{2 t} \left\| \mathbf{u} - \mathbf{x} \right\|_2^2 \; .


  This method should set ``self.X`` as a function of ``self.Y`` and
  gradient of :math:`f`.


* :meth:`.FISTA.combination_step`

  Compute

  .. math::
     \mathbf{y}^{(j+1)} = \mathbf{x}^{(j+1)} + \frac{t^{(j)} - 1}{t^{(j+1)}} \left( \mathbf{x}^{(j+1)} - \mathbf{x}^{(j)} \right) \;,


  with

  .. math::
     t^{(j+1)} = \frac{1}{2} \left ( 1 + \sqrt{1 + 4 \; (t^{(j)})^2} \right ) \;,


  starting with :math:`t^{(1)} = 1`.

  This method should set ``self.Y`` as a function of ``self.X`` and
  ``self.Xprv``.


* :meth:`.FISTA.compute_backtracking`

The bactracking process is an adaptive process to find the optimal step size for the gradient descent (:math:`L^{-1}`). Backtracking updates ``self.L`` until the condition :math:`F \leq Q_L` is satisfied. These are defined as 

.. math::
   F(\mathbf{x}) = f(\mathbf{x}) + g(\mathbf{x}) \;,

and 

.. math::
   Q_L(\mathbf{x},\mathbf{y}) = f(\mathbf{y}) + \langle \mathbf{x} - \mathbf{y}, \nabla f(\mathbf{y}) \rangle + \frac{L}{2} \left\| \mathbf{x} - \mathbf{y} \right\|_2^2 + g(\mathbf{x}) \;.

The backtracking process is optional. It is performed when the ``BackTrack``
mechanism is enabled. 



A derived class implementing a fully-specified FISTA problem (as
opposed to a partial specialisation) must define :meth:`.FISTA.eval_proxop`, :meth:`.FISTA.eval_grad` and :meth:`.FISTA.eval_R`. It is usually not necessary to override
:meth:`.FISTA.compute_backtracking` since it is defined in :class:`.FISTA` in terms of calls to :meth:`.FISTA.eval_grad`, :meth:`.FISTA.eval_R` and :meth:`.FISTA.proximal_step`. 


.. _sec-fista-residual-eval:

Residual Evaluation
-------------------

The following methods support evaluation of the residuals:

* :meth:`.FISTA.rsdl`

  This method has to be defined according to the stopping criterion to use. (It could be the relative difference between consecutive :math:`\mathbf{x}` iterates or a fixed point residual evaluating the difference between :math:`\mathbf{x}` and :math:`\mathbf{y}` states).


.. _sec-fista-iteration-stats:

Iteration Statistics
--------------------

There is a flexible but relatively complex mechanism supporting the
recording of statistics such as objective function and residual values
for each iteration.


IterationStats Definition
^^^^^^^^^^^^^^^^^^^^^^^^^

These statistics are recorded as a :func:`collections.namedtuple`
class, ``self.IterationStats``. The fields of this ``namedtuple`` are
defined by class method :meth:`.FISTA.itstat_fields`, which returns a
tuple of fields consisting of the following components:

* ``Iter`` : Iteration number
* A tuple of field names in :attr:`.FISTA.itstat_fields_objfn` : Fields
  representing the objective function and and its individual terms
* ``Rsdl`` : Norm of residual
* ``F_Btrack`` : Evaluation of :math:`F` (if backtracking is enabled)
* ``Q_Btrack`` : Evaluation of :math:`Q_L` (if backtracking is enabled)
* ``IterBTrack`` : Number of iterations used in backtracking (if backtracking is enabled)
* ``L`` : Inverse of gradient step size.
* A tuple of field names in :attr:`.FISTA.itstat_fields_extra` :
  Optional extra fields
* ``Time`` : Cumulative run time

In most cases a derived class will simply override
:attr:`.FISTA.itstat_fields_objfn` and
:attr:`.FISTA.itstat_fields_extra` to customise the desired iteration
statistics fields, but if more flexibility is required,
:meth:`.FISTA.itstat_fields` should be overridden.


.. _sec-fista-itstat-cnstr:

IterationStats Construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The actual construction of the ``self.IterationStats`` ``namedtuple``
for each iteration is performed by :meth:`.FISTA.iteration_stats`,
which expects that ``self.IterationStats`` follows the structure
defined by :meth:`.FISTA.itstat_fields`. Tuples of values corresponding
to the fields defined in :attr:`.FISTA.itstat_fields_objfn` and
:attr:`.FISTA.itstat_fields_extra` should be returned by
:meth:`.FISTA.eval_objfn` and :meth:`.FISTA.itstat_extra` respectively.

In :class:`.FISTA`, :attr:`.FISTA.itstat_fields_objfn` is defined as the
tuple ``('ObjFun', 'FVal', 'GVal')``, and :meth:`.FISTA.eval_objfn`
constructs the corresponding field values by calls to
:meth:`.FISTA.obfn_f` and :meth:`.FISTA.obfn_g`, which are expected to
return the values of :math:`f(\mathbf{x})` and :math:`g(\mathbf{x})`
respectively. In the simplest case it is sufficient to just define
:meth:`.FISTA.obfn_f` and :meth:`.FISTA.obfn_g` in a derived class, but
in most cases one would instead override
:attr:`.FISTA.itstat_fields_objfn` and :meth:`.FISTA.eval_objfn` (and
possibly :attr:`.FISTA.itstat_fields_extra` and
:meth:`.FISTA.itstat_extra` as well).



Status Display
^^^^^^^^^^^^^^

When option ``Verbose`` is enabled, a summary of the iterations
statistics is printed to the standard output. The printing of this
summary is controlled by :meth:`.FISTA.display_start`,
:meth:`.FISTA.display_status`, :meth:`.FISTA.display_end`, and the state of the ``BackTrack`` flag. These methods
will usually *not* need to be overridden since there is a flexible
method of customising the information displayed by these methods.

Class method :meth:`.FISTA.hdrtxt` returns a tuple of strings which
will be displayed as the headings for the displayed columns of iteration
statistics, and class method :meth:`.FISTA.hdrval` constructs a
dictionary that defines a mapping between these column heading strings
and corresponding field names in the ``self.IterationStats``
``namedtuple``. These two methods can be overridden if necessary, but
in most cases it is sufficient to override :attr:`.FISTA.hdrtxt_objfn`
and :attr:`.FISTA.hdrval_objfun`, which respectively define the header
strings and mappings for the statistics related to the objective
function (see :attr:`.FISTA.itstat_fields_objfn` and
:meth:`.FISTA.eval_objfn` in :ref:`sec-fista-itstat-cnstr`).


