PGM
=====

The fundamental class from which all PGM algorithms are derived is
:class:`.PGM`, which supports problems of the form

.. math::
   \mathrm{argmin}_{\mathbf{x}} \; f(\mathbf{x}) + g(\mathbf{x}) \;\;,

where :math:`f, g` are convex functions and :math:`f` is smooth. All
the updates are made in input domain.

Additional classes within the :mod:`.pgm.pgm` module support less
general forms of problem; this specialisation allows for a smaller
number of methods that have to be overriden in derived classes.

Classes derived from :class:`.PGM` should override/define the
methods and attributes in the following sections.

See :class:`.pgm.bpdn.BPDN` as an example of a class derived
from :class:`.PGM`.


Initialisation
--------------

The ``__init__`` method of the derived class should call the
:class:`.PGM` ``__init__`` method to ensure proper initialisation.

State variable :math:`\mathbf{x}` is initialised to zero by
:meth:`.PGM.xinit`. This method should be overridden if a different
initialization is desired.


.. _sec-pgm-update-steps:

Update Steps
------------

The PGM updates steps are defined by the following methods:


* :meth:`.PGM.xstep`

  Compute

  .. math::
     \mathbf{x}^{(j+1)} = \mathrm{prox}_{t}(g) \left(
     \mathbf{y}^{(j)} - \frac{1}{L}\; \nabla_{\mathbf{x}} f(\mathbf{y}^{(j)})
     \right) \;,

  where the proximal operation is defined as:

  .. math::
     \mathrm{prox}_{t}(g)\left( \mathbf{x} \right) =
     \mathrm{argmin}_{\mathbf{u}} \;\; g(\mathbf{u}) + \frac{1}{2 t}
     \left\| \mathbf{u} - \mathbf{x} \right\|_2^2 \; .


  This method should set ``self.X`` as a function of ``self.Y`` and
  gradient of :math:`f`.

  Optionally, a monotone PGM version from :cite:`beck-2009-tv` is available.
  The monotone version evaluates the functional value at the new update :math:`\mathbf{x}^{(j+1)}`. If this update causes an increment on the objective function :math:`f(\mathbf{x}) + g(\mathbf{x})` then it will revert to the previous iterate :math:`\mathbf{x}^{(j)}`. The :meth:`.PGM.ystep` is also modified in monotone PGM.


* :meth:`.PGM.ystep`

  Compute

  .. math::
     \mathbf{y}^{(j+1)} = \mathbf{x}^{(j+1)} + \frac{t^{(j)} - 1}{t^{(j+1)}}
     \left( \mathbf{x}^{(j+1)} - \mathbf{x}^{(j)} \right) \;,

  with

  .. math::
     t^{(j+1)} = \frac{1}{2} \left( 1 + \sqrt{1 + 4 \; (t^{(j)})^2}
     \right) \;,

  starting with :math:`t^{(1)} = 1`. This corresponds to the Nesterov's momentum update. Other updates of the momentum coefficient :math:`t^{(j+1)}` are available by deriving from :class:`.MomentumBase`.

  This method should set ``self.Y`` as a function of ``self.X`` and
  ``self.Xprv``. (This method is part of the standard PGM formulation
  :cite:`beck-2009-fast`.)

  The update for the monotone PGM version from :cite:`beck-2009-tv` corresponds to 

  .. math::
     \mathbf{y}^{(j+1)} = \mathbf{x}^{(j+1)} + \frac{t^{(j)}}{t^{(j+1)}}
     \left( \mathbf{z}^{(j+1)} - \mathbf{x}^{(j+1)} \right) + \frac{t^{(j)} - 1}{t^{(j+1)}}
     \left( \mathbf{x}^{(j+1)} - \mathbf{x}^{(j)} \right) \;,

  with :math:`\mathbf{z}^{(j+1)}` the proximal mapping computed in :meth:`.PGM.xstep` (which may not correspond to :math:`\mathbf{x}^{(j+1)}` if it has been replaced by the previous iterate due to the monotone restriction).


A derived class implementing a fully-specified PGM problem (as
opposed to a partial specialisation) must define
:meth:`.PGM.prox_g`, :meth:`.PGM.grad_f`,
:meth:`.PGM.obfn_f` and :meth:`.PGM.obfn_g`. 

.. _sec-pgm-residual-eval:

Residual Evaluation
-------------------

The following methods support evaluation of the residuals:

* :meth:`.PGM.rsdl`

  This method has to be defined according to the stopping criterion to
  use. (It could be the relative difference between consecutive
  :math:`\mathbf{x}` iterates or a fixed point residual evaluating the
  difference between :math:`\mathbf{x}` and :math:`\mathbf{y}`
  states).


.. _sec-pgm-iteration-stats:

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
* A tuple of field names in :attr:`.PGM.itstat_fields_objfn` : Fields
  representing the objective function and and its individual terms
* ``Rsdl`` : Norm of residual
* ``F_Btrack`` : Evaluation of :math:`F` (if backtrack is enabled)
* ``Q_Btrack`` : Evaluation of :math:`Q_L` (if backtrack is enabled)
* ``IterBTrack`` : Number of iterations used in backtrack (if backtrack is enabled)
* ``L`` : Inverse of gradient step size.
* A tuple of field names in :attr:`.PGM.itstat_fields_extra` :
  Optional extra fields
* ``Time`` : Cumulative run time

In most cases a derived class will simply override
:attr:`.PGM.itstat_fields_objfn` and
:attr:`.PGM.itstat_fields_extra` to customise the desired iteration
statistics fields, but if more flexibility is required,
:meth:`.IterativeSolver.itstat_fields` should be overridden.


.. _sec-pgm-itstat-cnstr:

IterationStats Construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The actual construction of the ``self.IterationStats`` ``namedtuple``
for each iteration is performed by :meth:`.PGM.iteration_stats`,
which expects that ``self.IterationStats`` follows the structure
defined by :meth:`.IterativeSolver.itstat_fields`. Tuples of values
corresponding to the fields defined in
:attr:`.PGM.itstat_fields_objfn` and
:attr:`.PGM.itstat_fields_extra` should be returned by
:meth:`.PGM.eval_objfn` and :meth:`.PGM.itstat_extra`
respectively.

In :class:`.PGM`, :attr:`.PGM.itstat_fields_objfn` is defined as
the tuple ``('ObjFun', 'FVal', 'GVal')``, and
:meth:`.PGM.eval_objfn` constructs the corresponding field values by
calls to :meth:`.PGM.obfn_f` and :meth:`.PGM.obfn_g`, which are
expected to return the values of :math:`f(\mathbf{x})` and
:math:`g(\mathbf{x})` respectively. In the simplest case it is
sufficient to just define :meth:`.PGM.obfn_f` and
:meth:`.PGM.obfn_g` in a derived class, but in most cases one would
instead override :attr:`.PGM.itstat_fields_objfn` and
:meth:`.PGM.eval_objfn` (and possibly
:attr:`.PGM.itstat_fields_extra` and :meth:`.PGM.itstat_extra` as
well).



Status Display
^^^^^^^^^^^^^^

When option ``Verbose`` is enabled, a summary of the iterations
statistics is printed to the standard output. The printing of this
summary is controlled by :meth:`.PGM.display_start`,
:meth:`.PGM.display_status`, :meth:`.PGM.display_end`, and the
state of the ``Backtrack`` auxiliary class. These methods will usually *not* need
to be overridden since there is a flexible method of customising the
information displayed by these methods.

Class method :meth:`.PGM.hdrtxt` returns a tuple of strings which
will be displayed as the headings for the displayed columns of
iteration statistics, and class method :meth:`.PGM.hdrval`
constructs a dictionary that defines a mapping between these column
heading strings and corresponding field names in the
``self.IterationStats`` ``namedtuple``. These two methods can be
overridden if necessary, but in most cases it is sufficient to
override :attr:`.PGM.hdrtxt_objfn` and :attr:`.PGM.hdrval_objfun`,
which respectively define the header strings and mappings for the
statistics related to the objective function (see
:attr:`.PGM.itstat_fields_objfn` and :meth:`.PGM.eval_objfn` in
:ref:`sec-pgm-itstat-cnstr`).
