ADMMConsensus
=============

:class:`.ADMMConsensus` specialises :class:`.ADMM` for solving
optimisation problems of the form

 .. math::
    \mathrm{argmin}_{\mathbf{x}} \; \sum_{i = 0}^{N_b - 1}
    f_i(\mathbf{x}) + g(\mathbf{x})

via an ADMM problem of the form

.. math::
   \mathrm{argmin}_{\mathbf{x}_i,\mathbf{y}} \;
   \sum_{i = 0}^{N_b - 1} f(\mathbf{x}_i) + g(\mathbf{y})
   \;\mathrm{such\;that}\;
   \left( \begin{array}{c} \mathbf{x}_0 \\ \mathbf{x}_1 \\
   \vdots \end{array} \right) = \left( \begin{array}{c}
   I \\ I \\ \vdots \end{array} \right) \mathbf{y} \;\;.

See :class:`.ConvCnstrMOD_Consensus` as an example of a class derived
from :class:`.ADMMConsensus`, or see the simple :ref:`usage example <examples_misc_admmcnsns>`.

Classes derived from :class:`.ADMMConsensus` should override/define
the methods and attributes in the following sections.


Initialisation
--------------

The ``__init__`` method of the derived class should call the
:class:`.ADMMConsensus` ``__init__`` method to ensure proper
initialisation. Note that this method assumes that the ADMM consensus
component blocks in working variable :math:`\mathbf{x}` will be stacked
on the final array index, and defines attribute ``self.xshape``
accordingly.

State variables :math:`\mathbf{y}` and :math:`\mathbf{u}` are
initialised to zero by inherited methods :meth:`.ADMM.yinit` and
:meth:`.ADMM.uinit` respectively (this behaviour is inherited from
:class:`.ADMM`). These methods should be overridden if a different
initialization is desired.



Update Steps
------------

The :math:`\mathbf{x}` update method :meth:`.ADMMConsensus.xstep`
calls :meth:`.ADMMConsensus.xistep` for each ADMM consensus component
block. In most cases a derived class will define
:meth:`.ADMMConsensus.xistep` rather than override
:meth:`.ADMMConsensus.xstep`. Method :meth:`.ADMMConsensus.xistep`
should solve

  .. math::
     \mathbf{x}_i^{(j+1)} = \mathrm{argmin}_{\mathbf{x}_i} \;\;
     f_i(\mathbf{x}_i) + \frac{\rho}{2} \left\| \mathbf{x}_i -
     \left( \mathbf{y}^{(j)} - \mathbf{u}_i^{(j)} \right)
     \right\|_2^2

setting a slice of ``self.X`` on the final index from the result.
   
The :math:`\mathbf{y}` update method :meth:`.ADMMConsensus.ystep`
solves

  .. math::
     \mathbf{y}^{(j+1)} = \mathrm{argmin}_{\mathbf{y}} \;\;
     g(\mathbf{y}) + \frac{N_b \rho}{2} \left\| \mathbf{y} -
     \mathbf{z}^{(j)} \right\|_2^2

where

  .. math::
     \mathbf{z}^{(j)} = \sum_{i = 0}^{N_b - 1} \left(
     \mathbf{x}_i^{(j+1)} + \mathbf{u}_i^{(j)} \right) \;.
     
A class derived from :class:`.ADMMConsensus` should override
:meth:`.ADMMConsensus.prox_g` to implement the proximal operator of
:math:`g(\cdot)`. Note that :math:`N_b \rho` is passed as a parameter
to :meth:`.ADMMConsensus.prox_g`; it is the responsibility of the
implementer of this method to understand that it implements what is in
mathematical terms the proximal operator of :math:`g(\cdot)` with
parameter :math:`(N_b \rho)^{-1}`.

The dual variable update is

 .. math::
     \mathbf{u}_i^{(j+1)} = \mathbf{u}_i^{(j)} + \mathbf{x}_i^{(j+1)} -
     \mathbf{y}^{(j+1)} \;.

This update is implemented in :meth:`.ADMM.ustep`, which will usually
not need to be overridden.

|

As in :class:`.ADMM`, if one of the update steps makes use of
pre-computed values that depend on the penalty parameter ``self.rho``,
:meth:`.ADMM.rhochange` should be with a method that updates these
pre-computed values.



Constraint Definition
---------------------

Class :class:`.ADMMConsensus` overrides all of the methods in
:ref:`sec-admm-residual-eval` and does not define any of the ADMM
constraint definition methods discussed in
:ref:`sec-admm-constraint-def`.



Residual Evaluation
-------------------

The residual evaluation methods :meth:`.ADMMConsensus.rsdl_r`,
:meth:`.ADMMConsensus.rsdl_s`, :meth:`.ADMMConsensus.rsdl_rn`,
:meth:`.ADMMConsensus.rsdl_sn` are all appropriately defined for a
general ADMM consensus problem, and will typically not need to be
overridden.


Iteration Statistics
--------------------

The iteration statistics mechanism, as described in
:ref:`sec-admm-iteration-stats`, is inherited largely unchanged from
:class:`.ADMM`. The only exception is that
:meth:`.ADMMConsensus.obfn_f` is defined to evaluate the sum over
class for each ADMM consensus block to :meth:`.ADMMConsensus.obfn_fi`,
which should be overridden in a derived class if it is desired to use
this simple iteration statistics mechanism rather than override
:meth:`.ADMM.eval_objfn`.
