StepSizePolicyBase
==================

The fundamental class from which all algorithms for adapting the step size are derived is :class:`.StepSizePolicyBase`.

The step size for the gradient descent (:math:`L^{-1}`) can be adapted to exploit information from past iterates and improve convergence. Different alternatives for computing the step size update yield different gradient algorithms. In contrast with the backtrack update, the step size update implemented in this class is not iterative and does not take into account the potential improvement of the functional value.

The step size adaptation process is optional. It is performed when the
``StepSizePolicy`` auxiliary class is enabled. A fixed step size is used by default.

Classes derived from :class:`.StepSizePolicyBase` should override/define the
method :meth:`.StepSizePolicyBase.update`. The step size adaptation is defined
in terms of calls to :meth:`.PGM.grad_f` and :meth:`.PGM.hessian_f` (the latter is necessary for Cauchy-based methods). Note that whenever both backtrack and step size classes are enabled, the backtrack class takes precedence.

.. _sec-stepsize-classes:


Step Size Classes
-----------------

The step size adaptation functionality is defined by the following classes:


* :class:`.StepSizePolicyCauchy`

 This implements the step size adaptation based on Cauchy's method as

.. math::
   \alpha = \frac{\left\| \nabla f \right\|_2^2}
       {\nabla f^T \: \mathrm{Hess}_f \nabla f} \; ,

with :math:`L = 1 / \alpha`.


* :class:`.StepSizePolicyBB`

 This implements the step size adaptation based on Barzilai-Borwein method from :cite:`barzilai-1988-stepsize` as

.. math::
   \Delta x = x_k - x_{k-1} \; \\
   \Delta g = \nabla f (x_k) - \nabla f (x_{k-1}) \; \\
   \alpha = \frac{\Delta x^T \Delta g}{\left\| \Delta g \right\|_2^2} \; ,

with :math:`L = 1 / \alpha`.

