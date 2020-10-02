BacktrackBase
=============

The fundamental class from which all backtrack algorithms are derived is
:class:`.BacktrackBase`.

The backtracking process is an adaptive process to find the optimal
step size for the gradient descent (:math:`L^{-1}`). Backtracking
updates ``self.L`` until the condition :math:`F \leq Q_L` is
satisfied. These are defined as

.. math::
   F(\mathbf{x}) = f(\mathbf{x}) + g(\mathbf{x}) \;,

and

.. math::
   Q_L(\mathbf{x},\mathbf{y}) = f(\mathbf{y}) + \langle \mathbf{x} -
   \mathbf{y}, \nabla f(\mathbf{y}) \rangle + \frac{L}{2} \left\|
   \mathbf{x} - \mathbf{y} \right\|_2^2 + g(\mathbf{x}) \;.

The backtracking process is optional. It is performed when the
``Backtrack`` auxiliary class is enabled.

Classes derived from :class:`.BacktrackBase` should override/define the
method :meth:`.BacktrackBase.update`. The backtrack functionality is defined
in terms of calls to :meth:`.PGM.grad_f`,
:meth:`.PGM.prox_g` and :meth:`.PGM.obfn_f`. Note that whenever both backtrack and step size classes are enabled, the backtrack class takes precedence.

.. _sec-backtrack-classes:


Backtrack Classes
-----------------

The PGM backtrack functionality is defined by the following classes:


* :class:`.BacktrackStandard`

 This implements the standard PGM variant from :cite:`beck-2009-fast`.

* :class:`.BacktrackRobust`

 This implements the robust PGM variant from :cite:`florea-2017-robust`.

