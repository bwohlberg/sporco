MomentumBase
============

The fundamental class from which all algorithms for computing momentum
coefficients are derived is :class:`.MomentumBase`.

The momentum coefficient can be adjusted to yield a smoother trajectory and mitigate the  zigzagging of gradient descent methods. The update for the auxiliary state sequence

.. math::
   \mathbf{y}^{(j+1)} = \mathbf{x}^{(j+1)} + \frac{t^{(j)} - 1}{t^{(j+1)}}
   \left( \mathbf{x}^{(j+1)} - \mathbf{x}^{(j)} \right) \;,

depends on the sequence of the momentum coefficient :math:`t^{(j+1)}`. The classes described in :ref:`sec-momentum-classes` provide different alternatives to updating this coefficient. The momentum coefficient used by default corresponds to the Nesterov method implemented in :class:`.MomentumNesterov`.

Classes derived from :class:`.MomentumBase` should override/define the
method :meth:`.MomentumBase.update`.

.. _sec-momentum-classes:


Momentum Classes
----------------

The momentum functionality is defined by the following classes:


* :class:`.MomentumNesterov`

  This implements the standard PGM variant from :cite:`beck-2009-fast`. The momentum coefficient is updated as

  .. math::
     t^{(j+1)} = \frac{1}{2} \left( 1 + \sqrt{1 + 4 \; (t^{(j)})^2}
     \right) \;,

  starting with :math:`t^{(1)} = 1`.


* :class:`.MomentumLinear`

  This implements the linear momentum coefficient variant from :cite:`chambolle-2015-convergence`. The momentum coefficient is updated as

  .. math::
     t^{(j+1)} = \frac{j + b}{b} \;,

  with :math:`b` a constant positive value usually selected as :math:`b \geq 2`.


* :class:`.MomentumGenLinear`

  This implements the generalized linear momentum coefficient variant from :cite:`rodriguez-2019-convergence`. The momentum coefficient is updated as

  .. math::
     t^{(j+1)} = \frac{j + a}{b} \;,

  with :math:`a` and :math:`b` constant positive values usually selected as :math:`a \in [50, 80]` and :math:`b \geq 2`.

