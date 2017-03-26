ADMMTwoBlockCnstrnt
===================

:class:`.ADMMTwoBlockCnstrnt` specialises :class:`.ADMM` for problems
of the form

.. math::
   \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
   f(\mathbf{x}) + g_0(\mathbf{y}_0) + g_0(\mathbf{y}_1)
   \;\text{such that}\;
   \left( \begin{array}{c} A_0 \\ A_1 \end{array} \right) \mathbf{x}
   - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1 \end{array}
   \right) = \left( \begin{array}{c} \mathbf{c}_0 \\
   \mathbf{c}_1 \end{array} \right) \;\;,

intended to solve problems that have the form

.. math::
   \mathrm{argmin}_{\mathbf{x}} \; f(\mathbf{x}) + g_0(A_0 \mathbf{x}) +
   g_1(A_1 \mathbf{x})

prior to variable splitting.


