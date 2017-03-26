ConvTwoBlockCnstrnt
===================

:class:`.ConvTwoBlockCnstrnt` specialises :class:`.ADMMTwoBlockCnstrnt`
for problems of the form

.. math::
   \mathrm{argmin}_{\mathbf{x},\mathbf{y}_0,\mathbf{y}_1} \;
   g_0(\mathbf{y}_0) + g_1(\mathbf{y}_1) \;\text{such that}\;
   \left( \begin{array}{c} D \\ I \end{array} \right) \mathbf{x}
   - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1 \end{array}
   \right) = \left( \begin{array}{c} \mathbf{s} \\
   \mathbf{0} \end{array} \right) \;\;.

where :math:`D \mathbf{x} = \sum_m \mathbf{d}_m * \mathbf{x}_m`,
intended to solve problems that have the form

.. math::
   \mathrm{argmin}_\mathbf{x} \;
   g_0(D \mathbf{x} - \mathbf{s}) + g_1(\mathbf{x}) \;\;,

prior to variable splitting.

See :class:`.ConvBPDNMaskDcpl` as an example of a class derived from
:class:`.ConvTwoBlockCnstrnt`.

