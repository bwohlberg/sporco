ADMM Algorithms
===============

ADMM algorithm development is supported by a set of base class from
which classes for specific optimisation problems may be derived. These
base class provides the machinery for managing iterations and stopping
conditions, computing residuals, etc. so that only methods providing
the details of a specific problem need to be defined in a derived
class:

* :doc:`ADMM <admm>` A completely generic ADMM algorithm, for
  problems of the form

  .. math::
    \mathrm{argmin}_{\mathbf{x},\mathbf{y}} \;\;
    f(\mathbf{x}) + g(\mathbf{y}) \;\mathrm{such\;that}\;
    A\mathbf{x} + B\mathbf{y} = \mathbf{c} \;\;.


* :doc:`ADMMEqual <admmequal>` A partial specialisation of
  :class:`.ADMM`, for problems of the form

  .. math::
    \mathrm{argmin}_{\mathbf{x},\mathbf{y}} \;
    f(\mathbf{x}) + g(\mathbf{y}) \;\mathrm{such\;that}\;
    \mathbf{x} = \mathbf{y} \;\;.


* :doc:`ADMMConsensus <admmcnsns>` A partial specialisation of
  :class:`.ADMM`, for solving optimisation problems of the form

  .. math::
     \mathrm{argmin}_{\mathbf{x}} \; \sum_i f_i(\mathbf{x}) + g(\mathbf{x})

  via an ADMM problem of the form

  .. math::
    \mathrm{argmin}_{\mathbf{x}_i,\mathbf{y}} \;
    \sum_i f(\mathbf{x}_i) + g(\mathbf{y}) \;\mathrm{such\;that}\;
    \left( \begin{array}{c} \mathbf{x}_0 \\ \mathbf{x}_1 \\
    \vdots \end{array} \right) = \left( \begin{array}{c}
    I \\ I \\ \vdots \end{array} \right) \mathbf{y} \;\;.


* :doc:`ADMMTwoBlockCnstrnt <admmtwoblk>` A partial
  specialisation of :class:`.ADMM`, for ADMM problems of the form

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


* :doc:`ConvTwoBlockCnstrnt <convtwoblk>` A partial
  specialisation of :class:`.ADMMTwoBlockCnstrnt` for problems of the
  form

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
    g_0(D \mathbf{x} - \mathbf{s}) + g_1(\mathbf{x})

  prior to variable splitting.



.. toctree::
   :hidden:

   admm
   admmequal
   admmcnsns
   admmtwoblk
   convtwoblk
