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

See :class:`.bpdn.MinL1InL2Ball` as an example of a class derived from
:class:`.ADMMTwoBlockCnstrnt`.


The methods of :class:`.ADMMTwoBlockCnstrnt` that are relevant to an
implementer of a derived class are described in the following
sections.


Initialisation
--------------

The ``__init__`` method of the derived class should call the
:class:`.ADMMTwoBlockCnstrnt` ``__init__`` method to ensure proper
initialisation. The interface is similar to that of :class:`.ADMM`
``__init__`` method except for the introduction of parameters
``blkaxis`` and ``blkidx`` that specify how the blocks of
:math:`\mathbf{y}` are concatenated into a single array.


.. _sec-admmtwoblk-block-access:

Block Access
------------

The following methods provide access to the block components of the
arrays representing :math:`\mathbf{y}` or :math:`\mathbf{u}`:

* :meth:`.ADMMTwoBlockCnstrnt.block_sep0` Return a slice of the
  argument array that corresponds to block 0.

* :meth:`.ADMMTwoBlockCnstrnt.block_sep1` Return a slice of the
  argument array that corresponds to block 1.

* :meth:`.ADMMTwoBlockCnstrnt.block_sep` Return a tuple containing
  block 0 and block 1 of the argument array.

* :meth:`.ADMMTwoBlockCnstrnt.block_cat` Return a single array
  constructed by appropriately concatenating the two argument arrays
  representing blocks 0 and 1.

* :meth:`.ADMMTwoBlockCnstrnt.var_y0` Return a slice of ``self.Y``
  corresponding to block :math:`\mathbf{y}_0`.

* :meth:`.ADMMTwoBlockCnstrnt.var_y1` Return a slice of ``self.Y``
  corresponding to block :math:`\mathbf{y}_1`.


Update Steps
------------

The update steps are as described in :ref:`sec-admm-update-steps`. The
:math:`\mathbf{y}` update :meth:`.ADMM.ystep` will typically make use
of the block access methods described above to apply separate updates
to the blocks :math:`\mathbf{y}_0` and :math:`\mathbf{y}_1`.


Constraint Definition
---------------------

The constraint definition methods :meth:`.ADMMTwoBlockCnstrnt.cnst_A`,
:meth:`.ADMMTwoBlockCnstrnt.cnst_AT`,
:meth:`.ADMMTwoBlockCnstrnt.cnst_B`, and
:meth:`.ADMMTwoBlockCnstrnt.cnst_c` are defined to implement the block
form constraint

.. math::
   \left( \begin{array}{c} A_0 \\ A_1 \end{array} \right) \mathbf{x}
   - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1 \end{array}
   \right) = \left( \begin{array}{c} \mathbf{c}_0 \\
   \mathbf{c}_1 \end{array} \right) \;\;.


A derived class should override the following methods:

* :meth:`.ADMMTwoBlockCnstrnt.cnst_A0`

  Compute and return :math:`A_0 \mathbf{x}`

* :meth:`.ADMMTwoBlockCnstrnt.cnst_A0T`

  Compute and return :math:`A_0^T \mathbf{u}`

* :meth:`.ADMMTwoBlockCnstrnt.cnst_A1`

  Compute and return :math:`A_1 \mathbf{x}`

* :meth:`.ADMMTwoBlockCnstrnt.cnst_A1T`

  Compute and return :math:`A_1^T \mathbf{u}`

* :meth:`.ADMMTwoBlockCnstrnt.cnst_c0`

  Return constant :math:`\mathbf{c}_0`

* :meth:`.ADMMTwoBlockCnstrnt.cnst_c1`

  Return constant :math:`\mathbf{c}_1`


The default definitions are :math:`A_0 = I`, :math:`A_1 = I`,
:math:`\mathbf{c}_0 = \mathbf{0}`, and :math:`\mathbf{c}_1 =
\mathbf{0}`; the corresponding methods need not be overridden where
these are the desired choices.


Residual Evaluation
-------------------

The residual evaluation methods :meth:`.ADMMTwoBlockCnstrnt.rsdl_r`,
:meth:`.ADMMTwoBlockCnstrnt.rsdl_s`,
:meth:`.ADMMTwoBlockCnstrnt.rsdl_rn`,
:meth:`.ADMMTwoBlockCnstrnt.rsdl_sn` are all appropriately defined in
terms of the constraint definition methods described above, and will
typically not need to be overridden.



Iteration Statistics
--------------------

The iteration statistics mechanism, as described in
:ref:`sec-admm-iteration-stats`, is inherited largely unchanged from
:class:`.ADMM`. The main exception is that
:meth:`.ADMMTwoBlockCnstrnt.obfn_g` is defined as the sum of the
values returned by :meth:`.ADMMTwoBlockCnstrnt.obfn_g0` and
:meth:`.ADMMTwoBlockCnstrnt.obfn_g1`. A derived class may either
just override these two methods, or override
:meth:`.ADMMTwoBlockCnstrnt.obfn_g` (or :meth:`.ADMM.eval_objfn`)
itself.
