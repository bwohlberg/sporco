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
   \mathbf{0} \end{array} \right) \;\;,

where :math:`D \mathbf{x} = \sum_m \mathbf{d}_m * \mathbf{x}_m`,
intended to solve problems that have the form

.. math::
   \mathrm{argmin}_\mathbf{x} \;
   g_0(D \mathbf{x} - \mathbf{s}) + g_1(\mathbf{x})

prior to variable splitting.

See :class:`.ConvBPDNMaskDcpl` as an example of a class derived from
:class:`.ConvTwoBlockCnstrnt`.


The methods of :class:`.ConvTwoBlockCnstrnt` that are relevant to an
implementer of a derived class are described in the following
sections.


Initialisation
--------------

The ``__init__`` method of the derived class should call the
:class:`.ConvTwoBlockCnstrnt` ``__init__`` method to ensure proper
initialisation.



Block Access
------------

Block access methods are as described in
:ref:`sec-admmtwoblk-block-access`, except that
:meth:`.ConvTwoBlockCnstrnt.block_sep0` and
:meth:`.ConvTwoBlockCnstrnt.block_cat` override
:meth:`.ADMMTwoBlockCnstrnt.block_sep` and
:meth:`.ADMMTwoBlockCnstrnt.block_cat` respectively, for the following
reason. In :class:`.ConvTwoBlockCnstrnt`, block 0 has the same shape
as input :math:`\mathbf{s}`, i.e. :math:`N \times C \times K \times 1`
(assuming two spatial dimensions), while block 1 has the dimensions of
:math:`\mathbf{x}`, i.e. :math:`N \times 1 \times K \times M` (again,
assuming two spatial dimensions). In order to allow the two blocks to
be concatenated into a single array, the channel and filter axes are
swapped prior to concatenation and after separation.


Update Steps
------------

The update steps are as described in
:ref:`sec-admm-update-steps`. Typically only the :math:`\mathbf{y}`
update :meth:`.ADMM.ystep` will need to be overridden since a full
implementation of the :math:`\mathbf{x}` update :meth:`.ADMM.xstep` is
provided.


Constraint Definition
---------------------

The constraint definition methods :meth:`.ADMMTwoBlockCnstrnt.cnst_A`,
:meth:`.ADMMTwoBlockCnstrnt.cnst_AT`,
:meth:`.ADMMTwoBlockCnstrnt.cnst_B`, and
:meth:`.ADMMTwoBlockCnstrnt.cnst_c` are inherited from
:class:`.ADMMTwoBlockCnstrnt`, and
:meth:`.ConvTwoBlockCnstrnt.cnst_A0`,
:meth:`.ConvTwoBlockCnstrnt.cnst_A0T`, and
:meth:`.ConvTwoBlockCnstrnt.cnst_c0` defined to implement the block
form constraint

.. math::
   \left( \begin{array}{c} D \\ I \end{array} \right) \mathbf{x}
   - \left( \begin{array}{c} \mathbf{y}_0 \\ \mathbf{y}_1 \end{array}
   \right) = \left( \begin{array}{c} \mathbf{s} \\
   \mathbf{0} \end{array} \right)

so that a derived class need typically not override any of these
methods.


Residual Evaluation
-------------------

The residual evaluation methods :meth:`.ADMMTwoBlockCnstrnt.rsdl_r`,
:meth:`.ConvTwoBlockCnstrnt.rsdl_s`,
:meth:`.ADMMTwoBlockCnstrnt.rsdl_rn`,
:meth:`.ConvTwoBlockCnstrnt.rsdl_sn` are all appropriately defined in
terms of the constraint definition methods described above, and will
typically not need to be overridden.



Iteration Statistics
--------------------

The iteration statistics mechanism, as described in
:ref:`sec-admm-iteration-stats`, is inherited largely unchanged from
:class:`.ADMMTwoBlockCnstrnt`.
