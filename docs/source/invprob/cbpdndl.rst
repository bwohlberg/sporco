Modules cbpdndl and parcnsdl
============================

These modules includes classes :class:`.admm.cbpdndl.ConvBPDNDictLearn` and :class:`.admm.parcnsdl.ConvBPDNDictLearn_Consensus` for solving the problem

.. math::
   \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
   \frac{1}{2} \sum_k \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right \|_2^2 + \lambda \sum_k \sum_m \| \mathbf{x}_{k,m} \|_1
   \quad \text{ s.t. } \quad \mathbf{d}_m \in C \;\; \forall m \;,

where :math:`C` is the feasible set consisting of filters with unit norm
and constrained support.

In addition classes :class:`.admm.cbpdndl.ConvBPDNMaskDcplDictLearn` and :class:`.admm.parcnsdl.ConvBPDNMaskDcplDictLearn_Consensus` support solving
the problem

.. math::
   \mathrm{argmin}_{\mathbf{d}, \mathbf{x}} \;
   \frac{1}{2} \sum_k \left \|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right) \right \|_2^2 + \lambda \sum_k \sum_m \|
   \mathbf{x}_{k,m} \|_1 \quad \text{ s.t. } \quad \mathbf{d}_m \in C \;\;
   \forall m \;,

where :math:`W` is a mask array.

On a multi-core host, :class:`.admm.parcnsdl.ConvBPDNDictLearn_Consensus` and
:class:`.admm.parcnsdl.ConvBPDNMaskDcplDictLearn_Consensus` can be
substantially faster than :class:`.admm.cbpdndl.ConvBPDNDictLearn` and
:class:`.admm.cbpdndl.ConvBPDNMaskDcplDictLearn` respectively.



Usage Examples
--------------

.. container:: toggle

    .. container:: header

	:class:`.cbpdndl.ConvBPDNDictLearn` usage (IterSM dictionary update,
	greyscale images, single-scale dictionary, see
	:cite:`wohlberg-2016-efficient`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_ism_gry_ssd.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

	:class:`.cbpdndl.ConvBPDNDictLearn` usage (consensus dictionary
	update, greyscale images, single-scale dictionary, see
	:cite:`garcia-2017-subproblem` :cite:`garcia-2017-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_cns_gry_ssd.py
       :language: python
       :lines: 9-



.. container:: toggle

    .. container:: header

       :class:`.parcnsdl.ConvBPDNDictLearn_Consensus` usage (parallel
       consensus dictionary update, greyscale images, see
       :cite:`garcia-2017-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_parcns_gry.py
       :language: python
       :lines: 9-



.. container:: toggle

    .. container:: header

	:class:`.cbpdndl.ConvBPDNDictLearn` usage (greyscale images,
	multi-scale dictionary, see :cite:`wohlberg-2016-efficient`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_gry_msd.py
       :language: python
       :lines: 9-



.. container:: toggle

    .. container:: header

	:class:`.cbpdndl.ConvBPDNDictLearn` usage (colour images, single-scale
	colour dictionary, see :cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_clr_ssd.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

	:class:`.cbpdndl.ConvBPDNDictLearn` usage (colour images, multi-scale
	colour dictionary, see :cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_clr_msd.py
       :language: python
       :lines: 9-



.. container:: toggle

    .. container:: header

	:class:`.parcnsdl.ConvBPDNDictLearn_Consensus` usage (parallel
	consensus dict. update, colour images, colour
	dictionary, see  :cite:`garcia-2017-convolutional`
	:cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_parcns_clr.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

	:class:`.cbpdndl.ConvBPDNMaskDcplDictLearn` usage (greyscale images,
	single-scale dictionary, see :cite:`wohlberg-2016-boundary`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_md_gry.py
       :language: python
       :lines: 11-



.. container:: toggle

    .. container:: header

	:class:`.cbpdndl.ConvBPDNMaskDcplDictLearn` usage (colour images,
	single-scale colour dictionary, see :cite:`wohlberg-2016-boundary`
	:cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_cbpdndl_md_clr.py
       :language: python
       :lines: 11-
