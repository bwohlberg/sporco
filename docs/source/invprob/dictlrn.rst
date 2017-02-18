Dictionary Learning
===================

The :mod:`.dictlrn` module includes the :class:`.DictLearn` class that
supports dictionary learning via alternation between user-specified
sparse coding and dictionary update steps, each of which is based on
an ADMM algorithm. This is a very flexible framework that supports
constucting a wide variety of dictionary learning algorithms based on
the different sparse coding and dictionary update methods provided in
SPORCO; some examples are provided below. 

The standard dictionary learning classes in :mod:`.admm.bpdndl`
and the convolutional dictionary learning classes in :mod:`.admm.cbpdndl`
are both derived from :class:`.DictLearn`. These two classes provide
less flexibility -- the sparse coding methods are fixed -- but are
somewhat simpler to use.



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.DictLearn` usage (:class:`.bpdn.BPDN` sparse coding)

    .. literalinclude:: ../../../examples/stdsparse/demo_dictlrn_bpdn.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.DictLearn` usage (:class:`.cbpdn.ConvBPDN` sparse
        coding, greyscale images, as in :cite:`wohlberg-2016-efficient`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_dictlrn_cbpdn_gry.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.DictLearn` usage (:class:`.cbpdn.ConvBPDN` sparse
	coding, colour images, colour dictionary, as in
	:cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_dictlrn_cbpdn_clr.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.DictLearn` usage (:class:`.cbpdn.ConvBPDNJoint`
	sparse coding, colour images, greyscale dictionary, as in
	:cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_dictlrn_cbpdnjnt_clr.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.DictLearn` usage (:class:`.cbpdn.ConvBPDNMaskDcpl`
	sparse coding, greyscale images, as in :cite:`wohlberg-2016-boundary`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_dictlrn_cbpdn_md.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.DictLearn` usage (:class:`.cbpdn.AddMaskSim` sparse
	coding, greyscale images, as in :cite:`wohlberg-2016-boundary`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_dictlrn_cbpdn_ams.py
       :language: python
       :lines: 14-
