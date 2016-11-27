Dictionary Learning
===================

The :mod:`.dictlrn` module includes the :class:`.DictLearn` class that supports
dictionary learning via alternation between user-specified sparse
coding and dictionary update steps, each of which is based on an ADMM
algorithm. The standard dictionary learning classes in :mod:`.admm.bpdndl`
and the convolutional dictionary learning classes in :mod:`.admm.cbpdndl`
are both derived from this class.



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.DictLearn` usage (BPDN)

    .. literalinclude:: ../../../examples/stdsparse/demo_dictlrn_bpdn.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.DictLearn` usage (ConvBPDN, greyscale images)

    .. literalinclude:: ../../../examples/cnvsparse/demo_dictlrn_cbpdn_gry.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.DictLearn` usage (ConvBPDNJoint, colour images,
	greyscale dictionary)

    .. literalinclude:: ../../../examples/cnvsparse/demo_dictlrn_cbpdnjnt_clr.py
       :language: python
       :lines: 14-


.. container:: toggle

    .. container:: header

        :class:`.DictLearn` usage (ConvBPDN, colour images, colour dictionary)

    .. literalinclude:: ../../../examples/cnvsparse/demo_dictlrn_cbpdn_clr.py
       :language: python
       :lines: 14-
