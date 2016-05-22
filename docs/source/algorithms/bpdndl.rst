bpdndl
======

This module includes the :class:`.BPDNDictLearn` class for solving the
problem

    .. math::
       \mathrm{argmin}_{D, X} \;
       (1/2) \| D X - S \|_F^2 + \lambda \| X \|_1 \quad \text{ s.t }
      \quad \|\mathbf{d}_m\|_2 = 1



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.BPDNDictLearn` usage

    .. literalinclude:: ../../../examples/demo_bpdndl.py
       :language: python
       :lines: 14-
