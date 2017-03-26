Module ccmod
============

This module includes the :class:`.ConvCnstrMOD` class for solving the
problem

.. math::
   \mathrm{argmin}_\mathbf{d} \;
   \frac{1}{2} \sum_k \left \| \sum_m \mathbf{d}_m * \mathbf{x}_{k,m} -
   \mathbf{s}_k \right \|_2^2 \quad \text{ such that } \quad \mathbf{d}_m
   \in C

where :math:`C` is the feasible set :math:`C` consisting of filters
with unit norm and constrained support.



Usage Examples
--------------

.. container:: toggle

    .. container:: header

        :class:`.ConvCnstrMOD` usage (greyscale images, as in
        :cite:`wohlberg-2016-efficient`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_ccmod_gry.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.ConvCnstrMOD` usage (colour images, greyscale
	dictionary, as in :cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_ccmod_clr_gd.py
       :language: python
       :lines: 9-


.. container:: toggle

    .. container:: header

        :class:`.ConvCnstrMOD` usage (colour images, colour
        dictionary, as in :cite:`wohlberg-2016-convolutional`)

    .. literalinclude:: ../../../examples/cnvsparse/demo_ccmod_clr_cd.py
       :language: python
       :lines: 9-
