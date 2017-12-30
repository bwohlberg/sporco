Module rpca
===========

This module includes the :class:`.RobustPCA` class for solving the
problem

.. math::
   \mathrm{argmin}_{X, Y} \;
   \| X \|_* + \lambda \| Y \|_1 \quad \text{ such that }
   \quad X + Y = S

A :ref:`usage example <example_misc_rpca>` is available.
