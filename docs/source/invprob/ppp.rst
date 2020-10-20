.. _invprob_ppp:

Modules admm.ppp and pgm.ppp
==============================

These modules support the solution of inverse problems via the Plug and Play Priors (PPP) framework :cite:`venkatakrishnan-2013-plugandplay2`. Module :mod:`.admm.ppp` supports the ADMM variant of PPP :cite:`venkatakrishnan-2013-plugandplay2` :cite:`sreehari-2016-plug`, and :mod:`.pgm.ppp` support the PGM variant :cite:`kamilov-2017-plugandplay`.

Classes :class:`.admm.ppp.GenericPPP` and :class:`.pgm.ppp.GenericPPP` are  solver bases classes from which solver classes for specific problems can be derived, and classes :class:`.admm.ppp.PPP` and :class:`.pgm.ppp.PPP` provide solvers that can be used without the need to derived new classes, the specific problem form being specified instead by passing relevant problem defining functions to the initialiser.

:ref:`Usage examples <examples_ppp_index>` are available.
