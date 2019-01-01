Dictionary Learning
===================

The :mod:`.dictlrn.dictlrn` module includes the :class:`.DictLearn`
class that supports dictionary learning via alternation between
user-specified sparse coding and dictionary update steps, each of
which is based on an ADMM algorithm. This is a very flexible framework
that supports constucting a wide variety of dictionary learning
algorithms based on the different sparse coding and dictionary update
methods provided in SPORCO; some examples are provided below.

The standard dictionary learning classes in :mod:`.dictlrn.bpdndl` and
the convolutional dictionary learning classes in
:mod:`.dictlrn.cbpdndl` and :mod:`.dictlrn.cbpdndlmd` are both derived
from :class:`.DictLearn`. These two classes provide less flexibility
-- the sparse coding methods are fixed -- but are somewhat simpler to
use.

A :ref:`usage example <examples_cdl_cbpdndl_jnt_clr>` is available.
