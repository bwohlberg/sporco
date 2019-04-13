Modules cbpdn and cbpdntv
=========================

Module :mod:`.admm.cbpdn` includes the following classes:

* :class:`.admm.cbpdn.ConvBPDN`

  Solve the basic Convolutional BPDN problem (see
  :cite:`wohlberg-2016-efficient`)

  .. math::
     \mathrm{argmin}_\mathbf{x} \;
     \frac{1}{2} \left \|  \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
     \right \|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1

  The function :func:`sporco.cuda.cbpdn` provides a GPU accelerated
  solver for this problem if the
  `sporco-cuda  <https://github.com/bwohlberg/sporco-cuda>`__ extension
  package is installed.


* :class:`.ConvBPDNJoint`

  Solve the Convolutional BPDN problem with joint sparsity over
  multiple signal channels via an :math:`\ell_{2,1}` norm term
  (see :cite:`wohlberg-2016-convolutional`)

  .. math::
       \mathrm{argmin}_\mathbf{x} \;
       \frac{1}{2} \sum_c \left\| \sum_m \mathbf{d}_m * \mathbf{x}_{c,m} -
       \mathbf{s}_c \right\|_2^2 + \lambda \sum_c \sum_m
       \| \mathbf{x}_{c,m} \|_1 + \mu \| \{ \mathbf{x}_{c,m} \} \|_{2,1}


* :class:`.ConvElasticNet`

  Solve the Convolutional Elastic Net (i.e. Convolutional BPDN with an
  additional :math:`\ell_2` penalty on the coefficient maps)

  .. math::
     \mathrm{argmin}_\mathbf{x} \;
     \frac{1}{2} \left \| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
     \right \|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
     \frac{\mu}{2} \sum_m \| \mathbf{x}_m \|_2^2


* :class:`.ConvBPDNGradReg`

  Solve Convolutional BPDN with an additional :math:`\ell_2` penalty
  on the gradient of the coefficient maps (see
  :cite:`wohlberg-2016-convolutional2`)

  .. math::
     \mathrm{argmin}_\mathbf{x} \;
     \frac{1}{2} \left \| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
     \right \|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
     \frac{\mu}{2} \sum_i \sum_m \| G_i \mathbf{x}_m \|_2^2

  where :math:`G_i` is an operator computing the derivative along index
  :math:`i`.

  The function :func:`sporco.cuda.cbpdngrd` provides a GPU accelerated
  solver for this problem if the
  `sporco-cuda  <https://github.com/bwohlberg/sporco-cuda>`__ extension
  package is installed.


* :class:`.ConvL1L1Grd`

  Solve a Convolutional Sparse Coding problem with an :math:`\ell_1`
  data fidelity term and both :math:`\ell_1` and :math:`\ell_2` of
  gradient regularisation terms :cite:`wohlberg-2016-convolutional2`


    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       \left\|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s}\right) \right\|_1 + \lambda \sum_m
       \| \mathbf{x}_m \|_1 + (\mu/2) \sum_i \sum_m
       \| G_i \mathbf{x}_m \|_2^2

  where :math:`W` is a mask array and :math:`G_i` is an operator
  computing the derivative along index :math:`i`.


* :class:`.ConvBPDNProjL1`

  Solve the convolutional sparse representation problem with an
  :math:`\ell_2` objective and an :math:`\ell_1` constraint

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       \frac{1}{2} \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 \; \text{such that} \; \| \mathbf{x}_m \|_1
       \leq \gamma


* :class:`.ConvMinL1InL2Ball`

  Solve the convolutional sparse representation problem with an
  :math:`\ell_1` objective and an :math:`\ell_2` constraint

    .. math::
       \mathrm{argmin}_\mathbf{x} \sum_m \| \mathbf{x}_m \|_1 \;
       \text{such that} \;  \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m
       - \mathbf{s} \right\|_2 \leq \epsilon


* :class:`.ConvBPDNMaskDcpl`

  Solve Convolutional BPDN with Mask Decoupling (see :cite:`heide-2015-fast`)

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       \frac{1}{2} \left\|  W \left(\sum_m \mathbf{d}_m * \mathbf{x}_m -
       \mathbf{s}\right) \right\|_2^2 + \lambda \sum_m
       \| \mathbf{x}_m \|_1

  where :math:`W` is a mask array.


* :class:`.AddMaskSim`

  A wrapper class for applying the Additive Mask Simulation (AMS) boundary
  handling technique (see :cite:`wohlberg-2016-boundary`) to any of the
  other :mod:`.admm.cbpdn` classes.

  If the `sporco-cuda <https://github.com/bwohlberg/sporco-cuda>`__
  extension package is installed, functions :func:`sporco.cuda.cbpdnmsk`
  and :func:`sporco.cuda.cbpdngrdmsk` provide GPU accelerated solvers for
  AMS variants of the ``ConvBPDN`` and ``ConvBPDNGradReg`` problems.


* :class:`.MultiDictConvBPDN`

  A wrapper class for solving a convolutional sparse coding problem
  fitting a single set of coefficient maps to multiple dictionaries
  and signals, e.g. when applied to :class:`.admm.cbpdn.ConvBPDN`,

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| D_0 \mathbf{x} - \mathbf{s}_0 \right\|_2^2 +
       (1/2) \left\| D_1 \mathbf{x} - \mathbf{s}_1 \right\|_2^2 +
       \lambda \| \mathbf{x} \|_1 \;\;,

  for input images :math:`\mathbf{s}_0`, :math:`\mathbf{s}_1`,
  dictionaries :math:`D_0` and :math:`D_0`, and coefficient map set
  :math:`\mathbf{x}`, where :math:`D_0 \mathbf{x} = \sum_m
  \mathbf{d}_{0,m} \mathbf{x}_m` and :math:`D_1 \mathbf{x} = \sum_m
  \mathbf{d}_{1,m} \mathbf{x}_m`.


|


Module :mod:`.admm.cbpdntv` includes the following classes:

* :class:`.ConvBPDNScalarTV`

  Solve Convolutional BPDN with an additional term penalising the total
  variation of each coefficient map (see :cite:`wohlberg-2017-convolutional`)

    .. math::
       \mathrm{argmin}_\mathbf{x} \; \frac{1}{2}
       \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
       \mu \sum_m \left\| \sqrt{\sum_i (G_i \mathbf{x}_m)^2} \right\|_1
       \;\;,

  where :math:`G_i` is an operator computing the derivative along index
  :math:`i`.


* :class:`.ConvBPDNVectorTV`

  Solve Convolutional BPDN with an additional term penalising the vector
  total variation of the coefficient maps (see
  :cite:`wohlberg-2017-convolutional`)

    .. math::
       \mathrm{argmin}_\mathbf{x} \; \frac{1}{2}
       \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
       \mu \left\| \sqrt{\sum_m \sum_i (G_i \mathbf{x}_m)^2} \right\|_1
       \;\;,

  where :math:`G_i` is an operator computing the derivative along index
  :math:`i`.


* :class:`.ConvBPDNRecTV`

  Solve Convolutional BPDN with an additional term penalising the total
  variation of the reconstruction from the sparse representation (see
  :cite:`wohlberg-2017-convolutional`)

    .. math::
       \mathrm{argmin}_\mathbf{x} \; \frac{1}{2}
       \left\| \sum_m \mathbf{d}_m * \mathbf{x}_m - \mathbf{s}
       \right\|_2^2 + \lambda \sum_m \| \mathbf{x}_m \|_1 +
       \mu \left\| \sqrt{\sum_i \left( G_i \left( \sum_m \mathbf{d}_m *
       \mathbf{x}_m  \right) \right)^2} \right\|_1 \;\;,

  where :math:`G_i` is an operator computing the derivative along index
  :math:`i`.

|

:ref:`Usage examples <examples_csc_index>` are available.




Multi-channel Data
------------------

Some of the example scripts demonstrate usage of the classes in the :mod:`.admm.cbpdn` module with multi-channel (all of these examples are for RGB colour images, but an arbitrary number of channels is supported) input images. Multi-channel input examples are not provided for all classes since the usage differences for single- and multi-channel inputs are the same across most of the classes. There are two fundamentally different ways of representing multi-channel input images: a single-channel dictionary together with a separate set of coefficient maps for each channel, or a multi-channel dictionary with a single set of coefficient maps shared across all channels. In the former case the coefficient maps can be independent across the different channels, or expected correlations between the channels can be modelled via a joint sparsity penalty. A more detailed discussion of these issues can be found in :cite:`wohlberg-2016-convolutional`.
