.. _cuda_package:

sporco.cuda package
===================

This package allows the `SPORCO-CUDA extension package <https://github.com/bwohlberg/sporco-cuda>`_ to be accessed within the ``sporco`` namespace, i.e.

::

  import sporco.cuda

instead of

::

  import sporco_cuda

The import of ``sporco.cuda`` will succeed even if the ``sporco-cuda`` extension package is not installed, but the availability of these extensions can be determined by checking the boolean value of ``sporco.cuda.have_cuda``. In addition, the function ``sporco.cuda.device_count()`` is available independent of whether the import succeeds, allowing it to be used as a reliable test for whether it is possible to run the optimisation functions from ``sporco-cuda``, since this requires both that ``sporco-cuda`` be installed and that the value returned by ``sporco.cuda.device_count()`` is greater than zero. For example

::

  from sporco import cuda
  from sporco.admm import cbpdn

  # ...
  # Load dictionary D and test image s (and highpass filter it) here
  # ...

  lmbda = 1e-2
  opt = cbpdn.ConvBPDN.Options({'MaxMainIter': 250'})
  if cuda.device_count() > 0:
      X = cuda.cbpdn(D, sh, lmbda, opt)
  else:
      c = cbpdn.ConvBPDN(D, sh, lmbda, opt)
      X = c.solve()


The content of the ``sporco.cuda`` namespace is summarised below. For full details of the functions listed here, see the `SPORCO-CUDA documentation <http://sporco-cuda.rtfd.io>`_.


Always available
~~~~~~~~~~~~~~~~

.. py:attribute:: have_cuda

   A boolean value indicating whether the import of ``sporco_cuda`` succeeded.


.. py:function:: device_count()

   Get the number of CUDA GPU devices installed on the host system. Returns 0
   if no devices are installed or if the import of ``sporco_cuda`` failed.

   Returns
   -------
   ndev : int
      Number of installed deviced


Only available if ``have_cuda`` is `True`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: current_device(id=None)

   Get or set the current CUDA GPU device. The current device is not set
   if `id` is None

   Parameters
   ----------
   id : int or None, optional (default None)
     Device number of device to be set as current device

   Returns
   -------
   id : int
     Device number of current device


.. py:function:: memory_info()

   Get memory information for the current CUDA GPU device.

   Returns
   -------
   free : int
     Free memory in bytes
   total : int
      Total memory in bytes


.. _cuda_cbpdn:
.. py:function:: cbpdn(D, S, lmbda, opt, dev=0)

   A GPU-accelerated version of :class:`.admm.cbpdn.ConvBPDN`. Multiple images
   and multi-channel images in input signal `S` are currently not supported.

   Parameters
   ----------
   D : array_like(float32, ndim=3)
     Dictionary array (three dimensional)
   S : array_like(ndim=2)
     Signal array (two dimensional)
   lmbda : float32
     Regularisation parameter
   opt : dict or :class:`.admm.cbpdn.ConvBPDN.Options` object
     Algorithm options
   dev : int
     Device number of GPU device to use

   Returns
   -------
   X : ndarray
     Coefficient map array (sparse representation)


.. _cuda_cbpdngrd:
.. py:function:: cbpdngrd(D, S, lmbda, mu, opt, dev=0)

   A GPU-accelerated version of :class:`.admm.cbpdn.ConvBPDNGradReg`. Multiple
   images and multi-channel images in input signal `S` are currently not
   supported.

   Parameters
   ----------
   D : array_like(float32, ndim=3)
     Dictionary array (three dimensional)
   S : array_like(ndim=2)
     Signal array (two dimensional)
   lmbda : float32
     Regularisation parameter (l1)
   mu : float
     Regularisation parameter (gradient)
   opt : dict or :class:`.admm.cbpdn.ConvBPDNGradReg.Options` object
     Algorithm options
   dev : int
     Device number of GPU device to use

   Returns
   -------
   X : ndarray
     Coefficient map array (sparse representation)
