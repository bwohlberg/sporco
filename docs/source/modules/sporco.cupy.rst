.. py:module:: sporco.cupy

.. _cupy_package:

sporco.cupy package
===================

This subpackage provides GPU acceleration for selected SPORCO modules via copies of these modules that are patched to replace `NumPy <http://www.numpy.org/>`__ arrays and operations with the equivalent ones provided by `CuPy <https://cupy.chainer.org/>`__ :cite:`okuta-2017-cupy`. The boolean value of attribute ``sporco.cupy.have_cupy`` indicates whether `CuPy <https://cupy.chainer.org/>`__ is installed and a GPU device is available. The modules within the ``sporco.cupy`` subpackage can still be used when ``sporco.cupy.have_cupy`` is ``False``, but they will not be GPU accelerated.

Note that the ``sporco.cupy`` subpackage is not supported under versions of Python, such as Python 2.7.x, that do not have the :mod:`importlib.util` module.


.. seealso::

   NumPy/CuPy compatibility
      `Table
      <https://docs-cupy.chainer.org/en/latest/reference/comparison.html>`__
      of NumPy functions that are implemented in CuPy


Installation and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use `CuPy <https://cupy.chainer.org/>`_, first `install CUDA <http://docs.nvidia.com/cuda/index.html#installation-guides>`_ and then `install CuPy <https://docs-cupy.chainer.org/en/stable/install.html#install-cupy/>`_. Note that it may be necessary to set the environment variables as in

::

   export CUDAHOME=/usr/local/cuda-10.2
   export PATH=${CUDAHOME}/bin:${PATH}

(substitute the appropriate path to the CUDA installation) to avoid a
``cupy.cuda.compiler.CompileException`` when using `CuPy
<https://cupy.chainer.org/>`_. If this does not rectify the problem,
the following may also be necessary:

::

   export LD_LIBRARY_PATH=${CUDAHOME}/lib64:$LD_LIBRARY_PATH



Supported Modules
~~~~~~~~~~~~~~~~~

The ``sporco.cupy`` subpackage currently provides `CuPy <https://cupy.chainer.org/>`__ acceleration of the following standard ``sporco`` modules:

=================================  ===============================
``sporco.cupy`` module             ``sporco`` module
=================================  ===============================
``sporco.cupy.cnvrep``             :mod:`sporco.cnvrep`
``sporco.cupy.common``             :mod:`sporco.common`
``sporco.cupy.linalg``             :mod:`sporco.linalg`
``sporco.cupy.metric``             :mod:`sporco.metric`
``sporco.cupy.prox``               :mod:`sporco.prox`
``sporco.cupy.util``               :mod:`sporco.util`
``sporco.cupy.admm.admm``          :mod:`sporco.admm.admm`
``sporco.cupy.admm.bpdn``          :mod:`sporco.admm.bpdn`
``sporco.cupy.admm.cbpdn``         :mod:`sporco.admm.cbpdn`
``sporco.cupy.admm.cbpdntv``       :mod:`sporco.admm.cbpdntv`
``sporco.cupy.admm.pdcsc``         :mod:`sporco.admm.pdcsc`
``sporco.cupy.admm.rpca``          :mod:`sporco.admm.rpca`
``sporco.cupy.admm.tvl1``          :mod:`sporco.admm.tvl1`
``sporco.cupy.admm.tvl2``          :mod:`sporco.admm.tvl2`
``sporco.cupy.pgm.cbpdn``          :mod:`sporco.pgm.cbpdn`
``sporco.cupy.dictlrn.onlinecdl``  :mod:`sporco.dictlrn.onlinecdl`
=================================  ===============================


Usage
~~~~~

To use the `CuPy <https://cupy.chainer.org/>`__ accelerated version of a SPORCO module:

#. import the module from ``sporco.cupy`` instead of ``sporco``
#. before calling functions/methods within the ``sporco.cupy`` module, convert `NumPy <http://www.numpy.org/>`__ arrays to `CuPy <https://cupy.chainer.org/>`__ arrays using :func:`np2cp`.
#. after calling functions/methods within the ``sporco.cupy`` module, convert `CuPy <https://cupy.chainer.org/>`_ arrays to `NumPy <http://www.numpy.org/>`__ arrays using :func:`cp2np`.

Usage examples are available for :ref:`sporco.cupy.admm.tvl1 <examples_tv_tvl1den_clr_cupy>`, :ref:`sporco.cupy.dictlrn.onlinecdl <examples_cdl_onlinecdl_clr_cupy>` and :ref:`sporco.cupy.admm.cbpdn <examples_csc_gwnden_clr>`.


Utility Functions
~~~~~~~~~~~~~~~~~

Since it is necessary to explicitly convert between `NumPy <http://www.numpy.org/>`__ arrays and `CuPy <https://cupy.chainer.org/>`__ arrays, a number of utility functions in ``sporco.cupy`` support this conversion in a way that behaves correctly independent of the value of ``sporco.cupy.have_cupy``, in that conversion is performed when the value is ``True``, and no conversion is perfomed when it is ``False``.


.. py:function:: array_module(*args)

   Get the array module (``numpy`` or ``cupy``) of the array argument. This
   function is an alias for :func:`cupy.get_array_module`.


.. py:function:: np2cp(u)

   Convert a ``numpy`` ndarray to a ``cupy`` array. This function is an alias
   for :func:`cupy.asarray`


.. py:function:: cp2np(u)

   Convert a ``cupy`` array to a ``numpy`` ndarray. This function is an alias
   for :func:`cupy.asnumpy`


.. py:function:: cupy_wrapper(func)

   A wrapper function that converts ``numpy`` ndarray arguments to ``cupy``
   arrays, and convert any ``cupy`` arrays returned by the wrapped function
   into ``numpy`` ndarrays.




|

Some additional utility functions provide useful functionality when package `GPUtil <https://github.com/anderskm/gputil>`__ is installed, and return fixed default return values when it is not installed:


.. py:function:: gpu_info()

   Return a list of namedtuples representing attributes of each GPU
   device. Returns an empty list if
   `GPUtil <https://github.com/anderskm/gputil>`_ is not installed.


.. py:function:: gpu_load(wproc=0.5, wmem=0.5)

   Return a list of namedtuples representing the current load for
   each GPU device. The processor and memory loads are fractions
   between 0 and 1. The weighted load represents a weighted average
   of processor and memory loads using the parameters `wproc` and
   `wmem` respectively. Returns an empty list if
   `GPUtil <https://github.com/anderskm/gputil>`_ is not installed.


.. py:function:: device_by_load(wproc=0.5, wmem=0.5)

   Get a list of GPU device ids ordered by increasing weighted
   average of processor and memory load. Returns an empty list if
   `GPUtil <https://github.com/anderskm/gputil>`_ is not installed.


.. py:function:: select_device_by_load(wproc=0.5, wmem=0.5)

   Set the current device for cupy as the device with the lowest
   weighted average of processor and memory load. Returns 0 if
   `GPUtil <https://github.com/anderskm/gputil>`_ is not installed.


.. py:function:: available_gpu(*args, **kwargs)

   Get the device id for an available GPU when multiple GPUs are installed.
   This function is an alias for ``GPUtil.getAvailable``. Returns 0 if
   `GPUtil <https://github.com/anderskm/gputil>`_ is not installed.


|

The current GPU device can be also selected using :meth:`cupy.cuda.Device.use`, e.g. to select device id 1

::

   cp.cuda.Device(1).use()

`CuPy <https://docs-cupy.chainer.org/en/stable/index.html>`__ also provides a `context manager for GPU device selection <https://docs-cupy.chainer.org/en/stable/tutorial/basic.html#current-device>`__.
