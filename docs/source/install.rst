Installation
============

SPORCO is supported under Python 3.x. It is currently also expected to function correctly under Python 2.7, but this is not expected to continue indefinitely, and use under Python 2.7 is no longer supported.

The simplest way to install the most recent release of SPORCO from
`PyPI <https://pypi.python.org/pypi/sporco/>`_ is

::

    pip install sporco


SPORCO can also be installed from source, either from the development
version from `GitHub <https://github.com/bwohlberg/sporco>`_, or from
a release source package downloaded from `PyPI
<https://pypi.python.org/pypi/sporco/>`_.

To install the development version from `GitHub
<https://github.com/bwohlberg/sporco>`_, either do

::

    pip install git+https://github.com/bwohlberg/sporco

or

::

    git clone https://github.com/bwohlberg/sporco.git

followed by

::

   cd sporco
   python setup.py build
   python setup.py test
   python setup.py install

Please report any test failures. The install command will usually have to be
performed with root permissions, e.g. on Ubuntu Linux

::

   sudo -H pip install sporco

or

::

   sudo python setup.py install

The procedure for installing from a source package downloaded from `PyPI
<https://pypi.python.org/pypi/sporco/>`_ is similar.


SPORCO can also be installed as a `conda <https://conda.io/docs/>`__ package from the `conda-forge <https://conda-forge.org/>`__ channel

::

   conda install -c conda-forge sporco


A summary of the most significant changes between SPORCO releases can
be found in the ``CHANGES.rst`` file. It is strongly recommended to
consult this summary when updating from a previous version.



Requirements
------------

The primary requirements are Python itself, and modules `future
<http://python-future.org>`__, `numpy <http://www.numpy.org>`__,
`scipy <https://www.scipy.org>`__, `imageio <https://imageio.github.io/>`__,
`pyfftw <https://hgomersall.github.io/pyFFTW>`__, and
`matplotlib <http://matplotlib.org>`__. Installation of these requirements
is system dependent:

.. tabs::

   .. group-tab:: :fa:`linux` Linux

      Under Ubuntu Linux 20.04, the following commands should be sufficient for Python 3

      ::

	sudo apt-get -y install python3-numpy python3-scipy python3-numexpr python3-matplotlib \
				python3-imageio python3-pip python3-future libfftw3-dev python3-pytest
	sudo -H pip3 install pyfftw pytest-runner

      The following optional dependencies are required only for the
      :ref:`PPP <invprob_ppp>` usage examples, for which Python 3 is
      required

      ::

	sudo apt-get -y install libopenblas-base
	sudo -H pip3 install bm3d

      Some additional dependencies are required for building the
      documentation from the package source, for which Python 3.3 or
      later is required

      ::

	sudo apt-get -y install python3-sphinx python3-numpydoc python3-pygraphviz pandoc
	sudo -H pip3 install sphinxcontrib-bibtex sphinx_tabs sphinx_fontawesome jonga \
			     jupyter py2jn


   .. group-tab:: :fa:`apple` Mac OS

      The first step is to install Python 3.x

      ::

	brew install python3

      The `FFTW library <http://www.fftw.org/>`_ is also required

      ::

	brew install fftw


      The Python modules required by SPORCO can be installed using `pip`

      ::

	pip3 install numpy scipy imageio matplotlib pyfftw
	pip3 install six future python-dateutil pyparsing cycler
	pip3 install pytz pytest pytest-runner

      The following optional dependency is required only for the
      :ref:`PPP <invprob_ppp>` usage examples, for which Python 3 is required

      ::

	pip3 install bm3d


      Some additional dependencies are required for building the
      documentation from the package source, for which Python 3 is required

      ::

	brew install graphviz
	pip3 install sphinx numpydoc sphinxcontrib-bibtex sphinx_tabs
	pip3 install sphinx_fontawesome jonga



   .. group-tab:: :fa:`windows` Windows

      A version of Python that includes NumPy and SciPy
      is required. The instructions given here are for installing a
      reference version from `python.org
      <https://www.python.org/downloads/windows/>`_, but a potentially
      simpler alternative would be to install one of the Windows
      versions of Python distributed with the SciPy stack that are
      listed at `scipy.org <https://scipy.org/install.html>`_.

      The first step is to install Python itself, e.g. for version
      3.6.2, download `python-3.6.2-amd64.exe
      <https://www.python.org/ftp/python/3.6.2/python-3.6.2-amd64.exe>`_
      and run the graphical installer. The easiest way of installing
      the main required packages is to download the binaries from the
      list of `Unofficial Windows Binaries for Python Extension
      Packages <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_. At the
      time of writing this documentation, the current versions of
      these binaries for each main package are

	* `NumPy <http://www.lfd.uci.edu/~gohlke/pythonlibs/tuft5p8b/numpy-1.13.1+mkl-cp36-cp36m-win_amd64.whl>`__
	* `SciPy <http://www.lfd.uci.edu/~gohlke/pythonlibs/tuft5p8b/scipy-0.19.1-cp36-cp36m-win_amd64.whl>`__
	* `Matplotlib <http://www.lfd.uci.edu/~gohlke/pythonlibs/tuft5p8b/matplotlib-2.0.2-cp36-cp36m-win_amd64.whl>`__
	* `pyFFTW <http://www.lfd.uci.edu/~gohlke/pythonlibs/tuft5p8b/pyFFTW-0.10.4-cp36-cp36m-win_amd64.whl>`__

      After downloading and saving each of these binaries, open a
      Command Prompt, change directory to the folder in which the
      binaries were saved, and enter

      ::

	pip install numpy-1.13.1+mkl-cp36-cp36m-win_amd64.whl
	pip install scipy-0.19.1-cp36-cp36m-win_amd64.whl
	pip install matplotlib-2.0.2-cp36-cp36m-win_amd64.whl
	pip install pyFFTW-0.10.4-cp36-cp36m-win_amd64.whl
	pip install future imageio


      The following optional dependency is required only for the
      :ref:`PPP <invprob_ppp>` usage examples

      ::

	pip install bm3d


      Some additional dependencies are required for building the
      documentation from the package source

      ::

	pip install sphinx numpydoc sphinxcontrib-bibtex sphinx_tabs
	pip install sphinx_fontawesome


      It is also necessary to download and install
      `Graphviz <http://www.graphviz.org/Download_windows.php>`__ and then
      set the Windows ``PATH`` environment variable to include the ``dot``
      command, e.g. to do this on the command line, for the current version
      of Graphviz

      ::

	set PATH=%PATH%;"C:\Program Files (x86)\Graphviz2.38\bin"


In addition to the required packages, a number of optional packages enable
additional features when installed:


.. |numexpr| replace:: `numexpr <https://github.com/pydata/numexpr>`__
.. |mpldatacursor| replace:: `mpldatacursor <https://github.com/joferkington/mpldatacursor>`__
.. |cupy| replace:: `cupy <https://github.com/cupy/cupy>`__
.. |wrltzr| replace:: `wurlitzer <https://github.com/minrk/wurlitzer>`__
.. |gputil| replace:: `GPUtil <https://github.com/anderskm/gputil>`__
.. |mpi4py| replace:: `mpi4py <https://github.com/mpi4py/mpi4py>`__
.. |bm3d| replace:: `bm3d <https://pypi.org/project/bm3d>`__
.. |cdmsc| replace:: `colour_demosaicing <https://github.com/colour-science/colour-demosaicing>`__


=================  ======================================================
Optional Package   Features Supported
=================  ======================================================
|numexpr|          Acceleration of some functions in :mod:`sporco.linalg`
|mpldatacursor|    Data cursor enabled for :func:`.plot.plot`,
		   :func:`.plot.contour`, and :func:`.plot.imview`
|cupy|             GPU acceleration of modules in :mod:`sporco.cupy`
|wrltzr|           Utility that supports capture of :mod:`sporco.cuda`
		   function output within Jupyter notebooks
|gputil|           Additional utility functions in :mod:`sporco.cupy`
|mpi4py|           Parallel computation of the grid search in
		   :mod:`sporco.mpiutil`
|bm3d|             Required by :ref:`demo scripts <examples_ppp_index>`
		   for :mod:`.admm.ppp` and :mod:`.pgm.ppp`
|cdmsc|            Required by :ref:`demo scripts <examples_ppp_index>`
		   for :mod:`.admm.ppp` and :mod:`.pgm.ppp`
=================  ======================================================
