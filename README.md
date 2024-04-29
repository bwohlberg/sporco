[![Supported Python Versions](https://img.shields.io/pypi/pyversions/sporco.svg)](https://github.com/bwohlberg/sporco)
[![Package License](https://img.shields.io/github/license/bwohlberg/sporco.svg)](https://github.com/bwohlberg/sporco/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/sporco/badge/?version=latest)](http://sporco.readthedocs.io/en/latest/?badge=latest)
[![Test status](https://github.com/bwohlberg/sporco/actions/workflows/pytest.yml/badge.svg)](https://github.com/bwohlberg/sporco/actions/workflows/pytest.yml)
[![Test Coverage](https://codecov.io/gh/bwohlberg/sporco/branch/master/graph/badge.svg)](https://codecov.io/gh/bwohlberg/sporco)\
[![PyPi Release](https://badge.fury.io/py/sporco.svg)](https://badge.fury.io/py/sporco)
[![PyPi Downloads](https://static.pepy.tech/personalized-badge/sporco?period=total&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/sporco)
[![Conda Forge Release](https://img.shields.io/conda/vn/conda-forge/sporco.svg)](https://anaconda.org/conda-forge/sporco)
[![Conda Forge Downloads](https://img.shields.io/conda/dn/conda-forge/sporco.svg)](https://anaconda.org/conda-forge/sporco)\
[![Binder](http://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/bwohlberg/sporco-notebooks/master?filepath=index.ipynb)
[![DOI](https://img.shields.io/badge/DOI-10.25080%2Fshinma--7f4c6e7--001-blue.svg)](https://dx.doi.org/10.25080/shinma-7f4c6e7-001)


# SParse Optimization Research COde (SPORCO)

SPORCO is a Python package for solving optimisation problems with sparsity-inducing regularisation. These consist primarily of sparse coding and dictionary learning problems, including convolutional sparse coding and dictionary learning, but there is also support for other problems such as Total Variation regularisation and Robust PCA. The optimisation algorithms in the current version are based on the Alternating Direction Method of Multipliers (ADMM) or on the Proximal Gradient Method (PGM).

If you use this software for published work, please [cite it](http://sporco.readthedocs.io/en/latest/overview.html#citing).


# Documentation

[Documentation](http://sporco.rtfd.io/) is available online, or can be built from the root directory of the source distribution by the command

	python setup.py build_sphinx

in which case the HTML documentation can be found in the `build/sphinx/html` directory (the top-level document is `index.html`).  Although the SPORCO package itself is compatible with Python 3.x, building the documention requires Python 3.3 or later due to the use of [Jonga](https://github.com/bwohlberg/jonga) to construct call graph images for the SPORCO optimisation class hierarchies.

An overview of the package design and functionality is also available in

> Brendt Wohlberg, [SPORCO: A Python package for standard and convolutional sparse representations](http://conference.scipy.org/proceedings/scipy2017/brendt_wohlberg.html),
> in Proceedings of the 15th Python in Science Conference, (Austin, TX, USA), doi:10.25080/shinma-7f4c6e7-001, pp. 1--8, Jul 2017


# Usage

Scripts illustrating usage of the package can be found in the `examples` directory of the source distribution. These examples can be run from the root directory of the package by, for example

	python examples/scripts/sc/bpdn.py

To run these scripts prior to installing the package it will be necessary to first set the `PYTHONPATH` environment variable to include the root directory of the package. For example, in a `bash` shell

	export PYTHONPATH=$PYTHONPATH:`pwd`

from the root directory of the package.

[Jupyter Notebook](http://jupyter.org/) examples are also [available](https://github.com/bwohlberg/sporco-notebooks). These examples can be viewed online via [nbviewer](https://nbviewer.jupyter.org/github/bwohlberg/sporco-notebooks/blob/master/index.ipynb), or run interactively at [binder](https://mybinder.org/v2/gh/bwohlberg/sporco-notebooks/master?filepath=index.ipynb).


# Requirements

The primary requirements are Python itself, and modules [future](http://python-future.org), [numpy](http://www.numpy.org), [scipy](https://www.scipy.org), [imageio](https://imageio.github.io/), [pyfftw](https://hgomersall.github.io/pyFFTW), and [matplotlib](http://matplotlib.org). Module [numexpr](https://github.com/pydata/numexpr) is not required, but some functions will be faster if it is installed. If module [mpldatacursor](https://github.com/joferkington/mpldatacursor) is installed, functions `plot.plot`, `plot.contour`, and `plot.imview` will support the data cursor that it provides.

Instructions for installing these requirements are provided in the [Requirements](http://sporco.rtfd.io/en/latest/install.html#requirements) section of the package documentation.


# Installation

To install the most recent release of SPORCO from [PyPI](https://pypi.python.org/pypi/sporco/) do

	pip install sporco

The [development version](https://github.com/bwohlberg/sporco) on GitHub can be installed by doing

	pip install git+https://github.com/bwohlberg/sporco

or by doing

	git clone https://github.com/bwohlberg/sporco.git

followed by

	cd sporco
	python setup.py build
	python setup.py install

The install commands will usually have to be performed with root privileges.

SPORCO can also be installed as a [conda](https://conda.io/docs/) package from the [conda-forge](https://conda-forge.org/) channel

	conda install -c conda-forge sporco

A summary of the most significant changes between SPORCO releases can be found in the `CHANGES.rst` file. It is strongly recommended to consult this summary when updating from a previous version.


# Extensions

Some additional components of SPORCO are made available in separate repositories:

-   [SPORCO-CUDA](https://github.com/bwohlberg/sporco-cuda):
	GPU-accelerated versions of selected convolutional sparse coding
	algorithms
-   [SPORCO Notebooks](https://github.com/bwohlberg/sporco-notebooks):
	Jupyter Notebook versions of the example scripts distributed with
	SPORCO
-   [SPORCO Extra](https://github.com/bwohlberg/sporco-extra):
	Additional examples, data, and contributed code

# License

SPORCO is distributed as open-source software under a BSD 3-Clause License (see the `LICENSE` file for details).
