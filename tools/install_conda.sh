#! /bin/bash

# This script installs a conda environment with all required and optional
# sporco dependencies, including cupy if a GPU is available. This script
# should function correctly under both Linux and OSX, but note that there
# are some additional complications in using a conda installed matplotlib
# under OSX
#   https://matplotlib.org/faq/osx_framework.html
# that are not addressed.
#
# Usage: install_conda.sh install_path

URLROOT=https://repo.continuum.io/miniconda/
INSTLINUX=Miniconda3-latest-Linux-x86_64.sh
INSTMACOSX=Miniconda3-latest-MacOSX-x86_64.sh

os=`uname -a | cut -d ' ' -f 1`
case "$os" in
    Linux)    SOURCEURL=$URLROOT$INSTLINUX;;
    Darwin)   SOURCEURL=$URLROOT$INSTMACOSX;;
    *)        echo "Error: unsupported operating system $os"; exit 1;;
esac

if [ ! "`which wget 2>/dev/null`" ]; then
    echo "Error: wget utility required but not found" >&2
    exit 2
fi

INSTALLROOT=$1
if [ ! -d "$INSTALLROOT" -o ! -w "$INSTALLROOT" ]; then
    echo "Error: installation root path \"$INSTALLROOT\" is not a directory "\
	 "or is not writable"
    exit 3
fi
CONDAHOME=$INSTALLROOT/miniconda3

if [ -d "$CONDAHOME" ]; then
    echo "Error: miniconda3 installation directory $CONDAHOME already exists"
    exit 4
fi

read -r -p "Confirm conda installation in root path $INSTALLROOT [y/N] "\
     cnfrm
if [ "$cnfrm" != 'y' -a "$cnfrm" != 'Y' ]; then
    echo "Cancelling installation"
    exit 5
fi


# Get miniconda bash archive and install it
wget $SOURCEURL -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $CONDAHOME

export PATH="$CONDAHOME/bin:$PATH"
hash -r
conda config --set always_yes yes
conda install -n root _license
conda update -q conda
conda info -a
conda config --add channels conda-forge
conda config --append channels bjornfjohansson # For mpldatacursor

conda create -n py37 python=3.7

source activate py37
conda install future numpy scipy imageio pyfftw numexpr matplotlib \
      mpldatacursor psutil pytest pytest-runner ipython jupyter sphinx \
      pygraphviz numpydoc cython mpi4py pyhamcrest sphinxcontrib-bibtex \
      jonga pypandoc libopenblas

# ffmpeg is required by imageio for reading mp4 video files
# it can also be installed via the system package manager, .e.g.
#   sudo apt install ffmpeg
if [ "`which ffmpeg`" = '' ]; then
    conda install ffmpeg
fi

pip install sphinx_tabs sphinx_fontawesome py2jn bm3d


# Install cupy and gputil if nvcc is present (indicating that the
# cuda toolkit is installed)
cupy_installed='False'
if [ "`which nvcc`" != '' ]; then
    cudaversion=`nvcc --version | grep -o 'release [0-9][0-9]*\.[[0-9][0-9]*' \
                                | sed -e 's/release //' -e 's/\.//'`
    source deactivate
    conda create --name py37cu --clone py37
    source activate py37cu
    conda install wurlitzer
    pip install gputil
    # Conda cupy package is currently only version 4.x
    # conda install cupy
    if [ "$os" = 'Linux' ]; then
	# Install pre-compiled wheel
        pip install cupy-cuda$cudaversion
    else
	# Build from source
	pip install cupy
    fi
    retval=$?
    if [ $retval -eq 0 ]; then
        cupy_installed='True'
    else
        echo "WARNING: cupy installation failed"
    fi
fi


echo "Add the following to your .bashrc or .bash_aliases file"
echo "  export CONDAHOME=$CONDAHOME"
echo "  export PATH=\$PATH:\$CONDAHOME/bin"
echo "Activate the conda environment with the command"
echo "  source activate py37"
if [ "$cupy_installed" = 'True' ]; then
    echo "or"
    echo "  source activate py37cu"
fi
echo "The environment can be deactivated with the command"
echo "  conda deactivate"
