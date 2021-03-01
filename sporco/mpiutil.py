# -*- coding: utf-8 -*-
# Copyright (C) 2017-2021 by Cristina Garcia-Cardona <cgarciac@lanl.gov>
#                            Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""Utility functions that make use of MPI for parallel computing."""

from __future__ import absolute_import, division, print_function
from builtins import range

from mpi4py import MPI
import itertools
import numpy as np


__author__ = """\n""".join(['Cristina Garcia-Cardona <cgarciac@lanl.gov>',
                            'Brendt Wohlberg <brendt@ieee.org>'])


__all__ = ['grid_search']


def _get_rank_limits(comm, arrlen):
    """Determine the chunk of the grid that has to be computed per
    process. The grid has been 'flattened' and has arrlen length. The
    chunk assigned to each process depends on its rank in the MPI
    communicator.

    Parameters
    ----------
    comm : MPI communicator object
      Describes topology of network: number of processes, rank
    arrlen : int
      Number of points in grid search.

    Returns
    -------
    begin : int
      Index, with respect to 'flattened' grid, where the chunk
      for this process starts.
    end : int
      Index, with respect to 'flattened' grid, where the chunk
      for this process ends.
    """

    rank = comm.Get_rank()  # Id of this process
    size = comm.Get_size()  # Total number of processes in communicator
    end = 0
    # The scan should be done with ints, not floats
    ranklen = int(arrlen / size)
    if rank < arrlen % size:
        ranklen += 1
    # Compute upper limit based on the sizes covered by the processes
    # with less rank
    end = comm.scan(sendobj=ranklen, op=MPI.SUM)
    begin = end - ranklen

    return (begin, end)



def grid_search(fn, grid, comm=None, mpidtype=None, fmin=True):
    """
    Grid search for optimal parameters of a specified function.

    Perform a grid search for optimal parameters of a specified
    function. In the simplest case the function returns a float value,
    and a single optimum value and corresponding parameter values are
    identified. If the function returns a tuple of values, each of
    these is taken to define a separate function on the search grid,
    with optimum function values and corresponding parameter values
    being identified for each of them. The computation of the function
    at the grid points is computed in parallel using MPI, as opposed
    to :func:`.util.grid_search`, which uses :mod:`multiprocessing`
    for parallelisation.

    The `mpi4py <https://mpi4py.readthedocs.io>`__ package is
    required for use of the ``mpiutil.grid_search`` function. It is also
    necessary to run Python with the ``mpiexec`` command; for example,
    if ``mpiscript.py`` calls this function, use::

      mpiexec -n 8 python mpiscript.py

    to distribute the grid search over 8 processors.


    Parameters
    ----------
    fn : function
      Function to be evaluated. It should take a tuple of parameter
      values as an argument, and return a float value or a tuple of
      float values.
    grid : tuple of array_like
      A tuple providing an array of sample points for each axis of the
      grid on which the search is to be performed.
    comm : MPI communicator object, optional (default None)
      Topology of network (number of processes and rank). If None,
      ``MPI.COMM_WORLD`` is used.
    mpidtype : MPI data type, optional (default None)
      Desired data type for consolidation operations. If None,
      ``MPI.DOUBLE`` is used
    fmin : bool, optional (default True)
      Determine whether optimal function values are selected as minima
      or maxima. If `fmin` is True then minima are selected.

    Returns
    -------
    sprm : ndarray
      Optimal parameter values on each axis. If `fn` is multi-valued,
      `sprm` is a matrix with rows corresponding to parameter values
      and columns corresponding to function values.
    sfvl : float or ndarray
      Optimum function value or values
    fvmx : ndarray
      Function value(s) on search grid
    sidx : tuple of int or tuple of ndarray
      Indices of optimal values on parameter grid
    """

    if comm is None:
        comm = MPI.COMM_WORLD
    if mpidtype is None:
        mpidtype = MPI.DOUBLE
    fprm = itertools.product(*grid)

    # Distribute computation among processes in MPI communicator
    afprm = np.asarray(list(fprm))  # Faster to communicate array data
    iterlen = afprm.shape[0]
    begin, end = _get_rank_limits(comm, iterlen)
    rankgrid = (afprm[begin:end, :])
    rankfval = np.asarray(list(map(fn, rankgrid)))

    if rankfval.ndim == 1:  # Function with a scalar return value
        fval = np.empty(afprm.shape[0])
        comm.Allgatherv([rankfval, mpidtype], [fval, mpidtype])
    else:  # Function with a vector return value
        # Vector for gathering all the results
        fval = np.empty([afprm.shape[0], rankfval.shape[1]])
        # Size of values to collect locally
        sizeL = np.array(np.prod(np.array(rankfval.shape)))
        # Number of processes to collect from
        sizeW = comm.Get_size()
        # Vector to collect the sizes from each process
        sizes = np.zeros(sizeW, dtype=int)
        comm.Allgather([sizeL, MPI.INT], [sizes, MPI.INT])
        # Vector to collect the offsets for each process
        offsets = np.zeros(sizeW, dtype=int)
        offsets[1:] = np.cumsum(sizes)[:-1]
        # Collecting variable size of vector return values
        comm.Allgatherv(rankfval, [fval, sizes, offsets, mpidtype])


    # Proceed as regular grid_search (all processes execute this)
    if fmin:
        slct = np.nanargmin
    else:
        slct = np.nanargmax

    if isinstance(fval[0], (tuple, list, np.ndarray)):
        nfnv = len(fval[0])
        fvmx = np.reshape(fval, [a.size for a in grid] + [nfnv,])
        sidx = np.unravel_index(slct(fvmx.reshape((-1, nfnv)), axis=0),
                                fvmx.shape[0:-1]) + (np.array((range(nfnv))),)
        sprm = np.array([grid[k][sidx[k]] for k in range(len(grid))])
        sfvl = tuple(fvmx[sidx])
    else:
        fvmx = np.reshape(fval, [a.size for a in grid])
        sidx = np.unravel_index(slct(fvmx), fvmx.shape)
        sprm = np.array([grid[k][sidx[k]] for k in range(len(grid))])
        sfvl = fvmx[sidx]

    return sprm, sfvl, fvmx, sidx
