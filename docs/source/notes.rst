Notes
=====

* Given the number of usage examples for the :doc:`inverse problems
  <invprob>`, it has not been feasible to optimise the algorithm
  options in all cases. While these examples should contain reasonable
  choices, they should not be assumed to be optimal.
* There is a `bug <https://github.com/pyFFTW/pyFFTW/issues/135>`_ in
  :mod:`pyfftw` that can lead to programs hanging if used within
  processes created by :mod:`multiprocessing`. A work-around is to
  disable multi-threading for the :mod:`pyfftw`-based FFT functions in
  :mod:`sporco.linalg` by including the following code::

      import sporco.linalg
      sporco.linalg.pyfftw_threads = 1

* When run with option `Verbose` enabled, the :doc:`inverse problems
  <invprob>` generate output in utf8 encoding, which may result in an
  error when piping the output to a file. The simplest solution is to
  define the environment variable ``PYTHONIOENCODING`` to ``utf-8``.
  For example, in a ``bash`` shell::

      export PYTHONIOENCODING=utf-8
