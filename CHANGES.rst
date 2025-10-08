====================
SPORCO Release Notes
====================


Version 0.2.3   (yyyy-mm-dd)
----------------------------

• No changes yet  


Version 0.2.2   (2025-10-08)
----------------------------

• Removed deprecated sporco.fista modules
• Removed deprecation warning redirects for functions renamed in 0.2.0
• Resolved a number of deprecation warnings
• Fixed bugs in a number of example scripts


Version 0.2.1   (2022-02-17)
----------------------------

• Added support for complex signals in admm.cbpdn, admm.ccmod, admm.tvl1,
  and admm.tvl2 modules
• Fixed bug in admm.cbpdnin when used with multi-signal arrays (K > 1)


Version 0.2.0   (2021-03-01)
----------------------------

• Major support module restructuring: numerous functions from sporco.util
  and sporco.linalg moved to new modules sporco.array, sporco.fft,
  and sporco.signal, and functions nkp, kpsvd, tikhonov_filter,
  gaussian, and local_contrast_normalise moved from sporco.util to
  sporco.linalg
• Added new functions prox.norm_dl1l2 and prox.prox_dl1l2 for difference of
  ℓ1 and ℓ2 norms and corresponding proximal operator
• Major restructuring of sporco.fista modules, now renamed to sporco.pgm
• Significant change to interface of fft.fftconv function
• New classes for sparse coding and dictionary learning with a weighted ℓ2
  data fidelity term
• Functionality depending on use of fork in multiprocessing (modules
  admm.parcbpdn and dictlrn.prlcnscdl, and parallel computation of
  util.grid_search) no longer supported under MacOS



Version 0.1.12   (2020-02-08)
-----------------------------

• Improved run time and memory usage of tikhonov_filter function
• Fixed bug in functional value calculation in class
  admm.pdcsc.ConvProdDictL1L1GrdJoint
• Minimum required SciPy version is now 0.19.1
• Renamed prox.prox_l1l2 to prox_sl1l2
• Added new modules and example scripts for the Plug and Play Priors method
• New functions for nearest Kronecker product, Kronecker product SVD in linalg
  module, and new functions for least absolute deviations linear regression,
  and least maximum error linear regression in interp module
• Renamed linalg.GradientFilters to linalg.gradient_filters, linalg.Gax to
  linalg.grad, linalg.GTax to linalg.gradT, util.extractblocks to
  util.extract_blocks, util.averageblocks to util.average_blocks, and
  util.combineblocks to util.combine_blocks
• Moved util.pca to linalg.pca



Version 0.1.11   (2019-04-14)
-----------------------------

• New module mpiutil (MPI utilities)
• New module admm.pdcsc (CSC with a product dictionary)
• New solver class admm.cbpdn.ConvL1L1Grd for CSC with an ℓ1 data
  fidelity term
• New solver class admm.cbpdn.MultiDictConvBPDN for coupled sparse
  coding with multiple dictionaries
• Additional solvers supported for use with CuPy
• Added support for robust variant of FISTA
• Switch to using imageio instead of scipy.misc for image read/write
• Fixed bug in zm parameter handling in cnvrep.getPcn
• Improved example index structure in docs
• Various minor fixes and improvements



Version 0.1.10   (2018-11-09)
-----------------------------

• Significant changes to online CDL module, including addition of support
  for masked online CDL problem, and optional use of CUDA accelerated
  CBPDN solver
• Added support for GPU acceleration of selected solvers via the CuPy
  package
• Fixed a bug in iteration statistics constructions in cbpdndl and
  cbpdndlmd modules
• Added support for mouse scroll wheel zooming in plot.plot, plot.contour,
  and plot.imview
• Major changes to docs structure and API documentation build mechanism



Version 0.1.9   (2018-07-15)
----------------------------

• Added new sub-package for FISTA algorithms, including algorithms for
  the CBPDN and CCMOD problems
• Moved all dictionary learning modules into new dictlrn sub-package and
  added new module for online CDL
• Simplified array shape requirements for option L1Weight in modules
  admm.cbpdn and admm.cbpdntv
• Minimum required NumPy version is now 1.11
• Added sporco.cuda interface to CUDA extension package sporco-cuda
• Completely restructured example scripts, which are now used to generate
  example Jupyter notebooks and corresponding docs pages
• Modifications to the interfaces of the plot module functions that will
  break existing code that uses the fgrf or axrf parameters
• Modifications to the interface of the plot.plot function that will
  break existing code that uses the lwidth, lstyle, msize, or mstyle
  parameters
• Replaced function util.imageblocks with util.extractblocks, and introduced
  new functions util.averageblocks and util.combineblocks
• Added new module admm.parcbpdn implementing the parallel ADMM CBPDN
  solver
• Fixed bugs in cbpdn.ConvBPDNProjL1 and cbpdn.ConvMinL1InL2Ball



Version 0.1.8   (2017-11-04)
----------------------------

• Added parallel processing implementation of convolutional dictionary
  learning algorithm based on the hybrid Mask Decoupling/Consensus
  dictionary update to module parcnsdl
• Fixed bug in package setup that resulted in example image files being
  omitted from package distributions



Version 0.1.7   (2017-10-15)
----------------------------

• New module cbpdntv with classes for CBPDN with additional Total
  Variation regularization terms
• Fixed bug in object initialisation timing
• Changed problematic image URLs in bin/sporco_get_images
• Added installation instructions for Mac OS and Windows
• Minimum required NumPy version is now 1.10
• Fixed bugs in admm.ccmod module
• Test images required by usage examples are now included with the package
• Modifications to util.ExampleImages interface (these changes will break
  code that uses the previous interface)
• Added call graph diagrams for many classes (see `Notes` in the package
  documentation)
• New class admm.ADMMConsensus for ADMM consensus problems
• Major changes to ccmod module, including restructuring class hierarchy,
  a new ADMM consensus solver, and moving of ccmod.ConvCnstrMODMaskDcpl to
  a separate module
• Changes to cbpdndl module related to ccmod module restructuring
• New module ccmodmd supporting multiple algorithms for the dictionary
  update with mask decoupling
• New module parcnsdl with a parallel processing implementation of
  convolutional dictionary learning with the ADMM consensus dictionary
  update
• Moved class ConvRepIndexing from cbpdn and ccmod modules to new module
  cnvrep. Additional classes from ccmod module also moved into cnvrep.
• New module prox supporting evaluation of various norms and their proximal
  and projection operators
• New classes bpdn.BPDNProjL1, bpdn.MinL1InL2Ball, cbpdn.ConvBPDNProjL1,
  and cbpdn.ConvMinL1InL2Ball supporting constrained forms of the BPDN
  and CBPDN problems



Version 0.1.6   (2017-05-22)
----------------------------

• Fixed functional evaluation error in cbpdn.ConvBPDNMaskDcpl
• Fixed bug in cbpdn.ConvTwoBlockCnstrnt with multi-channel dictionary
• New class ccmod.ConvCnstrMODMaskDcpl for dictionary update with mask
  decoupling
• New class cbpdndl.ConvBPDNMaskDcplDictLearn for dictionary learning
  with mask decoupling
• Corrected serious error in demo_dictlrn_cbpdn_md.py
• Fixed bug causing non-deterministic 'AuxVarObj' option behaviour
• New functions util.transpose_ntpl_list, util.complex_randn,
  util.idle_cpu_count
• In cmod and ccmod modules, renamed sparse representation variable from A
  to Z
• Changed callback function mechanism in admm.ADMM.solve and
  dictlrn.DictLearn.solve: callback function no longer takes iteration number
  as an argument (it is not available as a class attribute), and can terminate
  solve iterations by returning a boolean True value.
• New parameters in plot.plot for selecting marker size and style, and in
  plot.imview for specifying matplotlib.colors.Normalize object
• Added L21Weight option for cbpdn.ConvBPDNJoint
• Fixed bug in cbpdn.AddMaskSim handling of multi-channel dictionaries



Version 0.1.5   (2017-04-22)
----------------------------

• Fixed serious bug in cbpdn.ConvBPDNGradReg.setdict and
  cbpdn.ConvBPDNGradReg.xstep resulting in incorrect solution of
  linear system
• Fixed bug in cbpdn.GenericConvBPDN.xstep (and same method in some
  derived classes) affecting calculation of linear solver accuracy for
  single-channel dictionaries
• Fixed bug in multi-channel data handling in cbpdn.AddMaskSim
• Fixed bug in util.netgetdata
• New functions linalg.solvedbd_sm, linalg.solvedbd_sm_c
• Improved documentation of admm.admm module
• Changed default line width in plot.plot and added parameter for
  specifying label padding to plot.surf
• Improved capabilities of util.Timer class and modified admm.ADMM
  class to use it
• New FastSolve option instructs admm.ADMM class to skip
  non-essential calculations
• New AccurateDFid option for more accurate functional evaluation in
  admm.BPDNDictLearn and admm.ConvBPDNDictLearn
• New IterTimer option to select timer used for admm.ADMM iteration
  timing
• Introduced new inner product function linalg.inner and improved
  speed of linalg.solvedbi_sm by using it instead of np.sum and
  broadcast multiplication



Version 0.1.4   (2017-03-03)
----------------------------

• Bug fix release to correct error in Travis CI configuration
  resulting in PyPI releases with broken plotting capabilities



Version 0.1.3   (2017-03-03)
----------------------------

• Major changes to policy of downloading required data on package
  build: this functionality is now in script sporco_get_images, which
  is not called during package build
• New function util.netgetdata
• Major changes to util.ExampleImages
• Bug fix for multi-channel images in bpdn.AddMaskSim
• Improved handling of floating point images in plot.imview


Version 0.1.2   (2017-02-19)
----------------------------

• New functions util.ntpl2array, util.array2ntpl, plot.close
• Modified util.rgb2gray to support array containing multiple images
• Modified scaling of return value of linalg.fl2norm2 to match docs
• In module linalg, moved functions mae, mse, snr, and psnr to new
  module metric, and added new functions isnr, bsnr, pamse, and gmsd
  in this module
• New methods admm.ADMM.getmin, cbpdn.AddMaskSim.setdict,
  cbpdn.AddMaskSim.getcoef
• Modified classes in modules tvl1 and tvl2 to support Vector TV for
  multi-channel images
• Added Jupyter Notebook versions of some example scripts
• Added some new example scripts



Version 0.1.1   (2016-11-27)
----------------------------

• Moved plotting functions from util to new module plot
• New function util.grid_search supporting parallel processing
  evaluation of a function on a specified grid
• Extended capabilities of class util.ExampleImages
• New functions linalg.GradientFilters, linalg.promote16, linalg.roll,
  linalg.blockcirculant, linalg.mae
• Modified admm.ADMM class so that objects of this type can be pickled
• Changes to interface of admm.ADMM.__init__,
  admm.ADMM.iteration_stats, admm.ADMM.display_status,
  admm.ADMMEqual.__init__, admm.ADMMTwoBlockCnstrnt.__init__
• New methods admm.ADMM.set_dtype, admm.ADMM.set_attr,
  admm.ADMM.yinit, admm.ADMM.uinit, admm.ADMM.itstat_fields,
  admm.ADMM.hdrtxt, admm.ADMM.hdrval, admm.ADMM.itstat_extra,
  admm.ADMM.var_u
• In admm.ADMM and derived classes, major changes to object
  initialisation and iteration stats calculation mechanisms, including
  more careful initialisation of arrays to ensure consistent dtype
  across all working variables
• In module bpdn, created new common base class GenericBPDN
• In module cbpdn, created new common base class GenericConvBPDN
• Improvements to docs



Version 0.1.0   (2016-08-28)
----------------------------

• New module admm.dictlrn as base class for classes in admm.bpdndl and
  admm.cbpdndl
• New methods, admm.admm.ADMM.getitstat, admm.bpdn.getcoef,
  admm.cbpdn.getcoef, admm.cmod.getdict, admm.ccmod.getdict
• New classes admm.admm.ADMMTwoBlockCnstrnt, admm.bpdn.BPDNJoint,
  admm.cbpdn.ConvBPDNJoint, admm.cbpdn.ConvBPDNGradReg,
  admm.ccmod.DictionarySize, admm.ccmod.ConvRepIndexing
  admm.cbpdn.ConvBPDNMaskDcpl, admm.cbpdn.AddMaskSim
• New functions linalg.shrink12, linalg.proj_l2ball
• In admm.bpdn, moved functions factorise and linsolve into linalg
  module as lu_factor and lu_solve_ATAI respectively
• In admm.cmod, moved function factorise and linsolve into linalg
  module as lu_factor and lu_solve_AATI respectively
• Fixed multi-channel data handling problems in admm.cbpdn and
  admm.ccmod
• Bug fix in util.tiledict
• New global variable linalg.pyfftw_threads determining the number of
  threads used by pyFFTW
• Renamed util.zquotient to util.zdivide and improved implementation
• Header text for ADMM algorithms run in verbose mode is now in utf8
  encoding
• Moved example scripts into subdirectories indicating example
  categories
• Improvements to documentation



Version 0.0.4   (2016-06-14)
----------------------------

• In admm.admm.ADMM, modified relax_AX and compute_residuals methods
  for correct handling of relaxed and unrelaxed versions of X variable
• Improvements to plotting functions in util, including support for
  mpldatacursor if installed
• Minor improvements to docs


Version 0.0.3   (2016-06-05)
----------------------------

• Changed pyFFTW wrapper functions in linalg for compatibility with
  new interfaces introduced in pyFFTW 0.10.2
• Added new 3D convolutional dictionary learning example
  demo_cbpdndl_vid.py
• A number of bug fixes
• Improvements to docs



Version 0.0.2   (2016-05-27)
----------------------------

• Package modified for compatibility with Python 2 and 3
• New functions: util.complex_dtype, util.pyfftw_empty_aligned
• In admm.bpdn.BPDN and admm.cbpdn.ConvBPDN, introduced new
  NonNegCoef option
• New class admm.cbpdn.ConvRepIndexing
• Improvements to documentation
• Improvements to package configuration and metadata.
• Moved package version number into sporco/__init__.py



Version 0.0.1   (2016-04-21)
----------------------------

• Initial release
