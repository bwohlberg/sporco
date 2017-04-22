====================
SPORCO Release Notes
====================


Version 0.1.5   (2017-04-22)
----------------------------

- Fixed serious bug in cbpdn.ConvBPDNGradReg.setdict and
  cbpdn.ConvBPDNGradReg.xstep resulting in incorrect solution of
  linear system
- Fixed bug in cbpdn.GenericConvBPDN.xstep (and same method in some
  derived classes) affecting calculation of linear solver accuracy for
  single-channel dictionaries
- Fixed bug in multi-channel data handling in cbpdn.AddMaskSim
- Fixed bug in util.netgetdata
- New functions linalg.solvedbd_sm, linalg.solvedbd_sm_c
- Improved documentation of admm.admm module
- Changed default line width in plot.plot and added parameter for
  specifying label padding to plot.surf
- Improved capabilities of util.Timer class and modified admm.ADMM
  class to use it
- New FastSolve option instructs admm.ADMM class to skip
  non-essential calculations
- New AccurateDFid option for more accurate functional evaluation in
  admm.BPDNDictLearn and admm.ConvBPDNDictLearn
- New IterTimer option to select timer used for admm.ADMM iteration
  timing
- Introduced new inner product function linalg.inner and improved
  speed of linalg.solvedbi_sm by using it instead of np.sum and
  broadcast multiplication



Version 0.1.4   (2017-03-03)
----------------------------

- Bug fix release to correct error in Travis CI configuration
  resulting in PyPI releases with broken plotting capabilities



Version 0.1.3   (2017-03-03)
----------------------------

- Major changes to policy of downloading required data on package
  build: this functionality is now in script sporco_get_images, which
  is not called during package build
- New function util.netgetdata
- Major changes to util.ExampleImages
- Bug fix for multi-channel images in bpdn.AddMaskSim
- Improved handling of floating point images in plot.imview


Version 0.1.2   (2017-02-19)
----------------------------

- New functions util.ntpl2array, util.array2ntpl, plot.close
- Modified util.rgb2gray to support array containing multiple images
- Modified scaling of return value of linalg.fl2norm2 to match docs
- In module linalg, moved functions mae, mse, snr, and psnr to new
  module metric, and added new functions isnr, bsnr, pamse, and gmsd
  in this module
- New methods admm.ADMM.getmin, cbpdn.AddMaskSim.setdict,
  cbpdn.AddMaskSim.getcoef
- Modified classes in modules tvl1 and tvl2 to support Vector TV for
  multi-channel images
- Added Jypyter Notebook versions of some example scripts
- Added some new example scripts



Version 0.1.1   (2016-11-27)
----------------------------

- Moved plotting functions from util to new module plot
- New function util.grid_search supporting parallel processing
  evaluation of a function on a specified grid
- Extended capabilities of class util.ExampleImages
- New functions linalg.GradientFilters, linalg.promote16, linalg.roll,
  linalg.blockcirculant, linalg.mae
- Modified admm.ADMM class so that objects of this type can be pickled
- Changes to interface of admm.ADMM.__init__,
  admm.ADMM.iteration_stats, admm.ADMM.display_status,
  admm.ADMMEqual.__init__, admm.ADMMTwoBlockCnstrnt.__init__
- New methods admm.ADMM.set_dtype, admm.ADMM.set_attr,
  admm.ADMM.yinit, admm.ADMM.uinit, admm.ADMM.itstat_fields,
  admm.ADMM.hdrtxt, admm.ADMM.hdrval, admm.ADMM.itstat_extra,
  admm.ADMM.var_u
- In admm.ADMM and derived classes, major changes to object
  initialisation and iteration stats calculation mechanisms, including
  more careful initialisation of arrays to ensure consistent dtype
  across all working variables
- In module bpdn, created new common base class GenericBPDN
- In module cbpdn, created new common base class GenericConvBPDN
- Improvements to docs



Version 0.1.0   (2016-08-28)
----------------------------

- New module admm.dictlrn as base class for classes in admm.bpdndl and
  admm.cbpdndl
- New methods, admm.admm.ADMM.getitstat, admm.bpdn.getcoef,
  admm.cbpdn.getcoef, admm.cmod.getdict, admm.ccmod.getdict
- New classes admm.admm.ADMMTwoBlockCnstrnt, admm.bpdn.BPDNJoint,
  admm.cbpdn.ConvBPDNJoint, admm.cbpdn.ConvBPDNGradReg,
  admm.ccmod.DictionarySize, admm.ccmod.ConvRepIndexing
  admm.cbpdn.ConvBPDNMaskDcpl, admm.cbpdn.AddMaskSim
- New functions linalg.shrink12, linalg.proj_l2ball
- In admm.bpdn, moved functions factorise and linsolve into linalg
  module as lu_factor and lu_solve_ATAI respectively
- In admm.cmod, moved function factorise and linsolve into linalg
  module as lu_factor and lu_solve_AATI respectively
- Fixed multi-channel data handling problems in admm.cbpdn and
  admm.ccmod
- Bug fix in util.tiledict
- New global variable linalg.pyfftw_threads determining the number of
  threads used by pyFFTW
- Renamed util.zquotient to util.zdivide and improved implementation
- Header text for ADMM algorithms run in verbose mode is now in utf8
  encoding
- Moved example scripts into subdirectories indicating example
  categories
- Improvements to documentation



Version 0.0.4   (2016-06-14)
----------------------------

- In admm.admm.ADMM, modified relax_AX and compute_residuals methods
  for correct handling of relaxed and unrelaxed versions of X variable
- Improvements to plotting functions in util, including support for
  mpldatacursor if installed
- Minor improvements to docs


Version 0.0.3   (2016-06-05)
----------------------------

- Changed pyFFTW wrapper functions in linalg for compatibility with
  new interfaces introduced in pyFFTW 0.10.2
- Added new 3D convolutional dictionary learning example
  demo_cbpdndl_vid.py
- A number of bug fixes
- Improvements to docs



Version 0.0.2   (2016-05-27)
----------------------------

- Package modified for compatibility with Python 2 and 3
- New functions: util.complex_dtype, util.pyfftw_empty_aligned
- In admm.bpdn.BPDN and admm.cbpdn.ConvBPDN, introduced new
  NonNegCoef option
- New class admm.cbpdn.ConvRepIndexing
- Improvements to documentation
- Improvements to package configuration and metadata.
- Moved package version number into sporco/__init__.py



Version 0.0.1   (2016-04-21)
----------------------------

- Initial release
