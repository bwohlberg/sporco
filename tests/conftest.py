import platform
import matplotlib as mpl
mpl.use('Agg')


collect_ignore = []

# Ignore sporco.cupy tests if importlib.util or cupy not available
try:
    import importlib.util
except ImportError:
    collect_ignore.append('cupy')
else:
    try:
        import cupy
    except ImportError:
        collect_ignore.append('cupy')

# Ignore sporco.mpiutil tests if mpi4py not available
try:
    import mpi4py
except ImportError:
    collect_ignore.append('test_mpiutil.py')

# Ignore tests of modules depending on multiprocessing fork ability
# on Windows and MacOS platform
if platform.system() == 'Windows' or platform.system() == 'Darwin':
    collect_ignore.extend(['admm/test_parcbpdn.py',
                           'dictlrn/test_prlcnscdl.py'])
