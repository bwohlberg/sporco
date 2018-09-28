import matplotlib as mpl
mpl.use('Agg')


collect_ignore = []
try:
    import importlib.util
except ImportError:
    collect_ignore.append('cupy')
try:
    import cupy
except ImportError:
    if 'cupy' not in collect_ignore:
        collect_ignore.append('cupy')
try:
    import mpi4py
except ImportError:
    collect_ignore.append('test_mpiutil.py')
