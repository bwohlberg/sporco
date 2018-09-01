import matplotlib as mpl
mpl.use('Agg')

try:
    import importlib.util
except ImportError:
    collect_ignore = ['cupy']
