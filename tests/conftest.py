import matplotlib as mpl
mpl.use('Agg')

try:
    import importlib.util
except:
    collect_ignore = ['cupy']
