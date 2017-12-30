# -*- coding: utf-8 -*-

"""Interface to the sporco-cuda extension package"""


try:
    # Import functions in sporco_cuda.util and sporco_cuda.cbpdn into
    # namespace of sporco.cuda module
    from sporco_cuda.util import *
    from sporco_cuda.cbpdn import *
    # Flag indicating successful import
    have_cuda = True
except ImportError:
    # Flag indicating unsuccessful import
    have_cuda = False
    # Allow call to device_count when import fails
    def device_count():
        return 0
