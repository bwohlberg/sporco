from __future__ import absolute_import
import warnings

import sporco.fista.fista

warnings.simplefilter('always', DeprecationWarning)
warnings.warn('Module sporco.fista is deprecated; please use sporco.pgm',
              DeprecationWarning)
warnings.simplefilter('default', DeprecationWarning)
