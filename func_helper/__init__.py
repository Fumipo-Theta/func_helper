"""
func_helper
-----------
  compose
  pip
  tee
  identity
  over_args

Recursive function utility
--------------------------
  Y
  recursiveExtender
  memoize
  trace

Utility for iterator
--------------------
  mapping(deprecated)
  filtering(deprecated)
  reducing(deprecated)

Transducer
----------
  transducer

Iteratore
---------
  iterator

Dictionary
----------
  dictionary
"""

from .func_helper import compose, pip, tee, mapping, filtering, reducing, identity, over_args, memoize, trace, Y, recursiveExtender
from . import transducer
from . import iterator
from . import dictionary
