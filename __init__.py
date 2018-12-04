"""
Utility for function
--------------------
  compose
  pip
  tee
  identity
  over_args

Utility for recursive function
------------------------------
  Y
  recursiveExtender
  memoize
  trace

Depricated
----------
  (For iterator. Use func_helper.iterator package)
    mapping
    filtering
    reducing
"""

from .func_helper import compose, pip, tee, identity, over_args
from .func_helper import memoize, trace, Y, recursiveExtender
