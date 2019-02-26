from .array import *
from .col import *
from .df import *

# this is temporary for backward compatibility
from .array.array_random_shuffle import array_random_shuffle as shuffle

del utils, array, col, df

__version__ = "0.6"
