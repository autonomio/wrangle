import numpy as np


def array_to_multilabel(y):

    '''Converts a 1-d array with multiclass
    integer labels into corresponding 2-d multilabel
    version.'''

    try:
        y = y.astype(int)
    except ValueError:
        return "Could not convert array to int first."

    size = len(y)
    b = np.zeros((size, max(y) + 1))
    b[np.arange(size), y] = 1

    return b
