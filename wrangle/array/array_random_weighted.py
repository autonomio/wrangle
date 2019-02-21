import numpy as np


def array_random_weighted(x, weights):

    '''EXAMPLE:

    elements = ['one', 'two', 'three']
    weights = [0.2, 0.3, 0.5]
    array_random_weighted(elements, weights)

    '''

    return np.random.choice(x, p=weights)
