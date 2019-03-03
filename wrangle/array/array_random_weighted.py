import numpy as np


def array_random_weighted(x, weights, size):

    '''Weighted random pick from array

    EXAMPLE:

    elements = ['one', 'two', 'three']
    weights = [0.2, 0.3, 0.5]
    array_random_weighted(elements, weights)

    x : array
         Numpy array to draw the sample from
    weights : list or 'normal'
         Either a list of weights or 'normal' for
         normal distribution random pick.
    size : int
    '''

    if isinstance(weights, str):

        if weights == 'normal':
            distribution = np.random.normal(size=size)

        elif weights == 'poisson':
            distribution = np.random.poisson(size=size)

        elif weights == 'laplace':
            distribution = np.random.laplace(size=size)

        weights = np.random.dirichlet(distribution + 10, size=1)
        weights = weights.reshape((size,))

    return np.random.choice(x, p=weights, size=size)
