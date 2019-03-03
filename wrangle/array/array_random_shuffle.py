from numpy import arange, random


def array_random_shuffle(x, y):

    '''Randomize two arrays in same order'''

    x_shape = x.shape
    y_shape = y.shape

    random_index = arange(len(x))
    random.shuffle(random_index)

    x = x[random_index]
    y = y[random_index]

    if (x_shape == x.shape) and (y_shape == y.shape):
        return x, y
    else:
        print("The output shapes do not match")
