from numpy import arange, random


def shuffle(x, y):

    '''Randomize two arrays in same order'''

    random_index = arange(len(x))
    random.shuffle(random_index)

    x = x[random_index]
    y = y[random_index]

    return x, y
