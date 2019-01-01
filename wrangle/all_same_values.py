from numpy import unique


def all_same_values(x):

    '''Checks if all values in a column
    have the same value. This can be detrimental
    to a deep learning model.'''

    for i in range(x.shape[1]):
        if len(unique(x[:, i])) <= 1:
            print("#%d column has all same values" % i)
