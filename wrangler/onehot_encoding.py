from numpy import array, zeros, arange


def onehot(data):

    '''One Hot Encoding

    WHAT: Transform a feature in to binary columns.

    HOW: onehot(df.col)

    INPUT: An array, series, or list

    OUTPUT: Multiple columns of binary values that
    represent the input values.

    HOW TO REVERSE ONE-HOT ENCODING:

    np.argmax(data, axis=1)

    '''

    a = array(data)
    out = zeros((a.size, max(a) + 1))
    out[arange(a.size), a] = 1

    return out
