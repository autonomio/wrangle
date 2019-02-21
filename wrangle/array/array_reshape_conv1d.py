def array_reshape_conv1d(x):

    '''Reshapes regulard 2d array for Conv1D layer

    x : numpy array
        A 2-d numpy array to be respahed.
    '''

    out = x.reshape(x.shape[0], x.shape[1], 1)

    return out
