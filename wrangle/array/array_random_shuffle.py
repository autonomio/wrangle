def array_random_shuffle(x, y=None, multi_input=False):

    '''Shuffles single or multi-input data. Note that
    only x can be multi-input and x has to be more than one
    column of data.

    x | array or list | the x data to be shuffled
    y | array | the y data to be shuffled (optional)
    multi_input | bool | set to True if multi-input model data
    '''

    import numpy as np

    rng = np.random.default_rng()
    state = rng.bit_generator.state

    # data input is list but multi_input is not set to True
    if isinstance(x, list) and multi_input == False:

        raise TypeError("For multi-input x, set multi_input to True")

    # data input is list and multi_input is set to True
    elif isinstance(x, list) and multi_input == True:

        x_out = []

        for ar in x:

            rng.bit_generator.state = state
            rng.shuffle(ar, axis=0)
            x_out.append(ar)

        x = x_out

        if y is not None:
            rng.bit_generator.state = state

            try:
                y.shape[1]
                rng.shuffle(y, axis=0)
            except IndexError:
                rng.shuffle(y)

            return x, y

        else:
            return x

    # data input is assumably an array (not multi-input)
    elif isinstance(x, list) == False:

        rng.bit_generator.state = state
        rng.shuffle(x, axis=0)

        if y is not None:
            rng.bit_generator.state = state

            try:
                y.shape[1]
                rng.shuffle(y, axis=0)
            except IndexError:
                rng.shuffle(y)

            return x, y

        else:
            return x
