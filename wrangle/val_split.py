from wrangle.shuffle import shuffle


def val_split(x, y, split, shuffled=True):

    '''VALIDATION SPLIT OF X AND Y
    Based on the Scan() parameter val_split
    both 'x' and 'y' are split.
    '''

    if shuffled is True:
        x, y = shuffle(x, y)

    len_x = len(x)
    limit = int(len_x * (1 - split))

    x_train = x[:limit]
    y_train = y[:limit]

    x_val = x[limit:]
    y_val = y[limit:]

    return x_train, y_train, x_val, y_val
