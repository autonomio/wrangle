from numpy import median, unique, mean


def array_detect_task(y):

    '''Detects the prediction task type based on an array
    containing the prediction task labels.'''

    try:
        y_cols = y.shape[1]
    except IndexError:
        y_cols = 1
    y_max = y.max()
    y_uniques = len(unique(y))

    if y_cols > 1:
        y_type = 'category'
        y_range = y_cols
        y_format = 'onehot'
    else:
        if y_max == 1:
            y_type = 'binary'
            y_range = y_cols
            y_format = 'single'
        elif mean(y) == median(y):
            y_type = 'category'
            y_range = y_uniques
            y_format = 'single'
        else:
            y_type = 'continuous'
            y_range = y.max() - y.min()
            y_format = 'single'

    return y_type, y_range, y_format
