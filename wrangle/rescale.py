import math
import numpy as np
import pandas as pd


def max_rescale(values, scale=1, to_int=False):

    '''MinMax Rescaler

    WHAT: rescales a set of values on to a fixed scale.

    HOW: max_rescale([10,6,2],1)

    INPUT: an array, list or Series

    OUTPUT: an array with rescaled values.

    '''

    multiplier = scale / np.array(values).max().astype(float)
    new_shape = np.array(values) * multiplier

    if to_int is True:

        l = []

        for value in new_shape:
            l.append(int(math.ceil(value)))

        return np.array(l)

    else:
        return new_shape


def mean_zero(data, retain=None):

    '''ZERO MEAN SCALER
    Takes in a dataframe or series, and rescales features
    so that mean for each feature becomes 0 and standard
    deviation becomes 1.
    USE
    ===
    mean_zero(df[['Age','Fare','Pclass']],retain='Pclass')
    PARAMETERS
    ==========
    data :: pandas dataframe or series object
    retain :: if some column should be excluded from rescaling
    '''
    # avoiding transformation of y, labels, etc
    try:
        data = data.copy(deep=True)
    except TypeError:
        data = pd.DataFrame(data)
        is_array = True

    try:
        col_list = list(data.columns)
    except AttributeError:
        col_list = list(pd.DataFrame(data.columns))

    if retain is not None:
        col_list.remove(retain)

    for col in col_list:

        # storing the temp values
        data_mean = data[col].mean(axis=0)
        data_std = data[col].std(axis=0)

        # transforming the data
        col_data = data[col] - data_mean
        data[col] = col_data / data_std

    try: 
        is_array == True
        return data.values
    except UnboundLocalError:
        return data
