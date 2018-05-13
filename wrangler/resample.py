import pandas as pd
from .groupby import groupby_func


def equal_samples(data, col, sample_size):

    '''Resamples based on unique labels in a column.
    Results in a dataframe with equal number of rows
    per label in a given column.
    '''
    new_data = pd.DataFrame()

    for col_label in data[col].unique():
        sample = data[data[col] == col_label].sample(sample_size)
        new_data = new_data.append(sample)
    out = new_data

    return out


def intervals(data, x, dt_col, mode='first', freq=60):

    """BREAK TIMESERIES TO INTERVALS
    USE
    ===
    test = intervals(time_data, 'value','time_stamp', mode='min')
    PARAMETERS
    ----------
    data :: a dataframe with the values and a datetime column
    x :: the column with the value
    dt_col :: the column with the datetime values
    mode :: 'median', 'mean', 'mode', 'first', 'last', 'std', 'mode'
            'max', 'min', 'sum', 'random', 'freq'
    freq :: number of minutes per sample as int or a string 'quarter', 'half',
            'full' (days), 'week', 'month' (30 days), 'year'.
    """

    if type(freq) == type('s'):
        if freq == 'quarter':
            freq = 240
        elif freq == 'half':
            freq = 720
        elif freq == 'full':
            freq = 1440
        elif freq == 'week':
            freq = 10080
        elif freq == 'month':
            freq = 32400
        elif freq == 'year':
            freq = 525600

    freq = str(freq) + "Min"

    time_temp = data[[x, dt_col]].set_index(dt_col)
    time_temp = time_temp.groupby(pd.Grouper(freq=freq, label='right'))

    return groupby_func(data=time_temp, func=mode)
