import pandas as pd
from ..utils.groupby_func import groupby_func


def col_resample_interval(data, x, dt_col, mode='first', freq=60):

    '''Resample a dataframe by creating intervals based on a datetime
    column.

    data : pandas dataframe
        A dataframe with the values and a datetime column
    x : str
        The column where the actual data is
    dt_col : str
        The column with the datetime values
    mode : str
        The choices are 'median', 'mean', 'mode', 'first', 'last', 'std',
        'mode', 'max', 'min', 'sum', 'random', 'freq'
    freq : int
        Number of minutes per sample as int or a string 'quarter', 'half',
        'full' (days), 'week', 'month' (30 days), 'year'.
    '''

    if isinstance(freq, str):

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
