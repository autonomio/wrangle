import pandas as pd


def generate_datetime(data, start, end, freq):

    '''DATETIME GENERATOR
    Takes in a series or dataframe, and based on the length creates
    a datetime series with a desired frequency. It's important to
    provide the start and end time precisely, together with the time
    unit, as otherwise there will be a mismatch of the valuesself.
    1.USE
    =====
    generate_datetime(data, '1999-02-12','1999-03-05', 'hour')
    This example will generate datetimes with hourly frequency starting
    from midnight on the 12th of February (1999) up until midnight of
    5th of March.
    generate_datetime(data, '1999-02-12-09:00','1999-03-05', 'hour')
    This example will do the same, but will start on 9:00 am instead.
    start :: the startime of the data (first observation timestamp)
    end :: the endtime of the data (last observation timestamp)
    freq :: This should be the frequency of observations in the dataset.
            Can be either pd.date_range frequency parameter, or:
            'year', 'month', 'day', 'hour', 'minute', 'second'
    '''

    if freq == 'year':
        freq = '365D'
    elif freq == 'month':
        freq = '30D'
    elif freq == 'day':
        freq = '1D'
    elif freq == 'hour':
        freq = '60Min'
    elif freq == 'minute':
        freq = '1Min'
    elif freq == 'second':
        freq = '1S'

    out = pd.Series(pd.date_range(start, end, freq=freq))

    if len(out) > len(data):
        print("Too many observation for the selected parameters.")
    elif len(out) < len(data):
        print("Not enough observations for the selected parameters.")

    return out
