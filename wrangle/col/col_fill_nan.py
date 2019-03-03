def col_fill_nan(data, cols, fill_with=0):

    '''
    Fills nans in a specific column or a range of
    columns, using np.isnull() through pandas.

    This is for the cases where some nans are dropped,
    but some nans are converted in to zero (or other)
    first.

    '''

    if data[cols].dtype == 'O':
        fill_with = str(fill_with)

    if isinstance(cols, list) is False:
        cols = [cols]
    for col in cols:
        cols = [cols]
        data = data[col].fillna(fill_with)

    return data
