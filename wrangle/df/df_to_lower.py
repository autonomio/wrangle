def df_to_lower(data, cols=None):

    '''Convert all string values to lowercase

    data : pandas dataframe
        The dataframe to be cleaned
    cols : str, list, or None
        If None, an attempt will be made to turn
        all string columns into lowercase.

    '''

    if isinstance(cols, str):
        cols = [cols]
    elif cols is None:
        cols = data.columns

    for col in cols:
        try:
            data[col] = data[col].str.lower()
        except AttributeError:
            pass

    return data
