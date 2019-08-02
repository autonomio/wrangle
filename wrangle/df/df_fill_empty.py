def df_fill_empty(data, fill_with):

    '''Finds and replaces any value in dataframe that only consist of
    whitespace. A common scenario is where you first fill empties
    with np.nan and then handle nans as you would otherwise do it.

    NOTE: this results in all columns to be converted into strings. You can use
    wrangle.df_to_numeric() to convert all back to numeric.

    data : Pandas Dataframe
        The dataframe to be processed
    fill_with: str
        Fill the values with a string.
    '''
    return data.astype(str).apply(lambda x: x.str.strip().replace('', fill_with))
