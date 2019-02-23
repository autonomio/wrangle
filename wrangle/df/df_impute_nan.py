import wrangle as wr


def df_impute_nan(data,
                  cols='all',
                  impute_mode='mean_by_std',
                  destructive=False):

    '''Impute NaN values in a dataframe

    Provides five different options for imputing nan
    values within a a series / array of data.

    data : DataFrame
        A pandas dataframe with the data.
    cols : 'all' or list
        By default all columns will be imputed. Alternatively
        accepts a list of columns as input.
    impute_mode : str
        The default is 'mean_by_std', with other
        options 'mean', 'median', 'mode', and 'common'.
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.

    '''

    if destructive is False:
        data = data.copy(deep=True)

    if cols == 'all':
        cols = data.columns

    for col in cols:
        try:
            data[col] = wr.col_impute_nan(data[col])
        except TypeError:
            pass

    return data
