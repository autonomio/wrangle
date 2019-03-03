def df_corr_ols(data, y, destructive=False):

    '''1-by-1 correlation with OLS for categorical values as strings.

    Takes in a single column DataFrame where some columns are string
    values of categories. Then performs an OLS test individually on
    values of column returning an ordinal category representation (the greater
    the value, the higher the correlation coefficient.)

    data : dataframe
        A Pandas dataframe with the data
    y : str
        The prediction variable.
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.

    '''

    import wrangle as wr
    import warnings

    warnings.simplefilter('ignore')

    if destructive is False:
        data = data.copy(deep=True)

    for col in data.columns:
        try:
            wr.col_corr_ols(data, col, y, destructive=True)
        except ValueError:
            pass

    return data
