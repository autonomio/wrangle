def df_resample_stratified(data, strat_col, n=None, destructive=False):

    '''Stratified sampling of a DataFrame

    data : pandas dataframe
        A dataframe with one or more columns to be renamed.
    strat_col : str
        The column with categorical values that will be used
        for stratification.
    n : None or int
        If int value, then that many samples will be present for each
        class. Otherwise the number of samples for the label with the
        lowest number of observations (samples) is used.
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.
    '''

    if destructive is False:
        data = data.copy(deep=True)

    if n is None:
        n = data[strat_col].value_counts().values[-1]

    temp = data.groupby(strat_col, group_keys=False)
    return temp.apply(lambda x: x.sample(min(len(x), n)))
