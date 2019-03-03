def df_drop_weak(data,
                 y,
                 min_correlation=.2,
                 method='pearson',
                 drop_object=True,
                 destructive=False):

    '''Keeps features that have a degree of correlatio with the prediction
    feature. Drops other columns from the dataframe.

    data : dataframe
        A Pandas dataframe with the data
    y : str
        The prediction variable
    min_correlation : float
        The minimum absolute accepted correlation value
    method : str
        One of 'spearman', 'pearson', or 'kendall'
    drop_object : bool
        If object dtype columns should be dropped
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.

    '''

    if destructive is False:
        data = data.copy(deep=True)

    if drop_object is True:
        data = data.select_dtypes(exclude=['object'])

    temp = data.corr(method=method)[y].apply(abs)
    keys = list(temp.index)

    for key in keys:
        if temp[key] < min_correlation:
            data.drop(key, axis=1, inplace=True)

    return data
