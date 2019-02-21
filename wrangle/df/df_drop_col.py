def df_drop_col(data, cols, destructive=False):

    '''Drops one or more columns from a dataframe'''

    if destructive is False:
        data = data.copy(deep=True)

    data.drop(cols, axis=1, inplace=True)

    return data
