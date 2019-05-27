def df_rename_cols(data, exclude=None, prefix='C', destructive=False):

    '''Generate sequential alphabetic names for columns
    Takes in a dataframe and generates alphabetic
    column names automatically.
    data : pandas dataframe
        A dataframe with one or more columns to be renamed.
    exclude : str
        A column, such as the prediction feature, to be excluded.
    prefix : str
        The prefix to be appended before the sequential number
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.
    '''

    if destructive is False:
        data = data.copy(deep=True)

    if exclude is not None:
        data = data.drop(exclude, 1)
        temp = data[exclude].values

    no_of_cols = data.shape[1]
    l = []

    for i in range(no_of_cols):
        l.append(prefix + str(i))

    data.columns = l

    if exclude is not None:
        data[exclude] = temp

    return data
