def df_rename_col(data, col, rename_to, destructive=False):

    """Rename a single column

    data : pandas DataFrame
        Pandas dataframe with the column to be renamed.
    col : str
        Column to be renamed
    rename_to : str
        New name for the column to be renamed

    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.

    """

    if destructive is False:
        data = data.copy(deep=True)

    cols = list(data.columns)
    loc = cols.index(col)
    cols.insert(loc, rename_to)
    cols.remove(col)
    data.columns = cols

    return data
