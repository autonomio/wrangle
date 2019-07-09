def col_to_biclass(data, col, true_value, destructive=False):

    '''Takes in a dataframe column with two classes in
    string form, and converts it to integer binary class.

    data : pandas dataframe
        A pandas dataframe with the column to be converted
    col : str
        The column with the multiclass values
    true_value : str
        The column value to be considered as True
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.
    '''

    if destructive is False:
        data = data.copy(deep=True)

    data[col] = (data[col] == true_value).astype(int)

    return data
