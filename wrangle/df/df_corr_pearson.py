def df_corr_pearson(data, y, excluded_list=False):
    '''One-by-One Correlation

    Based on the input data, returns a dictionary where each column
    is provided a correlation coefficient and number of unique values.
    Columns with non-numeric values will have None as their coefficient.

    data : dataframe
        A Pandas dataframe with the data
    y : str
        The prediction variable
    excluded_list : bool
        If True, then also a list will be returned with column labels for
        non-numeric columns.

    '''

    out = {}
    category_columns = []

    for col in data.columns:
        try:
            out[col] = [data[y].corr(data[col]), len(data[col].unique())]
        except TypeError:
            out[col] = [None, len(data[col].unique())]
            category_columns.append(col)

    if excluded_list:
        return out, category_columns
    else:
        return out
