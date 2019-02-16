def df_corr_pearson(data, y, excluded_list=False):
    '''One-by-One Correlation

    Based on the input data, returns a dictionary and list where the
    dictionary contains pearson correlation coefficient and the list
    contains columns that are not in numeric form.

    data : dataframe
        A Pandas dataframe with the data
    y : str
        The prediction variable

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
