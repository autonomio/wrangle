import pandas as pd


def col_to_binary(data, col, func='median', destructive=False):

    '''Takes in a continuous feature and transforms into
    a binary class.

    df : pandas dataframe
        A pandas dataframe with the column to be converted
    col : str
        The column with the multiclass values
    func : str, float, or int
        'mean','median','mode',int (ge), string for
        interquartile range for binary conversion. 'cat_string'
        for converting strings in to categorical labels, and
        'cat_int' for doing the same with integer values.
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.
    '''

    if destructive is False:
        data = data.copy(deep=True)

    # if user input 'int' then function will be "greater than value"
    # if user input 'float' then function will be IQR range

    # below is for case where prediction is true or false
    # but the y-feature is in different format (e.g continuous)

    if func == 'mean':
        data[col] = data[col] >= data[col].mean()
    elif func == 'median':
        data[col] = data[col] >= data[col].median()
    elif func == 'mode':
        data[col] = data[col] >= data[col].mode()[0]
    elif isinstance(func, int):
        data[col] = data[col] >= func
    elif isinstance(func, float):
        data[col] = data[col] >= data[col].quantile(func)

    # below is for case where the y-feature is converted in
    # to a categorical, either if it's a number or string.

    elif func == 'cat_string':
        data[col] = pd.Categorical(data[col])
        data[col] = data[col].cat.codes

    elif func == 'cat_numeric':
        data[col] = pd.qcut(data[col], 5, duplicates='drop')
        data[col] = data[col].cat.codes

    # for cases when y-feature is already in the format
    # where the prediction output will be.

    elif func == 'none':
        pass

    return data
