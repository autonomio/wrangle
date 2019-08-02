import pandas as pd


def col_to_multilabel(data,
                      col,
                      colnames=None,
                      extended_colname=False,
                      extended_separator='_'):

    '''Takes in a dataframe column with multiple labels and converts it
    into several columns with binary labels.

    df : pandas dataframe
        A pandas dataframe with the column to be converted
    col : str
        The column with the multiclass values
    colnames : list
        A list of column names for the new values from 0 onwards
    extended_colname : bool
        If True, then the original column name will be a prefix for the
        new column names.
    extended_separator : str
        The separator to be used in case `extended_colname` is in play.
    '''

    data = data.copy(deep=True)

    temp_cols = pd.get_dummies(data[col])
    data = data.drop(col, axis=1)

    if isinstance(colnames, list):
        temp_cols.columns = colnames

    if extended_colname is True:
        temp_cols.columns = [col + extended_separator + str(i) for i in temp_cols.columns]

    return pd.merge(data, temp_cols, left_index=True, right_index=True)
