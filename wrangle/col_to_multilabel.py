from pandas import get_dummies, merge


def col_to_multilabel(df, col, colnames=None):

    '''Takes in a dataframe column with multiple labels and converts it
    into several columns with binary labels.

    df : pandas dataframe
        A pandas dataframe with the column to be converted
    col : str
        The column with the multiclass values
    colnames : list
        A list of column names for the new values from 0 onwards
    '''

    df = df.copy(deep=True)

    temp_cols = get_dummies(df[col])
    df = df.drop(col, axis=1)

    if isinstance(colnames, list):
        temp_cols.columns = colnames

    return merge(df, temp_cols, left_index=True, right_index=True)
