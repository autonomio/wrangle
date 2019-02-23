import pandas as pd


def col_check_allsame(data, col):

    '''Checks if all values in a column
    have the same value. This can be detrimental
    to a deep learning model.

    data : DataFrame
    col : str

    '''

    uniques = len(pd.unique(data[col]))

    if uniques == 1:
        return True
    else:
        return False
