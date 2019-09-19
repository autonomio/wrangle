import pandas as pd


def col_move_place(data, col, position='first', destructive=False):

    '''Moves column/s to being first or last column/s.

    df : pandas dataframe
        A pandas dataframe with the column/s to be moved
    col : str or list
        The column or columns to be moved
    position : str
        Either 'first' or 'last'
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.
    '''

    if destructive is False:
        data = data.copy(deep=True)

    if type(col) is str:
        col = [col]

    for col in col:
        temp_col = data[col]
        temp_col = pd.DataFrame(temp_col)
        data = data.drop(col, axis=1)

        if position == 'first':
            data = pd.merge(temp_col, data, left_index=True, right_index=True)
        else:
            data = pd.merge(data, temp_col, left_index=True, right_index=True)

    return data
