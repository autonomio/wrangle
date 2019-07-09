def df_to_xy(data, y_col):

    '''Split dataframe into x and y arrays

    data : pandas dataframe
        A pandas dataframe with the column to be converted
    col : str
        The column with the y values
        '''

    return data.drop(y_col, 1).values, data[y_col].values
