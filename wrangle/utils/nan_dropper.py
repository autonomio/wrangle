def nan_dropper(data, treshold=.9):

    '''
    Drops nans in a specific column or a range of
    columns, using np.isnull() through pandas.

    The treshold sets the % at which the whole column
    is droped (nan % of all values in the column).

    '''

    import wrangle as wr

    temp = wr.df_find_nan(data)
    temp = temp.transpose()

    for col in temp:
        if temp[col][1] < treshold:
            data = data.drop(col, axis=1)
            nan_rate = ((1 - temp[col][1]) * 100)
            print("DROPPED (%.2f%% nans): %s" % (nan_rate, col))

    data.dropna(inplace=True)

    return data
