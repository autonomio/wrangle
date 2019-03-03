def df_find_nan(data):

    '''
    Finds nans and provides a dataframe with the column names and fraction of
    nans in the column.

    '''

    temp = data.transpose().isnull()
    temp = temp.transpose().describe()
    temp = temp.transpose().drop('count', axis=1)
    temp.unique = temp.unique == 1
    temp.freq = temp.freq / len(data)
    temp.freq = temp.freq - temp.top
    temp = temp.drop('top', axis=1)
    temp.columns = ['no_nans', 'quality']

    # converting negative values to positive
    temp_list = []

    for value in temp['quality']:
        if value < 0:
            temp_list.append(1 - value - 1)
        else:
            temp_list.append(value)

    temp['quality'] = temp_list

    return temp
