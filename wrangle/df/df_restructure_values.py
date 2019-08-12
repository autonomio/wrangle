def df_restructure_values(data, structure='list', destructive=False):

    '''Takes in a dataframe, and restructures
    the values so that the output dataframe consist
    of columns where the value is coupled with the
    column header.

    data | DataFrame | a pandas dataframe
    structure | str | 'list', 'str', 'tuple',  or 'dict'
    destructive | bool | if False, the original dataframe will be retained

    '''

    if destructive is False:
        data = data.copy(deep=True)

    for col in data:

        if structure == 'list':
            data[col] = [[col, i] for i in data[col]]
        elif structure == 'str':
            data[col] = [col + ' ' + i for i in data[col].astype(str)]
        elif structure == 'dict':
            data[col] = [{col: i} for i in data[col]]
        elif structure == 'tuple':
            data[col] = [(col, i) for i in data[col]]

    return data
