import string


def col_to_split(data, col, col_names=None, destructive=False):

    '''Splits a single column to multiple columns. Replaces
    the original column so use destructive=True accordingly.

    Note that each row needs to have the same number of items
    after splitting.

    data : pandas dataframe
        The dataframe to be modified.
    col : str
        column to be split
    col_names : list
        Optional list of names of the new columns.

    '''

    if destructive is not False:
        data = data.copy(deep=True)

    org = data[col].str.split()

    if col_names is None:
        col_names = []
        for i in range(len(org[0])):
            col_names.append(col + '_' + string.ascii_lowercase[i])

    for i in range(len(org[0])):
        data[col_names[i]] = [ii[i] for ii in org]

    data.drop(col, axis=1, inplace=True)

    return data
