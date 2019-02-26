def col_to_buckets(data, col, cuts=5):

    '''Cut continuous feature into equally sized buckets

    USE: first['BUN_groups'] = test_cut(first, 'BUN', rounding=True)

    data : pandas DataFrame
        The dataframe with the data.
    col : str
        The column with thec continuous values.
    cuts : int
        Number of buckets to create.
    rounding : bool
        If True, then values will be rounded to zero decimal.

    '''

    out = []
    
    temp = pd.cut(data[col], cuts)
    for i in temp:
        left = i.left
        right = i.right
        if rounding is True:
            left = int(left)
            right = int(right)
        out.append(str(left) + ' to ' + str(right))

    return out

