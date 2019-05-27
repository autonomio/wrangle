import pandas as pd


def col_to_buckets(data, col, cuts=5, rounding=False):

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

        try:
            left = i.left
            right = i.right

            if rounding is True:
                left = int(left)
                right = int(right)

            out.append(str(left) + ' to ' + str(right))

        # where interval was not produced (value was nan etc.)
        except AttributeError:
            out.append('')

    return out
