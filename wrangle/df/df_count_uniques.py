def df_count_uniques(data):
    '''Returns the number of unique values in the columns of a dataframe'''

    return data.apply(lambda x: len(x.unique()))
