import numpy as np

def groupby_func(data, func):

    '''Streamlines the process of creating various
    groupby elements in Pandas. Takes in a dataframe
    and one of the supported functions as a string:

    'median'
    'mean'
    'first'
    'last'
    'std'
    'mode'
    'max'
    'min'
    'sum'
    'random'
    'freq'

    All are standard Pandas functions, except 'random'
    and 'freq' are custom.

    '''


    if func == 'median':
        out = data.median()
    elif func == 'mean':
        out = data.mean()
    elif func == 'first':
        out = data.first()
    elif func == 'last':
        out = data.last()
    elif func == 'std':
        out = data.std()
    elif func == 'mode':
        out = data.mode()
    elif func == 'max':
        out = data.max()
    elif func == 'min':
        out = data.min()
    elif func == 'sum':
        out = data.sum()
    elif func == 'random':
        out = data.agg(np.random.choice)
    elif func == 'freq':
        out = data.agg(lambda x: x.value_counts().index[0])

    out = out.reset_index()

    return out