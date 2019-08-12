def groupby_func(data, func):

    '''Streamlines the process of creating various
    groupby elements in Pandas. Takes in a dataframe
    and one of the supported functions as a string:

    data : pandas groupby
         A Pandas groupby object
    func : str
        The function to be used for grouping by

    'median'
    'mean'
    'first'
    'last'
    'std'
    'max'
    'min'
    'sum'
    'random'
    'freq'
    'string'
    'entropy'

    ...or you can simply input any custom function.

    '''

    import numpy as np
    import pandas as pd
    import scipy as sc

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
    elif func == 'string':
        out = data.apply(lambda x: "%s" % ' '.join(x))
        out = pd.DataFrame(out).reset_index()
    elif func == 'entropy':
        out = data.apply(lambda x: sc.stats.entropy(x)[0])
    elif callable(func):
        out = data.apply(func)

    if isinstance(out, type(pd.Series())):
        out = pd.DataFrame(out)
        out.columns = [1]

    out.reset_index(inplace=True)

    return out
