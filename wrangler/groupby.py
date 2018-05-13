import numpy as np


def group(data, by, func):

    '''GROUP BY
    Takes in a dataframe and returns it in a grouped by format.
    PARAMETERS
    ----------
    data :: a pandas dataframe
    by :: the column by which the grouping is done
    func ::
    '''

    temp = data.groupby(by)

    return groupby_func(data=temp, func=func)


def groupby_func(data, func):

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
