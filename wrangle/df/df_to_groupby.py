from ..utils.groupby_func import groupby_func


def df_to_groupby(data, by, func):

    '''Takes in a dataframe and returns it in a grouped by format.

    data : dataframe
        A pandas dataframe
    by : str
        The column by which the grouping is done
    func : str
        The function to be used for grouping by: 'median', 'mean', 'first',
        'last', 'std', 'mode', 'max', 'min', 'sum', 'random', 'freq', 'string'.
    '''

    temp = data.groupby(by)

    return groupby_func(data=temp, func=func)
