from pandas import Categorical
from .df_count_uniques import df_count_uniques


def df_to_multiclass(data, max_uniques=30, ignore_y=None, destructive=False):

    '''Transforms string or other values to multi-label (1d) categorical.
    Generally it might be a good idea to handle NaN values first, but in
    case NaN values are present, they are converted into -1.

    data : dataframe
        A pandas dataframe with the data to be transformed
    max_uniques : int
        A threshold at which a column is no longer converted to a category
    ignore_y : str
        The string column name for y value
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.

    TODO: (do we) need to add support for ordered/ordinal categoricals?'''

    if destructive is False:
        data = data.copy(deep=True)

    temp = df_count_uniques(data)
    keys = list(temp.index)

    if ignore_y is not None:
        keys.remove(ignore_y)

    for key in keys:
        if temp[key] <= max_uniques:
            data[key] = Categorical(data[key]).codes

    return data
