from pandas import get_dummies
from .df_count_uniques import df_count_uniques


def df_to_multilabel(data, max_uniques=30, ignore_y=None, destructive=False):

    '''Transforms string or other values to multi-class (n-d) categorical.
    Generally it might be a good idea to handle NaN values first, but in
    case NaN values are present, they are converted into to their own category.

    EXAMPLE USE:
    ------------

        to_multiclass(data)

    PARAMS:
    -------

    data : dataframe
        A pandas dataframe with the data to be transformed
    max_uniques : int
        A threshold at which a column is no longer converted to a category
    ignore_y : str
        The string column name for y value
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.'''

    if destructive is False:
        data = data.copy(deep=True)

    keys = list(df_count_uniques(data).index)

    if ignore_y is not None:
        keys.remove(ignore_y)

    for key in keys:
        if len(data[key].unique()) <= max_uniques:
            data = data.merge(get_dummies(data[key]),
                              left_index=True,
                              right_index=True)
            data.drop(key, axis=1, inplace=True)

    return data
