import scipy.stats as sc


def col_drop_outliers(data,
                      col,
                      mode='zscore',
                      threshold=3,
                      destructive=False):

    '''Drop rows based on outliers in a given column. Note that
    this drops NaN automatically.

    data : pandas dataframe
        A pandas dataframe to be processed
    col : str
        Name of the column to be processed. Note that rows will be
        dropped for the whole dataframe.
    mode : str
        The mode to be used for outlier detection. Either 'zscore' or 'iqr'
    threshold : int
        The threshold to be used for IQR based outlier removal.

    '''

    # avoid destruction

    if destructive is False:
        data = data.copy(deep=True)

    data = data[data[col].isna() == False]

    if mode == 'zscore':
        data['zscore'] = sc.zscore(data[col].astype(float))
        data = data[data.zscore < 3][data.zscore > -3].drop('zscore', axis=1)

    if mode == 'iqr':
        data = data[data[col] < sc.iqr(data[col]) * threshold]

    return data
