import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def df_corr_randomforest(data, y, destructive=True):

    '''Random Forrest based correlation

    Takes in a DataFrame with one or more columns of numeric
    data in addition to 'y' which is continuous.

    data : dataframe
        A Pandas dataframe with the data
    y : str
        The prediction variable.
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.

    '''
    if destructive is False:
        data = data.copy(deep=True)

    x = data.drop(y, 1).values
    labels = data.drop(y, 1).columns
    y = data[y].values

    reg = RandomForestRegressor(max_depth=3, n_estimators=100)

    reg.fit(x, y)

    return pd.Series(data=reg.feature_importances_,
                     index=labels).sort_values(ascending=False)
