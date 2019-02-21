import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def df_corr_randomforest(data, y):

    x = data.drop(y, 1).values
    labels = data.drop(y, 1).columns
    y = data[y].values

    reg = RandomForestRegressor(max_depth=3, n_estimators=100)

    reg.fit(x, y)

    return pd.Series(data=reg.feature_importances_,
                     index=labels).sort_values(ascending=False)
