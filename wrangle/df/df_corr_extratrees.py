import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


def df_corr_extratrees(data, y):

    ''''Extra Trees Classifier based correlation.

    data : pandas dataframe
        The dataset to perform the correlation on
    y : str
        Column name for a categorical feature such as

    '''

    x = data.drop(y, 1).values
    labels = data.drop(y, 1).columns
    y = data[y].values

    reg = ExtraTreesClassifier(max_depth=3, n_estimators=100)

    reg.fit(x, y)

    return pd.Series(data=reg.feature_importances_,
                     index=labels).sort_values(ascending=False)
