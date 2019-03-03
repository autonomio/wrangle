import pandas as pd
import numpy as np
import statsmodels.api as sm


def col_corr_ols(data, x, y, destructive=False):

    '''1-by-1 correlation with OLS for categorical values as strings.

    Takes in a single column 'x' of string values representing
    categories and performs an OLS test against a continuous
    'y' column. Returns an ordinal category representation (the greater
    the value, the higher the correlation coefficient.)

    data : dataframe
        A Pandas dataframe with the data
    x : str
        A feature with categorical/string values.
    y : str
        The truth variable. Has to be continuous.
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.

    TODO: Mark low correlation coefficients labels all as 0.
    '''

    if destructive is False:
        data = data.copy(deep=True)

    y = data[y].values
    dummies = pd.get_dummies(data[x]).values

    model = sm.OLS(y, dummies)
    results = model.fit()
    r = results.summary2()

    ordinal = list(np.argpartition(r.tables[1]['Coef.'], 0) + 1)

    data[x] = data[x].astype('category')
    data[x].cat.categories = ordinal

    return data
