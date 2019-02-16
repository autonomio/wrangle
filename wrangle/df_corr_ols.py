import pandas as pd
import numpy as np
import statsmodels.api as sm


def df_corr_ols(data, y):

    '''1-by-1 correlation with OLS for categorical values as strings.

    Takes in string values of categories, and ranks them based on
    strenght of relationship with 'y' and returns an ordinal
    category representation.

    data : dataframe
        A Pandas dataframe with the data
    y : str
        The prediction variable
    '''

    y = np.array(y)
    dummies = pd.get_dummies(data).values

    model = sm.OLS(y, dummies)
    results = model.fit()
    r = results.summary2()

    ordinal = list(np.argpartition(r.tables[1]['Coef.'], 0) + 1)

    temp = data.astype('category')
    temp.cat.categories = ordinal

    return temp.astype(float)
