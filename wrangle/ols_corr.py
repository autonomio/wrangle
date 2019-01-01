import pandas as pd
import numpy as np
import statsmodels.api as sm


def ols_corr(data, y):
    '''OLS Based Correlation for Strings

    Takes in string values of categories, and ranks them based on
    strenght of relationship with 'y' and returns an ordinal
    category representation.


    TODO: there should be option to just use the coef as value as
    opposed to integer or some other weighted option as well :) and
    maybe std could be used as confidence of some sort?
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
