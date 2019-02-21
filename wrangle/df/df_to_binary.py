import pandas as pd

from .df_rename_cols import df_rename_cols
from .df_impute_nan import df_impute_nan
from ..col.col_to_binary import col_to_binary
from ..col.col_fill_nan import col_fill_nan


def df_to_binary(data, y):

    '''Sohot DataFrame One Hot Encoding

    DANGER ZONE!!! this is where things get strange...

    WHAT: A way to one hot encode an entire dataframe
    minus the outcome variable of course. Handles automatically
    nans, categorical labeling, etc. Yes, everything.

    HOW: sohot(titanic,'Survived')

    INPUT: A pandas dataframe.

    OUTPUT: A one hot encoded transformation of the dataframe.
    That's right, pure binary goodness...gonna make your machine so hot!

    '''

    ind_var = data[y]

    temp = pd.DataFrame()

    for col in data.columns:
        if col != y:
            # handle columns with no nans
            if data[col].isnull().sum() == 0:
                if data[col].dtype != 'O':
                    temp = _concat_df(data[col], temp)

            # handle columns with nans
            else:
                if data[col].dtype != 'O':
                    imputed = df_impute_nan(data[col])
                    temp = _concat_df(imputed, temp)
                else:
                    filled = col_fill_nan(data, col, 0)
                    labeled = pd.Categorical(filled).codes
                    temp = _concat_df(labeled, temp)

    temp = df_rename_cols(temp)
    temp = temp.astype(int)
    temp = pd.concat([ind_var, temp], axis=1)

    return temp


def _concat_df(data, temp):

    oh = col_to_binary(data)
    temp = pd.concat([temp, oh], axis=1)

    return temp
