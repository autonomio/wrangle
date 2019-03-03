import numpy as np
import pandas as pd


def df_rescale_log(data, retain_cols=None, destructive=False):

    '''Transform dataframe to log1p.

    data : pandas dataframe
         A dataframe to be rescaled
    retain_cols : str or list
         The columns that should be excluded from rescaling
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.
    '''

    if destructive is False:
        data = data.copy(deep=True)

    if retain_cols is not None:
        data = data.drop(retain_cols, 1)
        temp = data[retain_cols]

    numeric = data.select_dtypes(include=['int', 'float']).columns

    data[numeric] = data[numeric].apply(np.log1p)

    if retain_cols is not None:
        return pd.merge(data, temp, left_index=True, right_index=True)
    else:
        return data
