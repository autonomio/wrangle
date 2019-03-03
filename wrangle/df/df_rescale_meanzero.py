import pandas as pd


def df_rescale_meanzero(data, retain=None):

    '''ZERO MEAN SCALER
    Takes in a dataframe or series, and rescales features
    so that mean for each feature becomes 0 and standard
    deviation becomes 1.
    USE
    ===
    mean_zero(df[['Age','Fare','Pclass']],retain='Pclass')
    PARAMETERS
    ==========
    data :: pandas dataframe or series object
    retain :: if some column should be excluded from rescaling
    '''
    # avoiding transformation of y, labels, etc
    try:
        data = data.copy(deep=True)
    except TypeError:
        data = pd.DataFrame(data)
        is_array = True

    try:
        col_list = list(data.columns)
    except AttributeError:
        col_list = list(pd.DataFrame(data.columns))

    if retain is not None:
        col_list.remove(retain)

    numeric = data.select_dtypes(exclude=['int', 'float'])
    numeric = list(numeric.columns)

    for col in numeric:
        col_list.remove(col)

    for col in col_list:

        # storing the temp values
        data_mean = data[col].mean(axis=0)
        data_std = data[col].std(axis=0)

        # transforming the data
        col_data = data[col] - data_mean
        data[col] = col_data / data_std

    try:
        is_array is True
        return data.values
    except UnboundLocalError:
        return data
