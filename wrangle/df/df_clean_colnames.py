import pandas as pd


def df_clean_colnames(data, destructive=False):

    '''
    data : dataframe
         A pandas dataframe with the data to be transformed
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.

    TODO: (do we) need to add support for ordered/ordinal categoricals?

    '''

    if destructive is False:
        data = data.copy(deep=True)

    temp = pd.Series(data.columns)

    # remove special characters
    temp = temp.str.replace(r'[^A-Za-z0-9]+', ' ')

    # remove multiple spaces
    temp = temp.str.replace(r' +', ' ')

    # convert to lower case
    temp = temp.str.lower()

    # replace spaces
    temp = temp.str.replace(' ', '_')

    # remove preceding/trailing underscore
    temp = temp.str.replace('_$', '')
    temp = temp.str.replace('^_', '')

    data.columns = temp

    return data
